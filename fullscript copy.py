import asyncio
import os
import sys
import json
import sqlite3
import math
import time
import threading
import platform
import glob
from array import array
from datetime import datetime
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel

from openai_realtime_client import RealtimeClient, AudioHandler, TurnDetectionMode
from llama_index.core.tools import FunctionTool

import numpy as np
import cv2
from flask import Flask, Response, request
import requests

try:
    import serial
except ImportError:
    serial = None

import torch
import torchaudio
if not hasattr(torchaudio, 'list_audio_backends'):
    torchaudio.list_audio_backends = lambda: ['soundfile']
if not hasattr(torchaudio, 'get_audio_backend'):
    torchaudio.get_audio_backend = lambda: 'soundfile'

import torchaudio.transforms as T
import sounddevice as sd
from speechbrain.inference.speaker import SpeakerRecognition
from insightface.app import FaceAnalysis
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# --- Group ID / robot memory settings ---
ROBOT_MEMORY_FILE = os.path.join("embeddings_db", "robotmemory")
FACE_SIM_THRESHOLD = 0.45
VOICE_SIM_THRESHOLD = 0.30
WAKE_UP_THRESHOLD = 0.3
SLEEP_TIMEOUT = 15
NN_SKIP_FRAMES = 30 # Increased to reduce CPU load/power draw

outputFrame = None
frame_lock = threading.Lock()
app = Flask(__name__)

# Saves all unique viewers of the camera stream endpoints (/video_feed)
# Each entry is a tuple: (ip, user_agent)
CAMERA_CHANNEL_USERS = set()
camera_channel_users_lock = threading.Lock()

GLOBAL_CAP = None
GLOBAL_CAP_LOCK = threading.Lock()
LATEST_FRAME = None
FRAME_READ_LOCK = threading.Lock()

# Removed camera_worker, functionality moved to unified_vision_worker

def open_camera(width=320, height=240):
    """
    Best-effort OpenCV camera open that is more reliable on Linux/Raspberry Pi.
    Returns an opened cv2.VideoCapture or None.
    """
    is_linux = platform.system().lower() == "linux"
    backend = cv2.CAP_V4L2 if is_linux and hasattr(cv2, "CAP_V4L2") else None

    # Build candidate list from /dev/video* (open first possible camera).
    candidates = []
    try:
        devs = sorted(glob.glob("/dev/video*"))
        for d in devs:
            base = os.path.basename(d)
            if base.startswith("video"):
                suf = base[5:]
                if suf.isdigit():
                    candidates.append(int(suf))
    except Exception:
        pass
    # De-duplicate while preserving order
    seen = set()
    candidates = [i for i in candidates if not (i in seen or seen.add(i))]

    for idx in candidates:
        cap = None
        try:
            cap = cv2.VideoCapture(idx, backend) if backend is not None else cv2.VideoCapture(idx)
            if cap is not None and cap.isOpened():
                cap.set(3, width)
                cap.set(4, height)
                return cap
        except Exception:
            pass
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass
    return None

robot_state = {
    "status": "BOOTING...",
    "subtext": "Loading Group Logic...",
    "color": (255, 255, 255),
    "mode": "AWAKE" # Always awake
}

# Group ID models (inited when starting group logic)
NATIVE_RATE = 44100
resampler = None
vad_model = None
speaker_model = None
face_app = None
hands_detector = None
_group_models_loaded = False

def init_group_models():
    """Load face/voice/hand models for Group ID (ultv1)."""
    global NATIVE_RATE, resampler, vad_model, speaker_model, face_app, hands_detector, _group_models_loaded
    if _group_models_loaded:
        return
    print(">>> [INIT] Загрузка моделей Group ID...")
    try:
        dev_info = sd.query_devices(kind='input')
        NATIVE_RATE = int(dev_info['default_samplerate'])
    except Exception:
        NATIVE_RATE = 44100
    resampler = T.Resample(NATIVE_RATE, 16000)
    vad_model, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', trust_repo=True)

    # Compatibility shim: older SpeechBrain versions call hf_hub_download(use_auth_token=...),
    # but newer huggingface_hub renamed it to token=...
    try:
        import inspect
        import huggingface_hub  # type: ignore

        sig = inspect.signature(huggingface_hub.hf_hub_download)
        if "use_auth_token" not in sig.parameters:
            _orig_hf_hub_download = huggingface_hub.hf_hub_download

            def _hf_hub_download_compat(*args, use_auth_token=None, **kwargs):
                if use_auth_token is not None and "token" not in kwargs:
                    kwargs["token"] = use_auth_token
                return _orig_hf_hub_download(*args, **kwargs)

            huggingface_hub.hf_hub_download = _hf_hub_download_compat
    except Exception:
        pass

    # Load speaker model (HuggingFace) with fallbacks.
    # If HF hosting changes or network is unavailable, we keep Group ID alive without voice-ID.
    speaker_model = None
    speaker_sources = [
        # common SpeechBrain repos (try in order)
        "speechbrain/spkrec-ecapa-voxceleb",
        "speechbrain/spkrec-xvect-voxceleb",
    ]
    # First try local cache dir if it exists (avoids network)
    try:
        local_savedir = Path("tmp_model")
        if local_savedir.exists():
            speaker_sources.insert(0, str(local_savedir))
    except Exception:
        pass

    last_err = None
    for src in speaker_sources:
        try:
            speaker_model = SpeakerRecognition.from_hparams(source=src, savedir="tmp_model")
            print(f">>> [INIT] Speaker model loaded from: {src}")
            break
        except Exception as e:
            last_err = e
            continue
    if speaker_model is None:
        print(f">>> [INIT] Speaker model unavailable (voice-ID disabled): {last_err}")
    face_app = FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider'])
    face_app.prepare(ctx_id=-1, det_size=(320, 320))
    hands_detector = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.5)
    _group_models_loaded = True
    print(">>> [INIT] Group ID модели загружены.")

def load_robot_db():
    if os.path.exists(ROBOT_MEMORY_FILE):
        try:
            with open(ROBOT_MEMORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_robot_db(db):
    with open(ROBOT_MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False)

def get_next_robot_id():
    db = load_robot_db()
    return f"User_{len(db) + 1:05d}"

def convert_audio(audio_np):
    waveform = torch.from_numpy(audio_np).float()
    if len(waveform.shape) == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.shape[0] != 1:
        waveform = waveform.t()
    return resampler(waveform)

def get_voice_embedding(audio_data):
    if speaker_model is None:
        return None
    wav_16k = convert_audio(audio_data)
    emb = speaker_model.encode_batch(wav_16k)
    return (emb.squeeze().cpu().numpy() / np.linalg.norm(emb.squeeze().cpu().numpy())).tolist()

def is_silence(audio_chunk):
    # In SLEEP mode we want to wake up on *any* audible input.
    # First do a simple energy gate; if there's any non-trivial level, treat as non-silence.
    try:
        level = float(np.max(np.abs(audio_chunk)))
        if not math.isnan(level) and level > 1e-5:
            return False
    except Exception:
        # If anything goes wrong, fall back to VAD below.
        pass
    wav_16k = convert_audio(audio_chunk)
    target = 512
    if wav_16k.shape[-1] > target:
        wav_16k = wav_16k[..., :target]
    elif wav_16k.shape[-1] < target:
        wav_16k = torch.nn.functional.pad(wav_16k, (0, target - wav_16k.shape[-1]))
    with torch.no_grad():
        conf = vad_model(wav_16k, 16000).item()
    return conf < WAKE_UP_THRESHOLD

def identify_person_visual(face_emb):
    db = load_robot_db()
    best_id = "Unknown"
    max_score = 0
    for uid, data in db.items():
        score = np.dot(face_emb, np.array(data["face_vec"]))
        if score > max_score:
            max_score = score
            best_id = uid
    return best_id, max_score

def find_speaker_in_group(voice_emb, visible_users):
    if voice_emb is None:
        return None, 0.0
    db = load_robot_db()
    best_speaker_id = None
    max_score = 0
    for user in visible_users:
        uid = user['id']
        if uid == "Unknown":
            continue
        user_data = db.get(uid)
        if user_data and user_data.get('voice_vec'):
            saved_voice = np.array(user_data['voice_vec'])
            score = np.dot(voice_emb, saved_voice)
            if score > max_score:
                max_score = score
                best_speaker_id = uid
    if max_score > VOICE_SIM_THRESHOLD:
        return best_speaker_id, max_score
    return None, 0.0

def is_waving(frame):
    """Проверяет, машет ли человек любой рукой (левой или правой)"""
    try:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands_detector.process(rgb)
        if res.multi_hand_landmarks:
            # Проверяем все обнаруженные руки
            for hand_landmarks in res.multi_hand_landmarks:
                lms = hand_landmarks.landmark
                # Проверяем махание: кончик указательного пальца выше запястья
                if lms[8].y < lms[0].y:
                    return True
    except Exception:
        pass
    finally:
        # Give a tiny break back to other threads
        time.sleep(0.01)
    return False

def run_registration():
    global robot_state
    robot_state["status"] = "REGISTRATION"
    robot_state["subtext"] = "Freeze..."
    robot_state["color"] = (255, 0, 255)
    time.sleep(1.0)
    new_id = get_next_robot_id()
    robot_state["subtext"] = f"SPEAK! ({new_id})"
    
    # Get current frame from global buffer
    with FRAME_READ_LOCK:
        if LATEST_FRAME is not None:
            frame = LATEST_FRAME.copy()
        else:
            frame = None

    if frame is None:
        robot_state["subtext"] = "No Frame!"
        time.sleep(1)
        return

    faces = face_app.get(frame)
    if not faces:
        robot_state["subtext"] = "No Face!"
        time.sleep(1)
        return
    face_emb = faces[0].normed_embedding.tolist()
    new_id = get_next_robot_id()
    robot_state["subtext"] = f"SPEAK! ({new_id})"
    try:
        rec_voice = sd.rec(int(4 * NATIVE_RATE), samplerate=NATIVE_RATE, channels=1, blocking=False)
        # Instead of blocking=True, we wait and update the state without freezing everything
        start_rec = time.time()
        while time.time() - start_rec < 4.0:
            time.sleep(0.1)
        sd.wait()
        voice_emb = get_voice_embedding(rec_voice)
    except Exception:
        return
    if voice_emb is None:
        robot_state["subtext"] = "Voice model missing"
        time.sleep(1)
        return
    db = load_robot_db()
    db[new_id] = {"face_vec": face_emb, "voice_vec": voice_emb, "created_at": time.time()}
    save_robot_db(db)
    robot_state["status"] = "SAVED"
    robot_state["subtext"] = new_id
    robot_state["color"] = (0, 255, 0)
    time.sleep(2)

# Removed logic_loop, functionality moved to unified_vision_worker

def run_flask():
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)

@app.route("/")
def group_index():
    return '<html><body style="background:#000;color:#0f0;text-align:center;"><h1>GROUP ID</h1><img src="/video_feed" style="width:100%;max-width:640px;border:2px solid #333;"></body></html>'

def gen_frames():
    global outputFrame
    while True:
        with frame_lock:
            if outputFrame is None:
                pass
            else:
                (flag, enc) = cv2.imencode(".jpg", outputFrame)
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(enc) + b'\r\n')
        time.sleep(0.05)

@app.route("/video_feed")
def video_feed():
    try:
        ip = (request.headers.get("X-Forwarded-For") or request.remote_addr or "").split(",")[0].strip()
        ua = (request.headers.get("User-Agent") or "").strip()
        with camera_channel_users_lock:
            CAMERA_CHANNEL_USERS.add((ip, ua))
    except Exception:
        pass
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

# ==========================================
# 🦊 FOXGLOVE SLAM: лидар, трекинг человека, радар, команды роботу (всегда независимо)
# Порт 5001, камера FOXGLOVE_CAMERA_INDEX (1 по умолчанию, чтобы не конфликтовать с Group ID на 0)
# ==========================================
FOXGLOVE_CAMERA_FOV = 60
FOXGLOVE_MAX_RADAR_DIST_MM = 4000
FOXGLOVE_ARDUINO_URL = "http://192.168.4.1"
FOXGLOVE_CAMERA_INDEX = 1  # попытка 1, затем 0 (Group ID использует 0)
FOXGLOVE_PORT = 5001

# Consolidating into single app
foxglove_current_frame = None
foxglove_latest_scan = [0] * 360
foxglove_robot_command = "S"

def foxglove_send_command(cmd):
    global foxglove_robot_command
    if cmd != foxglove_robot_command:
        try:
            requests.get(f"{FOXGLOVE_ARDUINO_URL}/{cmd}", timeout=0.5)
            foxglove_robot_command = cmd
            print(f"🤖 FOXGLOVE ВІДПРАВЛЕНО: {cmd}")
        except Exception:
            pass

def foxglove_create_tracker():
    try:
        return cv2.TrackerKCF_create()
    except AttributeError:
        try:
            return cv2.legacy.TrackerKCF_create()
        except Exception:
            return cv2.TrackerMIL_create()

def foxglove_get_robust_distance(scan, target_angle, window_size=5):
    # valid_dists = []
    # for i in range(-window_size, window_size + 1):
    #     angle = int(target_angle + i) % 360
    #     dist = scan[angle]
    #     if 50 < dist < 8000:
    #         valid_dists.append(dist)
    # if not valid_dists:
    #     return None
    # valid_dists.sort()
    # return valid_dists[len(valid_dists) // 2]
    return 1500 # Always 1.5m (150 cm)

def foxglove_draw_radar_map(scan, target_angle=None, target_dist=None):
    size = 400
    center = (size // 2, size // 2)
    scale = (size / 2) / FOXGLOVE_MAX_RADAR_DIST_MM
    radar_img = np.zeros((size, size, 3), dtype=np.uint8)
    for r in [1000, 2000, 3000, 4000]:
        r_px = int(r * scale)
        cv2.circle(radar_img, center, r_px, (40, 40, 40), 1)
    fov_half = FOXGLOVE_CAMERA_FOV / 2.0
    cv2.ellipse(radar_img, center, (int(FOXGLOVE_MAX_RADAR_DIST_MM*scale), int(FOXGLOVE_MAX_RADAR_DIST_MM*scale)),
                -90, -fov_half, fov_half, (20, 20, 60), -1)
    cv2.line(radar_img, center, (center[0], 0), (0, 255, 0), 2)
    for angle in range(360):
        dist = scan[angle]
        if 50 < dist < FOXGLOVE_MAX_RADAR_DIST_MM:
            rad = math.radians(angle - 90)
            x = int(center[0] + dist * scale * math.cos(rad))
            y = int(center[1] + dist * scale * math.sin(rad))
            cv2.circle(radar_img, (x, y), 2, (255, 255, 255), -1)
    if target_angle is not None and target_dist is not None and target_dist > 0:
        rad = math.radians(target_angle - 90)
        x = int(center[0] + target_dist * scale * math.cos(rad))
        y = int(center[1] + target_dist * scale * math.sin(rad))
        cv2.line(radar_img, center, (x, y), (0, 0, 255), 1)
        cv2.circle(radar_img, (x, y), 6, (0, 0, 255), -1)
    cv2.circle(radar_img, center, 8, (0, 255, 255), -1)
    return radar_img

def foxglove_lidar_thread():
    global foxglove_latest_scan
    if serial is None:
        return
    BAUDRATE = 230400
    port_candidates = ["/dev/ttyAMA10"]
    try:
        port_candidates = sorted(set(
            port_candidates
            + glob.glob("/dev/ttyACM*")
            + glob.glob("/dev/ttyUSB*")
        ))
    except Exception:
        pass
    lidar_serial = None
    for port in port_candidates:
        try:
            lidar_serial = serial.Serial(port, BAUDRATE, timeout=1)
            break
        except Exception:
            lidar_serial = None
    if lidar_serial is None:
        print("!!!!!!!! lidar_serial is None")
        return
    print("lidar port is", lidar_serial)
    scan_data = [0] * 360
    last_angle = 0.0
    buffer = bytearray()
    while True:
        time.sleep(0.02)
        try:
            if lidar_serial.in_waiting > 0:
                buffer.extend(lidar_serial.read(lidar_serial.in_waiting))
            else:
                buffer.extend(lidar_serial.read(1))
            print(buffer)
            while len(buffer) >= 47:
                if buffer[0] == 0x54 and buffer[1] == 0x2C:
                    packet = buffer[:47]
                    del buffer[:47]
                    start_angle = (packet[4] + packet[5] * 256) / 100.0
                    end_angle = (packet[42] + packet[43] * 256) / 100.0
                    if start_angle > 360.0 or end_angle > 360.0:
                        continue
                    step = (end_angle - start_angle)
                    if step < 0:
                        step += 360.0
                    step /= 11.0
                    if start_angle < last_angle - 180:
                        foxglove_latest_scan = scan_data.copy()
                        scan_data = [0] * 360
                    for i in range(12):
                        # distance = packet[6 + i*3] + packet[7 + i*3] * 256
                        distance = 1500 # Always 1.5m (150 cm)
                        if 50 < distance < 8000:
                            scan_data[int((start_angle + i * step) + 0.5) % 360] = distance
                    last_angle = start_angle
                else:
                    del buffer[0:1]
        except Exception:
            time.sleep(0.01)

def unified_vision_worker():
    global LATEST_FRAME, GLOBAL_CAP, outputFrame, foxglove_current_frame, robot_state
    print(">>> [SYSTEM] Unified Vision Worker Started.", flush=True)
    
    # Init Group ID models
    try:
        init_group_models()
    except Exception as e:
        print(f">>> [INIT] Group ID early init failed: {e}", flush=True)

    # Init Foxglove components
    mp_pose_fox = mp.solutions.pose
    pose_fox = mp_pose_fox.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    tracking_active = False
    lock_counter = 0
    smoothed_human_distance = None
    tracker = None
    
    frame_counter = 0
    cached_people = []

    while True:
        # 1. Camera I/O
        with GLOBAL_CAP_LOCK:
            if GLOBAL_CAP is None or not GLOBAL_CAP.isOpened():
                GLOBAL_CAP = open_camera()
                if GLOBAL_CAP is None:
                    time.sleep(1)
                    continue
            ret, raw_frame = GLOBAL_CAP.read()
            if not ret:
                GLOBAL_CAP.release()
                GLOBAL_CAP = None
                continue
        
        with FRAME_READ_LOCK:
            LATEST_FRAME = raw_frame.copy()

        frame = raw_frame.copy()
        h, w = frame.shape[:2]
        
        # 2. Group ID Logic (Every NN_SKIP_FRAMES)
        frame_counter += 1
        if frame_counter % NN_SKIP_FRAMES == 0:
            try:
                faces = face_app.get(frame)
                cached_people = []
                if faces:
                    if is_waving(frame):
                        run_registration()
                    else:
                        for face in faces:
                            fid, fscore = identify_person_visual(face.normed_embedding)
                            if fscore < FACE_SIM_THRESHOLD: fid = "Unknown"
                            cached_people.append({
                                'id': fid, 'bbox': face.bbox.astype(int), 'role': 'Listener'
                            })
                        has_known = any(u['id'] != 'Unknown' for u in cached_people)
                        if has_known:
                            try:
                                check_sound = sd.rec(int(0.1 * NATIVE_RATE), samplerate=NATIVE_RATE, channels=1, blocking=True)
                                if np.max(np.abs(check_sound)) > 0.05:
                                    v_emb = get_voice_embedding(check_sound)
                                    speaker_id, score = find_speaker_in_group(v_emb, cached_people)
                                    if speaker_id:
                                        for user in cached_people:
                                            if user['id'] == speaker_id: user['role'] = 'SPEAKER'
                            except Exception: pass
            except Exception as e:
                print(f">>> [GROUP ID] Error: {e}")

        # Draw Group ID UI on 'group_frame'
        group_frame = frame.copy()
        if cached_people:
            for p in cached_people:
                bbox, role, uid = p['bbox'], p['role'], p['id']
                color = (0, 255, 0) if role == 'SPEAKER' else (0, 255, 255) if uid != "Unknown" else (0, 0, 255)
                text = f"SPEAKING: {uid}" if role == 'SPEAKER' else f"{uid} (Silent)" if uid != "Unknown" else "Unknown"
                cv2.rectangle(group_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                cv2.putText(group_frame, text, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            robot_state["status"], robot_state["subtext"] = "MONITORING", f"People: {len(cached_people)}"
        else:
            robot_state["status"], robot_state["subtext"] = "SEARCHING", "..."

        # MediaPipe Hands (always)
        try:
            if hands_detector:
                rgb_h = cv2.cvtColor(group_frame, cv2.COLOR_BGR2RGB)
                res_h = hands_detector.process(rgb_h)
                if res_h.multi_hand_landmarks:
                    for hlms in res_h.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(group_frame, hlms, mp.solutions.hands.HAND_CONNECTIONS)
        except Exception: pass

        cv2.rectangle(group_frame, (0, 0), (320, 30), (0, 0, 0), -1)
        cv2.putText(group_frame, robot_state["status"], (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, robot_state["color"], 1)
        cv2.rectangle(group_frame, (0, 210), (320, 240), (0, 0, 0), -1)
        cv2.putText(group_frame, robot_state["subtext"], (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        with frame_lock:
            outputFrame = group_frame.copy()

        # 3. Foxglove tracking (Every Frame)
        fox_image = frame.copy()
        target_angle = None
        human_distance = None
        angle_offset = 0
        cmd = "S"
        
        cv2.line(fox_image, (w//2, 0), (w//2, h), (255, 255, 255), 1)
        x_m15 = int(w/2 - (15.0 / FOXGLOVE_CAMERA_FOV) * w)
        x_p15 = int(w/2 + (15.0 / FOXGLOVE_CAMERA_FOV) * w)
        cv2.line(fox_image, (x_m15, 0), (x_m15, h), (100, 255, 100), 1)
        cv2.line(fox_image, (x_p15, 0), (x_p15, h), (100, 255, 100), 1)

        if not tracking_active:
            res_p = pose_fox.process(cv2.cvtColor(fox_image, cv2.COLOR_BGR2RGB))
            if res_p.pose_landmarks:
                lm = res_p.pose_landmarks.landmark
                pts = [lm[11], lm[12], lm[23], lm[24]]
                xs, ys = [p.x * w for p in pts], [p.y * h for p in pts]
                xb, yb, wb, hb = int(min(xs)), int(min(ys)), int(max(xs)-min(xs)), int(max(ys)-min(ys))
                angle_offset = ( (xb+wb/2.0) / w - 0.5) * FOXGLOVE_CAMERA_FOV
                target_angle = int(angle_offset) % 360
                raw_dist = foxglove_get_robust_distance(foxglove_latest_scan, target_angle)
                if raw_dist: smoothed_human_distance = raw_dist if smoothed_human_distance is None else int(0.2*raw_dist + 0.8*smoothed_human_distance)
                human_distance = smoothed_human_distance or 0
                cv2.rectangle(fox_image, (xb, yb), (xb+wb, yb+hb), (0, 255, 255), 2)
                if -10 <= angle_offset <= 10 and 400 < human_distance < 2000:
                    lock_counter += 1
                    if lock_counter > 15:
                        tracker = foxglove_create_tracker()
                        tracker.init(frame, (max(0,xb), max(0,yb), min(w-xb,wb), min(h-yb,hb)))
                        tracking_active = True
                else: lock_counter = 0
        else:
            ok, bbox = tracker.update(frame)
            if ok:
                xb, yb, wb, hb = [int(v) for v in bbox]
                angle_offset = ( (xb+wb/2.0) / w - 0.5) * FOXGLOVE_CAMERA_FOV
                target_angle = int(angle_offset) % 360
                raw_dist = foxglove_get_robust_distance(foxglove_latest_scan, target_angle)
                if raw_dist: smoothed_human_distance = raw_dist if smoothed_human_distance is None else int(0.2*raw_dist + 0.8*smoothed_human_distance)
                human_distance = smoothed_human_distance or 0
                cv2.rectangle(fox_image, (xb, yb), (xb+wb, yb+hb), (0, 255, 0), 3)
                if human_distance > 100:
                    if human_distance < 600: cmd = "B"
                    elif human_distance < 1000: cmd = "S"
                    else: cmd = "L" if angle_offset < -15 else "R" if angle_offset > 15 else "F"
            else:
                tracking_active = False
                lock_counter = 0
                smoothed_human_distance = None

        foxglove_send_command(cmd)
        cv2.putText(fox_image, f"Angle: {int(angle_offset)} deg", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(fox_image, f"CMD: {foxglove_robot_command}", (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        radar_img = foxglove_draw_radar_map(foxglove_latest_scan, target_angle, human_distance)
        combined_img = np.hstack((cv2.resize(fox_image, (int(w * 400/h), 400)), radar_img))
        ret, buf = cv2.imencode(".jpg", combined_img)
        if ret: foxglove_current_frame = buf.tobytes()

        time.sleep(0.1)

def foxglove_generate_frames():
    global foxglove_current_frame
    while True:
        if foxglove_current_frame is not None:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + foxglove_current_frame + b"\r\n")
        else:
            time.sleep(0.05)

@app.route("/foxglove")
def foxglove_index():
    return """<!DOCTYPE html><html><head><title>Robot Follow (Foxglove)</title></head>
    <body style="background:#111; text-align:center; color:white; font-family:sans-serif;">
        <h3>Зліва: Camera Tracker (HUD) &nbsp;|&nbsp; Справа: 2D Радар</h3>
        <img src="/foxglove/video_feed" style="max-width: 100%; border: 3px solid #00ffcc; border-radius: 5px;">
    </body></html>"""

@app.route("/foxglove/video_feed")
def foxglove_video_feed():
    try:
        ip = (request.headers.get("X-Forwarded-For") or request.remote_addr or "").split(",")[0].strip()
        ua = (request.headers.get("User-Agent") or "").strip()
        with camera_channel_users_lock:
            CAMERA_CHANNEL_USERS.add((ip, ua))
    except Exception:
        pass
    return Response(foxglove_generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

# Removed run_foxglove_flask to consolidate apps

# Import existing functionality from chat.py
def init_embeddings_db(folder: Path) -> str:
    """
    Initializes folder and SQLite DB for storing facts and embeddings.
    Returns path to database file (string).
    """
    try:
        folder.mkdir(parents=True, exist_ok=True)
        db_path = folder / "embeddings.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Create facts table (if not created)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dialogue_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                date TEXT NOT NULL,
                fact_text TEXT NOT NULL,
                people TEXT NOT NULL,
                objects TEXT NOT NULL,
                importance REAL DEFAULT 0.0,
                embedding TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Try to add importance column (for old schemas)
        try:
            cursor.execute('ALTER TABLE facts ADD COLUMN importance REAL DEFAULT 0.0')
        except sqlite3.OperationalError:
            # column already exists — ignore
            pass

        conn.commit()
        conn.close()
        return str(db_path)
    except Exception as e:
        print(f"Error initializing embeddings database: {e}")
        # Return path even on error (can be seen), but many operations will fail later
        return str(folder / "embeddings.db")

def generate_embedding(client, text):
    """Generates embedding for text using OpenAI"""
    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

def save_fact_embedding(db_path, dialogue_id, timestamp, date, fact_text, people, objects, importance, embedding):
    """Saves fact and its embedding to SQLite database and (if possible) to sqlite-vec table"""
    if embedding is None:
        return False

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Convert embedding (list of float) to JSON string
        embedding_json = json.dumps(embedding)
        people_json = json.dumps(people, ensure_ascii=False)
        objects_json = json.dumps(objects, ensure_ascii=False)

        cursor.execute('''
            INSERT INTO facts
            (dialogue_id, timestamp, date, fact_text, people, objects, importance, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (dialogue_id, timestamp, date, fact_text, people_json, objects_json, importance, embedding_json))

        fact_id = cursor.lastrowid

        # Try to initialize sqlite-vec and add record to vector table
        if init_sqlite_vec(conn):
            try:
                # Pack embedding into BLOB with float32 (required by sqlite-vec)
                vec_blob = sqlite3.Binary(array('f', embedding).tobytes())
                cursor.execute(
                    "INSERT OR REPLACE INTO facts_vec(rowid, embedding) VALUES (?, ?)",
                    (fact_id, vec_blob),
                )
            except Exception as e:
                # If failed to update vector table — log, but don't crash
                print(f"Warning: failed to update facts_vec: {e}")

        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error saving embedding for fact: {e}")
        return False

def cosine_similarity(vec1, vec2):
    """Calculates cosine similarity between two vectors"""
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)

def init_sqlite_vec(conn, dim: int = 1536) -> bool:
    """
    Initializes sqlite-vec extension and facts_vec virtual table.
    Returns True if extension successfully initialized, otherwise False.
    """
    try:
        conn.enable_load_extension(True)

        # Try several typical extension names
        loaded = False
        for ext_name in ("sqlite-vec", "vec0"):
            try:
                conn.load_extension(ext_name)
                loaded = True
                break
            except Exception:
                continue

        if not loaded:
            # If extension failed to load — just return False,
            # code can use slow Python search later
            return False

        # Create virtual table for vector search (if doesn't exist)
        conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS facts_vec USING vec0(embedding float[{dim}])"
        )
        return True
    except Exception as e:
        print(f"Warning: failed to initialize sqlite-vec: {e}")
        return False

def search_similar_facts(client, db_path, query_text, top_n: int = 5, importance: float = 0.0):
    """
    For given phrase calculates embedding and searches for N closest facts from database.
    Uses sqlite-vec if possible, otherwise — fallback search in Python.
    Filters facts by importance.

    Args:
        client: OpenAI client
        db_path: Path to database
        query_text: Text for search
        top_n: Number of closest facts to return
        importance: Importance threshold for fact (0.0-1.0), default 0.0.
                    - 0.0: includes only meaningless facts (importance = 0.0)
                    - 0.1-0.9: includes general knowledge and above (importance >= importance)
                    - 1.0: includes only personal knowledge (importance = 1.0)

                    Importance assessment rules:
                    - 0.0: meaningless facts ("false", "true", "good", "yes", "no")
                    - 0.1-0.9: general knowledge ("Apples grow on trees", "Sky is blue")
                    - 1.0: personal knowledge about people ("Max likes math", "Peter is doing well")

    Returns list of dictionaries:
    [
      {
        "id": int,
        "dialogue_id": str,
        "timestamp": str,
        "date": str,
        "fact_text": str,
        "people": list[str],
        "objects": list[str],
        "importance": float,
        "similarity": float
      },
      ...
    ]
    """
    # Generate embedding for query
    query_embedding = generate_embedding(client, query_text)
    if query_embedding is None:
        return []

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
    except Exception as e:
        print(f"Error opening embeddings database: {e}")
        return []

    # First try to use sqlite-vec
    if init_sqlite_vec(conn):
        try:
            # Pack query vector into float32 BLOB
            query_blob = sqlite3.Binary(array('f', query_embedding).tobytes())
            cursor.execute(
                f"""
                SELECT f.id, f.dialogue_id, f.timestamp, f.date,
                       f.fact_text, f.people, f.objects, f.importance,
                       v.distance
                FROM facts_vec AS v
                JOIN facts AS f ON f.id = v.rowid
                WHERE v.embedding MATCH ? AND (f.importance IS NOT NULL AND f.importance >= ?)
                ORDER BY v.distance ASC
                LIMIT ?
                """,
                (query_blob, importance, top_n),
            )
            rows = cursor.fetchall()
            conn.close()

            results = []
            for row in rows:
                fact_id, dialogue_id, timestamp, date_str, fact_text, people_json, objects_json, fact_importance, distance = row

                # Filter by importance
                importance_value = fact_importance if fact_importance is not None else 0.0
                if importance == 0.0:
                    # If importance = 0.0, include only meaningless facts
                    if importance_value != 0.0:
                        continue
                else:
                    # If importance > 0.0, include facts with importance >= importance
                    if importance_value < importance:
                        continue

                try:
                    people = json.loads(people_json)
                except Exception:
                    people = []
                try:
                    objects = json.loads(objects_json)
                except Exception:
                    objects = []

                # Convert distance to "similarity": smaller distance = higher similarity
                similarity = 1.0 / (1.0 + float(distance))

                results.append({
                    "id": fact_id,
                    "dialogue_id": dialogue_id,
                    "timestamp": timestamp,
                    "date": date_str,
                    "fact_text": fact_text,
                    "people": people,
                    "objects": objects,
                    "importance": importance_value,
                    "similarity": similarity,
                })

            return results
        except Exception as e:
            print(f"Error in vector search via sqlite-vec, using fallback search: {e}")
            # If something went wrong — fall back to Python variant below

    # Fallback slow variant: read all facts and calculate cosine similarity in Python
    try:
        cursor.execute(
            """
            SELECT id, dialogue_id, timestamp, date, fact_text, people, objects, importance, embedding
            FROM facts
            """
        )
        rows = cursor.fetchall()
        conn.close()
    except Exception as e:
        print(f"Error reading from embeddings database: {e}")
        return []

    results = []
    for row in rows:
        fact_id, dialogue_id, timestamp, date_str, fact_text, people_json, objects_json, fact_importance, embedding_json = row
        try:
            fact_embedding = json.loads(embedding_json)
            similarity = cosine_similarity(query_embedding, fact_embedding)
        except Exception:
            # If something wrong with specific record — skip
            continue

        # Filter by importance
        importance_value = fact_importance if fact_importance is not None else 0.0
        if importance == 0.0:
            # If importance = 0.0, include only meaningless facts
            if importance_value != 0.0:
                continue
        else:
            # If importance > 0.0, include facts with importance >= importance
            if importance_value < importance:
                continue

        try:
            people = json.loads(people_json)
        except Exception:
            people = []

        try:
            objects = json.loads(objects_json)
        except Exception:
            objects = []

        results.append({
            "id": fact_id,
            "dialogue_id": dialogue_id,
            "timestamp": timestamp,
            "date": date_str,
            "fact_text": fact_text,
            "people": people,
            "objects": objects,
            "importance": importance_value,
            "similarity": similarity,
        })

    # Sort by similarity (higher to lower) and return top_n
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:top_n]

class FactItem(BaseModel):
    fact: str
    people: list[str]
    objects: list[str]
    importance: float = 0.0

class DialogueExtract(BaseModel):
    facts: list[FactItem]
    date: str

def create_extract(client, messages, dialogue_date):
    """Creates dialogue extract with facts, where each fact has people and objects specified"""
    # Form dialogue text for analysis ONLY from user messages
    dialogue_text = "\n".join([
        f"{msg['role']}: {msg['content']}"
        for msg in messages
    ])

    # Prompt for dialogue analysis
    analysis_prompt = f"""Analyze the following dialogue and create a structured extract.

Dialogue:
{dialogue_text}

Create an object with these fields, using ONLY information from user statements (USER:):
- "facts": array of objects, where EACH object describes a SEPARATE fact.
  For each fact there should be fields:
  - "fact": fact text in one sentence
  - "people": array of names of people related to this specific fact (if none — empty array [])
  - "objects": array of objects related to the fact (if none — empty array [])
  - "importance": importance of fact (float from 0.0 to 1.0), MANDATORILY determine according to these rules:
     * 0.0 — meaningless facts without specific content (for example: "false", "true", "good", "yes", "no", "ok", "alright")
     * 0.1-0.9 — general knowledge and facts about the world (for example: "Apples grow on trees", "Sky is blue", "Water boils at 100 degrees")
     * 1.0 — personal knowledge about specific people, their preferences, life events (for example: "Max likes math", "Peter is doing well", "Anna works in IT")
- "date": dialogue date in YYYY-MM-DD format

Respond without additional text. If some element is missing, use empty array [].

Example format:
{{
  "facts": [
    {{
      "fact": "Max likes math",
      "people": ["Max"],
      "objects": ["math"],
      "importance": 1.0
    }},
    {{
      "fact": "Peter is doing well",
      "people": ["Peter"],
      "objects": [],
      "importance": 1.0
    }},
    {{
      "fact": "Apples grow on trees",
      "people": [],
      "objects": ["apples", "trees"],
      "importance": 0.5
    }},
    {{
      "fact": "Sky is blue",
      "people": [],
      "objects": ["sky"],
      "importance": 0.3
    }},
    {{
      "fact": "good",
      "people": [],
      "objects": [],
      "importance": 0.0
    }},
    {{
      "fact": "false",
      "people": [],
      "objects": [],
      "importance": 0.0
    }}
  ],
  "date": "2024-01-15"
}}"""

    try:
        response = client.responses.parse(
            model="gpt-4o-mini",
            input=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert in text analysis. "
                        "Analyze user statements and fill Pydantic schema "
                        "with facts, people and objects. "
                        "\n\nCRITICALLY IMPORTANT to correctly determine importance for each fact:\n"
                        "- 0.0 — meaningless facts without specific content (one-word responses like 'yes', 'no', 'good', 'false', 'true')\n"
                        "- 0.1-0.9 — general knowledge about the world, nature, science, culture (facts that concern everyone, not a specific person)\n"
                        "- 1.0 — personal knowledge about specific people: their preferences, life events, personal characteristics\n"
                        "\nIf something is missing, use empty list."
                    ),
                },
                {"role": "user", "content": analysis_prompt},
            ],
            text_format=DialogueExtract,
            temperature=0.3,
        )

        extract_model: DialogueExtract = response.output_parsed
        extract_data = extract_model.model_dump()

        # Overwrite dialogue date with actual
        extract_data["date"] = dialogue_date

        return extract_data
    except Exception as e:
        print(f"\nError creating extract: {e}")
        # Return basic structure with date
        return {
            "facts": [],
            "date": dialogue_date
        }

def save_dialogue(messages, dialogues_dir, summaries_dir, client, db_path):
    """Saves dialogue to file and creates extract with embeddings for facts (with people and objects for each fact)"""
    if not messages:
        return

    # Create folder if doesn't exist
    dialogues_dir.mkdir(exist_ok=True)
    summaries_dir.mkdir(exist_ok=True)

    # Create filename with date and time
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dialogue_date = datetime.now().strftime("%Y-%m-%d")
    dialogue_id = f"dialogue_{timestamp}"
    filename = dialogues_dir / f"{dialogue_id}.json"

    # Create structure for saving
    dialogue_data = {
        "timestamp": datetime.now().isoformat(),
        "messages": messages
    }

    # Save in JSON format
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(dialogue_data, f, ensure_ascii=False, indent=2)

    print(f"\nDialogue saved: {filename}")

    # Create extract
    print("Creating dialogue extract...")
    extract_data = create_extract(client, messages, dialogue_date)

    # Save extract
    extract_filename = summaries_dir / f"extract_{timestamp}.json"
    with open(extract_filename, 'w', encoding='utf-8') as f:
        json.dump(extract_data, f, ensure_ascii=False, indent=2)

    print(f"Extract saved: {extract_filename}")

    # Generate and save embeddings for individual facts
    print("Generating embeddings for facts...")
    for fact_item in extract_data.get("facts", []):
        # Expected structure:
        # {
        #   "fact": "...",
        #   "people": [...],
        #   "objects": [...],
        #   "importance": 0.0-1.0
        # }
        fact_text = fact_item.get("fact")
        if not fact_text:
            continue

        people = fact_item.get("people", [])
        objects = fact_item.get("objects", [])
        importance = fact_item.get("importance", 0.0)

        # Form text for embedding: fact + related people and objects
        combined_parts = [fact_text]
        if people:
            combined_parts.append("People: " + ", ".join(people))
        if objects:
            combined_parts.append("Objects: " + ", ".join(objects))
        embedding_text = "\n".join(combined_parts)

        fact_embedding = generate_embedding(client, embedding_text)
        if fact_embedding:
            saved = save_fact_embedding(
                db_path=db_path,
                dialogue_id=dialogue_id,
                timestamp=timestamp,
                date=dialogue_date,
                fact_text=fact_text,
                people=people,
                objects=objects,
                importance=importance,
                embedding=fact_embedding,
            )
            if not saved:
                print(f"Failed to save embedding for fact: {fact_text}")

# Global variables for conversation state
messages = []
dialogues_dir = Path("dialogues")
summaries_dir = Path("summaries")
embeddings_db_folder = Path("embeddings_db")
db_path = None
openai_client = None

# Global state tracking for interruptions
current_question = ""
current_answer = ""
interruption_count = 0
is_responding = False
answer_started = False

class ConversationManager:
    def __init__(self):
        self.messages = []
        self.db_path = None
        self.client = None

    def initialize(self, client, db_path):
        self.client = client
        self.db_path = db_path

    def add_user_message(self, content):
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content):
        self.messages.append({"role": "assistant", "content": content})

    def remove_incomplete_assistant_messages(self):
        """Remove all assistant messages after the last user message"""
        # Find the last user message
        last_user_idx = -1
        for i in range(len(self.messages) - 1, -1, -1):
            if self.messages[i]["role"] == "user":
                last_user_idx = i
                break
        
        # Remove all assistant messages after the last user message
        if last_user_idx >= 0:
            self.messages = self.messages[:last_user_idx + 1]

    def get_recent_messages(self, limit=20):
        return self.messages[-limit:] if len(self.messages) > limit else self.messages

    def save_conversation(self):
        if not self.messages:
            return "No conversation to save."

        save_dialogue(self.messages, dialogues_dir, summaries_dir, self.client, self.db_path)
        return "Conversation saved and facts extracted."

conversation_manager = ConversationManager()

def search_facts_tool(query: str) -> str:
    """Search for similar facts in the knowledge base based on the query."""
    global openai_client, db_path
    if not openai_client or not db_path:
        return "Knowledge base not initialized."

    # Get recent conversation context
    recent_messages = conversation_manager.get_recent_messages(10)
    context_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_messages])

    # Combine query with context for better search
    search_query = f"{query}\n\nRecent conversation context:\n{context_text}"

    similar_facts = search_similar_facts(openai_client, db_path, search_query, top_n=5, importance=0.2)

    if not similar_facts:
        return "No relevant facts found in knowledge base."

    result_parts = []
    for i, fact in enumerate(similar_facts, start=1):
        result_parts.append(f"- Fact {i} (date: {fact['date']}): {fact['fact_text']}")
        if fact.get("people"):
            result_parts.append(f"  People: {', '.join(fact['people'])}")
        if fact.get("objects"):
            result_parts.append(f"  Objects: {', '.join(fact['objects'])}")

    return "\n".join(result_parts)

def save_conversation_tool() -> str:
    """Save the current conversation and extract facts from it."""
    return conversation_manager.save_conversation()

# Create tools for the realtime client
tools = [
    FunctionTool.from_defaults(fn=search_facts_tool),
    FunctionTool.from_defaults(fn=save_conversation_tool),
]

async def main():
    global db_path, openai_client

    # Foxglove SLAM: лидар + камера + радар + команды роботу — всегда и независимо от остального
    # Combined Vision Worker: Capturing + Group ID + Foxglove Slam
    threading.Thread(target=unified_vision_worker, daemon=True).start()
    threading.Thread(target=foxglove_lidar_thread, daemon=True).start()
    threading.Thread(target=run_flask, daemon=True).start()
    print("Robot Interface: http://0.0.0.0:5000/ (Group ID)")
    print("Foxglove SLAM: http://0.0.0.0:5000/foxglove (camera + radar)")

    # Robustly load .env (search parent folders) and validate API key
    dotenv_path = find_dotenv() or ""
    if dotenv_path:
        load_dotenv(dotenv_path)
        print(f"Loaded .env from: {dotenv_path}")
    else:
        load_dotenv()
        print("No .env found by find_dotenv; attempted load_dotenv() in current directory.")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found.")
        print("Please create .env file and add:")
        print("OPENAI_API_KEY=your_api_key_here")
        print("Debug: Script path:", Path(__file__).resolve())
        sys.exit(1)

    # sanitize key: strip, unquote, and try to extract substring starting with 'sk-'
    api_key = api_key.strip()
    if (api_key.startswith('"') and api_key.endswith('"')) or (api_key.startswith("'") and api_key.endswith("'")):
        api_key = api_key[1:-1]

    if 'sk-' in api_key and not api_key.startswith('sk-'):
        # if there are stray characters before 'sk-', trim them
        idx = api_key.find('sk-')
        print(f"Notice: trimming API key prefix before 'sk-' at pos {idx}")
        api_key = api_key[idx:]

    if not api_key.startswith('sk-') or len(api_key) < 20:
        print("Error: OPENAI_API_KEY appears invalid after sanitization. Please check your .env file.")
        print(f"Sanitized key preview: {api_key[:8]}...")
        sys.exit(1)

    masked = api_key[:4] + '...' + api_key[-4:]
    print(f"Using OPENAI_API_KEY (masked): {masked}")

    # Initialize OpenAI client
    openai_client = OpenAI(api_key=api_key)

    # Initialize database
    db_path = init_embeddings_db(embeddings_db_folder)

    # Initialize conversation manager
    conversation_manager.initialize(openai_client, db_path)

    # Start Group ID (ultv1): logic loop + Flask video feed on port 5000
    # Vision worker already started above

    audio_handler = AudioHandler()

    def on_text_delta(text):
        global current_answer, is_responding, answer_started
        # Обрабатываем ответы только в режиме AWAKE
        if robot_state["mode"] != "AWAKE":
            return
        print(f"Assistant: {text}", end="", flush=True)
        # Add to conversation when assistant speaks
        if text.strip():
            conversation_manager.add_assistant_message(text)
            # Mark that we started responding and accumulate answer text
            if not answer_started and is_responding:
                answer_started = True
            if is_responding:
                current_answer += text

    def on_audio_delta(audio):
        # Воспроизводим аудио только в режиме AWAKE
        if robot_state["mode"] != "AWAKE":
            return
        print("Playing audio delta")
        audio_handler.play_audio(audio)

    async def generate_full_answer():
        """Generate full answer to current question using OpenAI client"""
        try:
            recent_messages = conversation_manager.get_recent_messages(10)
            # Remove the last assistant message if it's incomplete
            if recent_messages and recent_messages[-1].get("role") == "assistant":
                recent_messages = recent_messages[:-1]
            
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=recent_messages + [
                    {"role": "user", "content": current_question}
                ],
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"\nError generating response: {e}")
            return current_answer

    async def handle_interrupt_response():
        """Handle the response after interruption based on interruption count"""
        global interruption_count, current_question, current_answer, is_responding, answer_started
        
        # Remove incomplete assistant messages that were added during streaming
        conversation_manager.remove_incomplete_assistant_messages()
        
        # Generate full answer if we don't have it yet or need to regenerate
        if not current_answer.strip() or interruption_count == 1:
            full_answer = await generate_full_answer()
            if full_answer:
                current_answer = full_answer
        
        if interruption_count == 1:
            # First interruption: ask politely and give the answer
            response_text = "Прошу вас не перебивать меня. " + current_answer
            print(f"\nAssistant: {response_text}")
            conversation_manager.add_assistant_message(response_text)
            # Keep is_responding = True so we can detect second interruption
            answer_started = False
                
        elif interruption_count >= 2:
            # Two or more interruptions: give full answer with question reference
            response_text = f'Ответ на вопрос "{current_question}". {current_answer}'
            response_text += " Какие ваши следующие вопросы?"
            print(f"\nAssistant: {response_text}")
            conversation_manager.add_assistant_message(response_text)
            
            # Reset state for next question
            interruption_count = 0
            current_question = ""
            current_answer = ""
            is_responding = False
            answer_started = False

    def on_interrupt():
        global interruption_count, is_responding
        print("\nStopping audio playback due to interrupt")
        audio_handler.stop_playback_immediately()
        
        if is_responding:
            interruption_count += 1
            # Schedule the response handling using the current event loop
            loop = asyncio.get_running_loop()
            if loop.is_running():
                asyncio.create_task(handle_interrupt_response())

    def on_input_transcript(transcript):
        global current_question, current_answer, interruption_count, is_responding, answer_started
        # Always process transcripts since we are always AWAKE
        print(f"Input transcript: {transcript}")
        # Reset state when new question comes in
        current_question = transcript
        current_answer = ""
        interruption_count = 0
        is_responding = True
        answer_started = False
        # Add user message to conversation
        conversation_manager.add_user_message(transcript)

    def on_output_transcript(transcript):
        global is_responding, answer_started
        print(f"Output transcript: {transcript}")
        # Mark that we're done responding
        if transcript.strip():
            is_responding = False
            answer_started = False

    realtime_client = RealtimeClient(
        api_key=api_key,
        model="gpt-4o-realtime-preview",
        instructions="ти Вася",
        on_text_delta=on_text_delta,
        on_audio_delta=on_audio_delta,
        on_interrupt=on_interrupt,
        on_input_transcript=on_input_transcript,
        on_output_transcript=on_output_transcript,
        turn_detection_mode=TurnDetectionMode.SERVER_VAD,
        tools=tools,
    )

    try:
        try:
            await realtime_client.connect()
            # Start message handler
            message_handler = asyncio.create_task(realtime_client.handle_messages())
            
            # Start streaming immediately
            streaming_task = asyncio.create_task(audio_handler.start_streaming(realtime_client))
            
            print("Connected to OpenAI Realtime API!")
            print("Audio streaming started.")
            
            # Run until one of the tasks finishes or the process is interrupted
            await asyncio.gather(message_handler, streaming_task)
            
        except Exception as conn_err:
            print(f"Realtime API connection failed: {conn_err}")
            # Keep running even without voice assistant
            pass

    except KeyboardInterrupt:
        # Graceful stop (Ctrl+C)
        pass
    except Exception as e:
        print(f"Error in Realtime loop: {e}")
    finally:
        # Пытаемся аккуратно завершить Realtime‑клиент (если он вообще инициализировался)
        try:
            conversation_manager.save_conversation()
        except Exception as save_err:
            print(f"Error while saving conversation: {save_err}")

        try:
            audio_handler.stop_streaming()
            audio_handler.cleanup()
        except Exception as audio_err:
            print(f"Error while cleaning up audio handler: {audio_err}")

        try:
            await realtime_client.close()
        except Exception as close_err:
            print(f"Error while closing realtime client: {close_err}")

    # ВАЖНО: не выходим из main(), чтобы фонові потоки (Group ID, Foxglove) жили всегда
    while True:
        await asyncio.sleep(3600)

if __name__ == "__main__":
    print("Starting Realtime API CLI with Knowledge Base...")
    asyncio.run(main())