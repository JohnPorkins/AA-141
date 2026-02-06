import os
import shutil
import json
import time
import threading
import numpy as np
import cv2
from flask import Flask, Response

# --- 1. ФИКСЫ ---
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

# --- 2. НАСТРОЙКИ ---
DB_FILE = "robot_memory.json"
MIC_DEVICE = None            
FACE_SIM_THRESHOLD = 0.45    
VOICE_SIM_THRESHOLD = 0.30   
WAKE_UP_THRESHOLD = 0.3      
SLEEP_TIMEOUT = 15           
NN_SKIP_FRAMES = 15          # Анализ каждые 15 кадров

outputFrame = None
lock = threading.Lock()
app = Flask(__name__)

robot_state = {
    "status": "BOOTING...",
    "subtext": "Loading Group Logic...",
    "color": (255, 255, 255),
    "mode": "SLEEP"
}

# --- 3. ИНИЦИАЛИЗАЦИЯ ---
print(">>> [INIT] Загрузка моделей...")

try:
    dev_info = sd.query_devices(kind='input')
    NATIVE_RATE = int(dev_info['default_samplerate'])
except:
    NATIVE_RATE = 44100

resampler = T.Resample(NATIVE_RATE, 16000)
vad_model, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', trust_repo=True)
speaker_model = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="tmp_model")
# Используем Small для скорости в толпе
face_app = FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=-1, det_size=(320, 320))

mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5)

# --- 4. БАЗОВЫЕ ФУНКЦИИ ---
def load_db():
    if os.path.exists(DB_FILE):
        try:
            # ИСПРАВЛЕНО: Развернуто на несколько строк
            with open(DB_FILE, "r") as f:
                return json.load(f)
        except:
            pass
    return {}

def save_db(db):
    with open(DB_FILE, "w") as f:
        json.dump(db, f)

def get_next_id():
    db = load_db()
    return f"User_{len(db) + 1:05d}"

def convert_audio(audio_np):
    waveform = torch.from_numpy(audio_np).float()
    if len(waveform.shape) == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.shape[0] != 1:
        waveform = waveform.t()
    return resampler(waveform)

def get_voice_embedding(audio_data):
    wav_16k = convert_audio(audio_data)
    emb = speaker_model.encode_batch(wav_16k)
    return (emb.squeeze().cpu().numpy() / np.linalg.norm(emb.squeeze().cpu().numpy())).tolist()

def is_silence(audio_chunk):
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
    """Определяет ID только по лицу"""
    db = load_db()
    best_id = "Unknown"
    max_score = 0
    
    for uid, data in db.items():
        score = np.dot(face_emb, np.array(data["face_vec"]))
        if score > max_score:
            max_score = score
            best_id = uid
            
    return best_id, max_score

def find_speaker_in_group(voice_emb, visible_users):
    """
    Среди тех, кого мы видим (visible_users), ищем того, чей голос звучит.
    visible_users = [{'id': 'User_01'}, {'id': 'Unknown'}...]
    """
    db = load_db()
    best_speaker_id = None
    max_score = 0
    
    for user in visible_users:
        uid = user['id']
        if uid == "Unknown": continue # У незнакомцев нет голоса в базе
        
        user_data = db.get(uid)
        if user_data and user_data.get('voice_vec'):
            saved_voice = np.array(user_data['voice_vec'])
            score = np.dot(voice_emb, saved_voice)
            
            if score > max_score:
                max_score = score
                best_speaker_id = uid
                
    # Если сходство выше порога - мы нашли болтуна
    if max_score > VOICE_SIM_THRESHOLD:
        return best_speaker_id, max_score
    return None, 0.0

def is_waving(frame):
    try:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands_detector.process(rgb)
        if res.multi_hand_landmarks:
            lms = res.multi_hand_landmarks[0].landmark
            return lms[8].y < lms[0].y
    except: pass
    return False

# --- 5. РЕГИСТРАЦИЯ ---
def run_registration():
    global robot_state
    robot_state["status"] = "REGISTRATION"
    robot_state["subtext"] = "Freeze..."
    robot_state["color"] = (255, 0, 255)
    time.sleep(1.0)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): cap = cv2.VideoCapture(1)
    cap.set(3, 320); cap.set(4, 240)
    ret, frame = cap.read()
    cap.release()
    
    if not ret: return
    faces = face_app.get(frame)
    if not faces:
        robot_state["subtext"] = "No Face!"
        time.sleep(1); return
    face_emb = faces[0].normed_embedding.tolist()

    new_id = get_next_id()
    robot_state["subtext"] = f"SPEAK! ({new_id})"
    try:
        rec_voice = sd.rec(int(4 * NATIVE_RATE), samplerate=NATIVE_RATE, channels=1, blocking=True)
        voice_emb = get_voice_embedding(rec_voice)
    except: return

    db = load_db()
    db[new_id] = {"face_vec": face_emb, "voice_vec": voice_emb, "created_at": time.time()}
    save_db(db)
    
    robot_state["status"] = "SAVED"
    robot_state["subtext"] = new_id
    robot_state["color"] = (0, 255, 0)
    time.sleep(2)

# --- 6. ЛОГИКА ГРУППЫ ---
def logic_loop():
    global outputFrame, robot_state
    
    ratio = NATIVE_RATE / 16000
    block_size = int(np.ceil(512 * ratio))
    
    robot_state["mode"] = "SLEEP"
    last_activity = time.time()
    cap = None
    
    frame_counter = 0
    # Кэш теперь хранит список словарей с полной инфой
    cached_people = [] # [{'id': 'User1', 'bbox': [...], 'role': 'Listener'}, ...]
    
    print(">>> [SYSTEM] Group Logic Active.")
    
    while True:
        # === СПИМ ===
        if robot_state["mode"] == "SLEEP":
            robot_state["status"] = "SLEEP MODE"
            robot_state["subtext"] = "Silence..."
            robot_state["color"] = (100, 100, 100)
            
            if outputFrame is not None:
                with lock: outputFrame[:] = 0 
            
            try:
                with sd.InputStream(samplerate=NATIVE_RATE, channels=1, dtype='float32', blocksize=block_size) as stream:
                    while True:
                        chunk, _ = stream.read(block_size)
                        if not is_silence(chunk):
                            print(">>> ЗВУК!")
                            robot_state["mode"] = "AWAKE"
                            last_activity = time.time()
                            frame_counter = 0
                            break
            except: time.sleep(1)
        
        # === БОДРСТВУЕМ ===
        elif robot_state["mode"] == "AWAKE":
            if cap is None or not cap.isOpened():
                cap = cv2.VideoCapture(0)
                if not cap.isOpened(): cap = cv2.VideoCapture(1)
                cap.set(3, 320); cap.set(4, 240)
            
            ret, frame = cap.read()
            if not ret: continue
            
            frame_counter += 1
            
            # --- 1. АНАЛИЗ ЛИЦ (Раз в N кадров) ---
            if frame_counter % NN_SKIP_FRAMES == 0:
                faces = face_app.get(frame)
                cached_people = [] # Очищаем кэш
                
                if faces:
                    # Сначала распознаем всех визуально
                    visible_users = []
                    for face in faces:
                        fid, fscore = identify_person_visual(face.normed_embedding)
                        if fscore < FACE_SIM_THRESHOLD: fid = "Unknown"
                        
                        visible_users.append({
                            'id': fid,
                            'face_emb': face.normed_embedding,
                            'bbox': face.bbox.astype(int),
                            'role': 'Listener' # По умолчанию все слушатели
                        })
                    
                    # --- 2. КТО ГОВОРИТ? ---
                    # Если есть знакомые - проверяем голос
                    has_known = False
                    for u in visible_users:
                        if u['id'] != 'Unknown':
                            has_known = True
                            break
                    
                    if has_known:
                        # Временно освобождаем камеру
                        cap.release()
                        robot_state["subtext"] = "Listening..."
                        
                        try:
                            # Пишем 2 секунды (достаточно для верификации)
                            check_sound = sd.rec(int(2 * NATIVE_RATE), samplerate=NATIVE_RATE, channels=1, blocking=True)
                            
                            # Если звук достаточно громкий
                            if np.max(np.abs(check_sound)) > 0.05:
                                v_emb = get_voice_embedding(check_sound)
                                speaker_id, score = find_speaker_in_group(v_emb, visible_users)
                                
                                if speaker_id:
                                    # Нашли болтуна!
                                    for user in visible_users:
                                        if user['id'] == speaker_id:
                                            user['role'] = 'SPEAKER'
                                    print(f">>> ГОВОРИТ: {speaker_id} ({int(score*100)}%)")
                                else:
                                    print(">>> Голос чужой.")
                        except: pass
                        
                        # Возвращаем камеру
                        cap = cv2.VideoCapture(0)
                        if not cap.isOpened(): cap = cv2.VideoCapture(1)
                        cap.set(3, 320); cap.set(4, 240)

                    # Если незнакомцы - проверяем жесты
                    for user in visible_users:
                        if user['id'] == "Unknown":
                            if is_waving(frame):
                                if cap.isOpened(): cap.release()
                                run_registration()
                                robot_state["mode"] = "AWAKE"
                                cached_people = []
                                continue

                    cached_people = visible_users # Обновляем глобальный кэш
            
            # --- 3. ОТРИСОВКА (ИЗ КЭША) ---
            if cached_people:
                last_activity = time.time()
                
                for p in cached_people:
                    bbox = p['bbox']
                    role = p['role']
                    uid = p['id']
                    
                    if role == 'SPEAKER':
                        color = (0, 255, 0) # Зеленый (Говорит)
                        text = f"SPEAKING: {uid}"
                    elif uid == "Unknown":
                        color = (0, 0, 255) # Красный
                        text = "Unknown (Wave?)"
                    else:
                        color = (0, 255, 255) # Желтый (Слушает)
                        text = f"{uid} (Silent)"
                    
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                    cv2.putText(frame, text, (bbox[0], bbox[1]-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                robot_state["status"] = "MONITORING"
                robot_state["subtext"] = f"People: {len(cached_people)}"
            else:
                robot_state["status"] = "SEARCHING"
                robot_state["subtext"] = "..."

            # Таймер сна
            if time.time() - last_activity > SLEEP_TIMEOUT:
                if cap and cap.isOpened(): cap.release()
                robot_state["mode"] = "SLEEP"
                continue

            # Интерфейс
            cv2.rectangle(frame, (0, 0), (320, 30), (0,0,0), -1)
            cv2.putText(frame, robot_state["status"], (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, robot_state["color"], 1)
            
            cv2.rectangle(frame, (0, 210), (320, 240), (0,0,0), -1)
            cv2.putText(frame, robot_state["subtext"], (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            with lock: outputFrame = frame.copy()

# --- 7. WEB ---
@app.route("/")
def index():
    return '<html><body style="background:#000;color:#0f0;text-align:center;"><h1>GROUP ID</h1><img src="/video_feed" style="width:100%;max-width:640px;border:2px solid #333;"></body></html>'

def gen():
    global outputFrame
    while True:
        with lock:
            if outputFrame is None: continue
            (flag, enc) = cv2.imencode(".jpg", outputFrame)
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(enc) + b'\r\n')
        time.sleep(0.05)

@app.route("/video_feed")
def video_feed(): return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    t = threading.Thread(target=logic_loop)
    t.daemon = True
    t.start()
    app.run(host="0.0.0.0", port=5000, debug=False)