import asyncio
import os
import sys
import json
import sqlite3
import math
import time
import threading
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
from flask import Flask, Response

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
NN_SKIP_FRAMES = 15

outputFrame = None
frame_lock = threading.Lock()
group_app = Flask(__name__)

robot_state = {
    "status": "BOOTING...",
    "subtext": "Loading Group Logic...",
    "color": (255, 255, 255),
    "mode": "SLEEP"
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
    speaker_model = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="tmp_model")
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
    return False

def run_registration():
    global robot_state
    robot_state["status"] = "REGISTRATION"
    robot_state["subtext"] = "Freeze..."
    robot_state["color"] = (255, 0, 255)
    time.sleep(1.0)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)
    cap.set(3, 320)
    cap.set(4, 240)
    ret, frame = cap.read()
    cap.release()
    if not ret:
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
        rec_voice = sd.rec(int(4 * NATIVE_RATE), samplerate=NATIVE_RATE, channels=1, blocking=True)
        voice_emb = get_voice_embedding(rec_voice)
    except Exception:
        return
    db = load_robot_db()
    db[new_id] = {"face_vec": face_emb, "voice_vec": voice_emb, "created_at": time.time()}
    save_robot_db(db)
    robot_state["status"] = "SAVED"
    robot_state["subtext"] = new_id
    robot_state["color"] = (0, 255, 0)
    time.sleep(2)

def logic_loop():
    global outputFrame, robot_state
    init_group_models()
    ratio = NATIVE_RATE / 16000
    block_size = int(np.ceil(512 * ratio))
    robot_state["mode"] = "SLEEP"
    last_activity = time.time()
    cap = None
    frame_counter = 0
    cached_people = []
    print(">>> [SYSTEM] Group Logic Active.")
    while True:
        if robot_state["mode"] == "SLEEP":
            robot_state["status"] = "SLEEP MODE"
            robot_state["subtext"] = "Silence..."
            robot_state["color"] = (100, 100, 100)
            # Закрываем камеру в режиме SLEEP
            if cap is not None and cap.isOpened():
                cap.release()
                cap = None
            # Очищаем кадр
            if outputFrame is not None:
                with frame_lock:
                    outputFrame = None
            # Только слушаем микрофон, не общаемся
            try:
                with sd.InputStream(samplerate=NATIVE_RATE, channels=1, dtype='float32', blocksize=block_size) as stream:
                    while robot_state["mode"] == "SLEEP":
                        chunk, _ = stream.read(block_size)
                        if not is_silence(chunk):
                            print(">>> ЗВУК! Переход в режим AWAKE")
                            robot_state["mode"] = "AWAKE"
                            last_activity = time.time()
                            frame_counter = 0
                            break
            except Exception:
                time.sleep(1)
        elif robot_state["mode"] == "AWAKE":
            if cap is None or not cap.isOpened():
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    cap = cv2.VideoCapture(1)
                cap.set(3, 320)
                cap.set(4, 240)
            ret, frame = cap.read()
            if not ret:
                continue
            frame_counter += 1
            if frame_counter % NN_SKIP_FRAMES == 0:
                faces = face_app.get(frame)
                cached_people = []
                if faces:
                    # Проверяем махание рукой ПЕРЕД обработкой лиц
                    waving_detected = is_waving(frame)
                    
                    visible_users = []
                    for face in faces:
                        # Показываем лица, но НЕ записываем их автоматически
                        # Проверяем только для отображения, не сохраняем в базу
                        fid, fscore = identify_person_visual(face.normed_embedding)
                        if fscore < FACE_SIM_THRESHOLD:
                            fid = "Unknown"
                        visible_users.append({
                            'id': fid,
                            'face_emb': face.normed_embedding,
                            'bbox': face.bbox.astype(int),
                            'role': 'Listener'
                        })
                    
                    # Если человек машет рукой - запускаем регистрацию
                    if waving_detected:
                        print(">>> Обнаружено махание рукой! Запуск регистрации...")
                        if cap.isOpened():
                            cap.release()
                        run_registration()
                        robot_state["mode"] = "AWAKE"
                        cached_people = []
                        continue
                    
                    has_known = any(u['id'] != 'Unknown' for u in visible_users)
                    if has_known:
                        cap.release()
                        robot_state["subtext"] = "Listening..."
                        try:
                            check_sound = sd.rec(int(2 * NATIVE_RATE), samplerate=NATIVE_RATE, channels=1, blocking=True)
                            if np.max(np.abs(check_sound)) > 0.05:
                                v_emb = get_voice_embedding(check_sound)
                                speaker_id, score = find_speaker_in_group(v_emb, visible_users)
                                if speaker_id:
                                    for user in visible_users:
                                        if user['id'] == speaker_id:
                                            user['role'] = 'SPEAKER'
                                    print(f">>> ГОВОРИТ: {speaker_id} ({int(score*100)}%)")
                                else:
                                    print(">>> Голос чужой.")
                        except Exception:
                            pass
                        cap = cv2.VideoCapture(0)
                        if not cap.isOpened():
                            cap = cv2.VideoCapture(1)
                        cap.set(3, 320)
                        cap.set(4, 240)
                    cached_people = visible_users
            if cached_people:
                last_activity = time.time()
                for p in cached_people:
                    bbox = p['bbox']
                    role = p['role']
                    uid = p['id']
                    if role == 'SPEAKER':
                        color = (0, 255, 0)
                        text = f"SPEAKING: {uid}"
                    elif uid == "Unknown":
                        color = (0, 0, 255)
                        text = "Unknown (Wave to register)"
                    else:
                        color = (0, 255, 255)
                        text = f"{uid} (Silent)"
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                    cv2.putText(frame, text, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                robot_state["status"] = "MONITORING"
                robot_state["subtext"] = f"People: {len(cached_people)}"
            else:
                robot_state["status"] = "SEARCHING"
                robot_state["subtext"] = "..."
            if time.time() - last_activity > SLEEP_TIMEOUT:
                if cap and cap.isOpened():
                    cap.release()
                robot_state["mode"] = "SLEEP"
                continue
            # Рисуем руки используя MediaPipe Hands
            try:
                if hands_detector is not None:
                    frame.flags.writeable = False
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    hands_results = hands_detector.process(rgb_frame)
                    frame.flags.writeable = True
                    if hands_results.multi_hand_landmarks:
                        for hand_landmarks in hands_results.multi_hand_landmarks:
                            mp_drawing.draw_landmarks(
                                frame,
                                hand_landmarks,
                                mp.solutions.hands.HAND_CONNECTIONS,
                                mp_drawing_styles.get_default_hand_landmarks_style(),
                                mp_drawing_styles.get_default_hand_connections_style(),
                            )
            except Exception:
                pass
            
            cv2.rectangle(frame, (0, 0), (320, 30), (0, 0, 0), -1)
            cv2.putText(frame, robot_state["status"], (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, robot_state["color"], 1)
            cv2.rectangle(frame, (0, 210), (320, 240), (0, 0, 0), -1)
            cv2.putText(frame, robot_state["subtext"], (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            with frame_lock:
                outputFrame = frame.copy()

def run_flask_group_id():
    group_app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)

@group_app.route("/")
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

@group_app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

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
    threading.Thread(target=logic_loop, daemon=True).start()
    threading.Thread(target=run_flask_group_id, daemon=True).start()
    print("Group ID: http://0.0.0.0:5000/ (video feed)")

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
        # Обрабатываем транскрипты только в режиме AWAKE
        if robot_state["mode"] != "AWAKE":
            print(f"Ignoring transcript in SLEEP mode: {transcript}")
            return
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
        except Exception as conn_err:
            # WebSocket/connection error — print details and re-raise
            import traceback
            print("Failed to connect to Realtime API:")
            print(repr(conn_err))
            print(traceback.format_exc())

            # Try to detect common causes
            # e.g., websockets.exceptions.InvalidStatusCode has .status_code
            status = None
            if hasattr(conn_err, 'status_code'):
                status = getattr(conn_err, 'status_code')
            elif hasattr(conn_err, 'response') and hasattr(conn_err.response, 'status_code'):
                status = conn_err.response.status_code

            if status:
                print(f"Server returned status code: {status}")
                if status >= 500:
                    print("Server error (5xx) — may be temporary. Try again later or check service status.")
            # Also print the sanitized API key prefix to confirm which key was used
            print(f"OPENAI_API_KEY preview: {masked}")
            raise

        message_handler = asyncio.create_task(realtime_client.handle_messages())

        print("Connected to OpenAI Realtime API!")
        print("Audio streaming will start automatically.")
        print("Use Ctrl+C or your service manager to stop the app.")
        print("")

        # Start continuous audio streaming
        streaming_task = asyncio.create_task(audio_handler.start_streaming(realtime_client))

        # Run until one of the tasks finishes or the process is interrupted
        await asyncio.gather(message_handler, streaming_task)

    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Save conversation on exit
        conversation_manager.save_conversation()

        audio_handler.stop_streaming()
        audio_handler.cleanup()
        await realtime_client.close()

if __name__ == "__main__":
    print("Starting Realtime API CLI with Knowledge Base...")
    asyncio.run(main())