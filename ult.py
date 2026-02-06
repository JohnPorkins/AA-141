import os
import shutil
import json
import time
import threading
import numpy as np
import cv2
from flask import Flask, Response

# --- 1. ОЧИСТКА ПРИ СТАРТЕ ---
DB_FILE = "robot_memory.json"
print(">>> [SYSTEM] Очистка базы данных...")
if os.path.exists(DB_FILE):
    os.remove(DB_FILE)
    print(">>> База удалена. Чистый старт.")

# --- 2. ФИКСЫ ---
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

# --- 3. НАСТРОЙКИ ---
MIC_DEVICE = None            
FACE_SIM_THRESHOLD = 0.5     
VOICE_SIM_THRESHOLD = 0.30   # Порог для голоса
WAKE_UP_THRESHOLD = 0.3      
SLEEP_TIMEOUT = 15           
NN_SKIP_FRAMES = 15          

outputFrame = None
lock = threading.Lock()
app = Flask(__name__)

robot_state = {
    "status": "BOOTING...",
    "subtext": "Loading...",
    "color": (255, 255, 255),
    "mode": "SLEEP"
}

# --- 4. ИНИЦИАЛИЗАЦИЯ ---
print(">>> [INIT] Загрузка нейросетей... (Ждите)")

try:
    dev_info = sd.query_devices(kind='input')
    NATIVE_RATE = int(dev_info['default_samplerate'])
except:
    NATIVE_RATE = 44100

resampler = T.Resample(NATIVE_RATE, 16000)
vad_model, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', trust_repo=True)
speaker_model = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="tmp_model")
face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=-1, det_size=(320, 320))
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6)

# --- 5. ФУНКЦИИ ---
def load_db():
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, "r") as f: return json.load(f)
        except: pass
    return {}

def save_db(db):
    with open(DB_FILE, "w") as f: json.dump(db, f)

def get_next_id():
    db = load_db()
    return f"User_{len(db) + 1:05d}"

def convert_audio(audio_np):
    waveform = torch.from_numpy(audio_np).float()
    if len(waveform.shape) == 1: waveform = waveform.unsqueeze(0)
    elif waveform.shape[0] != 1: waveform = waveform.t()
    return resampler(waveform)

def get_voice_embedding(audio_data):
    wav_16k = convert_audio(audio_data)
    emb = speaker_model.encode_batch(wav_16k)
    return (emb.squeeze().cpu().numpy() / np.linalg.norm(emb.squeeze().cpu().numpy())).tolist()

def is_silence(audio_chunk):
    wav_16k = convert_audio(audio_chunk)
    target = 512
    if wav_16k.shape[-1] > target: wav_16k = wav_16k[..., :target]
    elif wav_16k.shape[-1] < target: wav_16k = torch.nn.functional.pad(wav_16k, (0, target - wav_16k.shape[-1]))
    with torch.no_grad(): conf = vad_model(wav_16k, 16000).item()
    return conf < WAKE_UP_THRESHOLD

def identify_person(face_emb):
    db = load_db()
    best_id = "Unknown"; max_score = 0
    for uid, data in db.items():
        score = np.dot(face_emb, np.array(data["face_vec"]))
        if score > max_score: max_score = score; best_id = uid
    return best_id, max_score

def verify_voice(user_id, voice_emb):
    """Сверяет текущий голос с сохраненным в базе для user_id"""
    db = load_db()
    if user_id not in db: return 0.0
    
    saved_voice = db[user_id].get("voice_vec")
    if saved_voice is None: return 0.0
    
    score = np.dot(voice_emb, np.array(saved_voice))
    return score

def is_waving(frame):
    try:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands_detector.process(rgb)
        if res.multi_hand_landmarks:
            lms = res.multi_hand_landmarks[0].landmark
            return lms[8].y < lms[0].y
    except: pass
    return False

# --- 6. ПОДРОБНАЯ РЕГИСТРАЦИЯ ---
def run_registration_detailed():
    global robot_state
    
    print("\n" + "="*40)
    print(">>> [РЕГ] НАЧАЛО РЕГИСТРАЦИИ НОВОГО ПОЛЬЗОВАТЕЛЯ")
    
    robot_state["status"] = "REGISTRATION"
    robot_state["subtext"] = "Freeze for Photo..."
    robot_state["color"] = (255, 0, 255) # Магента
    time.sleep(1.5)
    
    # 1. ФОТО
    print(">>> [РЕГ] Делаю фото...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): cap = cv2.VideoCapture(1)
    cap.set(3, 320); cap.set(4, 240)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("!!! [РЕГ] Ошибка камеры.")
        return

    faces = face_app.get(frame)
    if not faces:
        print("!!! [РЕГ] Лицо не найдено.")
        return
        
    face_emb = faces[0].normed_embedding.tolist()
    print(">>> [РЕГ] Фото успешно обработано и зарегистрировано.")
    
    # 2. ГОЛОС
    new_id = get_next_id()
    robot_state["subtext"] = f"SPEAK NOW! ({new_id})"
    print(f">>> [РЕГ] Запись голоса для {new_id} (4 сек)...")
    
    try:
        rec_voice = sd.rec(int(4 * NATIVE_RATE), samplerate=NATIVE_RATE, channels=1, blocking=True)
        print(">>> [РЕГ] Обработка голоса...")
        voice_emb = get_voice_embedding(rec_voice)
        print(f">>> [РЕГ] Голос зарегистрирован для {new_id}.")
    except:
        print("!!! [РЕГ] Ошибка микрофона.")
        return

    # 3. СОХРАНЕНИЕ
    db = load_db()
    db[new_id] = {
        "face_vec": face_emb,
        "voice_vec": voice_emb,
        "created_at": time.time()
    }
    save_db(db)
    
    print(f">>> [РЕГ] УСПЕХ! Пользователь {new_id} полностью добавлен в базу.")
    print("="*40 + "\n")
    
    robot_state["status"] = "SUCCESS"
    robot_state["subtext"] = f"Welcome {new_id}"
    robot_state["color"] = (0, 255, 0)
    time.sleep(2)

# --- 7. ЛОГИКА (ГЛАЗА + УШИ) ---
def logic_loop():
    global outputFrame, robot_state
    
    ratio = NATIVE_RATE / 16000
    block_size = int(np.ceil(512 * ratio))
    
    robot_state["mode"] = "SLEEP"
    last_activity = time.time()
    cap = None
    
    frame_counter = 0
    cached_faces = []
    
    print(">>> [SYSTEM] Логика запущена.")
    
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
                            print(">>> [СЛУХ] Звук обнаружен! Просыпаюсь...")
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
            
            # --- НЕЙРОСЕТЬ (раз в N кадров) ---
            if frame_counter % NN_SKIP_FRAMES == 0:
                faces = face_app.get(frame)
                cached_faces = faces
                
                # ЕСЛИ ЛИЦО НАЙДЕНО -> ЗАПУСКАЕМ ПРОВЕРКУ
                if faces:
                    face = faces[0]
                    fid, fscore = identify_person(face.normed_embedding)
                    
                    if fscore > FACE_SIM_THRESHOLD:
                        # 1. ЛИЦО УЗНАЛИ
                        print(f"\n>>> [ЛИЦО] OK: {fid} ({int(fscore*100)}%)")
                        robot_state["status"] = f"ID: {fid}"
                        robot_state["color"] = (0, 255, 0)
                        
                        # 2. ТЕПЕРЬ ПРОВЕРЯЕМ ГОЛОС
                        # (Делаем это редко, чтобы не спамить, например раз в 50 кадров, если лицо в кадре)
                        # Но для теста сделаем при каждом распознавании (учтите, это остановит видео на 3 сек)
                        
                        # Временное отключение камеры для аудио
                        cap.release() 
                        robot_state["subtext"] = "Verifying Voice..."
                        print(">>> [ГОЛОС] Слушаю для проверки (3 сек)...")
                        
                        try:
                            rec_check = sd.rec(int(3 * NATIVE_RATE), samplerate=NATIVE_RATE, channels=1, blocking=True)
                            v_emb = get_voice_embedding(rec_check)
                            v_score = verify_voice(fid, v_emb)
                            
                            if v_score > VOICE_SIM_THRESHOLD:
                                print(f">>> [ГОЛОС] OK! Это он ({int(v_score*100)}%)")
                                robot_state["subtext"] = "Voice Confirmed: OK"
                            else:
                                print(f"!!! [ВНИМАНИЕ] Голос НЕ совпадает! ({int(v_score*100)}%)")
                                print("!!! [ВНИМАНИЕ] С нами говорит другой человек.")
                                robot_state["subtext"] = "Voice MISMATCH! Other person."
                                robot_state["color"] = (0, 165, 255) # Оранжевый (Warning)
                        except:
                            pass
                        
                        # Возвращаем камеру
                        cap = cv2.VideoCapture(0)
                        if not cap.isOpened(): cap = cv2.VideoCapture(1)
                        cap.set(3, 320); cap.set(4, 240)
                        
                    else:
                        # НЕЗНАКОМЕЦ
                        print(">>> [ЛИЦО] Незнакомец.")
                        robot_state["status"] = "UNKNOWN"
                        robot_state["subtext"] = "Wave to Register"
                        robot_state["color"] = (0, 0, 255)
                        
                        if is_waving(frame):
                            if cap.isOpened(): cap.release()
                            run_registration_detailed()
                            robot_state["mode"] = "AWAKE"
                            cached_faces = []
                            continue
            
            else:
                faces = cached_faces # Используем кэш для отрисовки рамки, пока нет новой
            
            # --- ОТРИСОВКА ---
            if faces:
                last_activity = time.time() # Продлеваем жизнь
                bbox = faces[0].bbox.astype(int)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), robot_state["color"], 2)
            else:
                robot_state["status"] = "SEARCHING..."
                robot_state["subtext"] = "Looking for face..."
                robot_state["color"] = (255, 255, 0)

            # Таймер сна
            if time.time() - last_activity > SLEEP_TIMEOUT:
                if cap and cap.isOpened(): cap.release()
                robot_state["mode"] = "SLEEP"
                print(">>> [СОН] Никого нет. Ухожу в ожидание.")
                continue

            # Интерфейс
            cv2.rectangle(frame, (0, 0), (320, 30), (0,0,0), -1)
            cv2.putText(frame, robot_state["status"], (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, robot_state["color"], 1)
            
            cv2.rectangle(frame, (0, 210), (320, 240), (0,0,0), -1)
            cv2.putText(frame, robot_state["subtext"], (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            with lock: outputFrame = frame.copy()

# --- 8. ВЕБ СЕРВЕР ---
@app.route("/")
def index():
    return """
    <html>
    <head>
        <title>Robot Interface</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { 
                background-color: #121212; 
                color: #00ff00; 
                font-family: monospace; 
                text-align: center; 
                margin: 0; padding: 20px;
            }
            .container {
                display: inline-block;
                border: 2px solid #333;
                background: #000;
            }
            img {
                width: 100%;
                max-width: 640px; 
                height: auto;
                display: block;
            }
        </style>
    </head>
    <body>
        <h1>🤖 ROBOT SYSTEM</h1>
        <div class="container">
            <img src="/video_feed">
        </div>
        <p>Status: Active | Logs in Terminal</p>
    </body>
    </html>
    """

def gen():
    global outputFrame
    while True:
        with lock:
            if outputFrame is None: continue
            (flag, enc) = cv2.imencode(".jpg", outputFrame)
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(enc) + b'\r\n')
        time.sleep(0.05)

@app.route("/video_feed")
def video_feed():
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    t = threading.Thread(target=logic_loop)
    t.daemon = True
    t.start()
    app.run(host="0.0.0.0", port=5000, debug=False)