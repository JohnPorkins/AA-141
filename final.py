import os
import shutil
import json
import time
import threading
import numpy as np
import cv2

# --- 1. ВАЖНО: ИМПОРТ TORCH И ФИКС (В САМОМ НАЧАЛЕ) ---
import torch
import torchaudio

# !!! ЭТОТ БЛОК ДОЛЖЕН БЫТЬ ДО ИМПОРТА SPEECHBRAIN !!!
if not hasattr(torchaudio, 'list_audio_backends'):
    torchaudio.list_audio_backends = lambda: ['soundfile']
if not hasattr(torchaudio, 'get_audio_backend'):
    torchaudio.get_audio_backend = lambda: 'soundfile'
# -----------------------------------------------------

# --- 2. ТЕПЕРЬ МОЖНО ИМПОРТИРОВАТЬ ОСТАЛЬНОЕ ---
import torchaudio.transforms as T
import torch.nn.functional as F
from speechbrain.inference.speaker import SpeakerRecognition
import sounddevice as sd
from insightface.app import FaceAnalysis
import mediapipe as mp

# --- 3. НАСТРОЙКИ ---
DB_FILE = "robot_memory.json"
MIC_DEVICE = None            
FACE_SIM_THRESHOLD = 0.5     
VOICE_SIM_THRESHOLD = 0.30   
WAKE_UP_THRESHOLD = 0.3      
SLEEP_TIMEOUT = 15           

# Шаблонные ответы
RESPONSES = {
    "User_00001": "Приветствую, Администратор! Системы в норме.",
    "default": "Здравствуйте! Рад вас видеть."
}

# --- 4. ИНИЦИАЛИЗАЦИЯ ---
print("\n>>> [SYSTEM] Загрузка нейросетей... (Ждите)")

# 4.1 Аудио
try:
    dev_info = sd.query_devices(kind='input')
    NATIVE_RATE = int(dev_info['default_samplerate'])
    print(f">>> Микрофон: {dev_info['name']} ({NATIVE_RATE} Hz)")
except:
    print(">>> Микрофон не найден, используем 44100")
    NATIVE_RATE = 44100

resampler = T.Resample(NATIVE_RATE, 16000)

vad_model, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', trust_repo=True)
speaker_model = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="tmp_model")

# 4.2 Видео
face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=-1, det_size=(320, 320))

mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# --- 5. ФУНКЦИИ БАЗЫ ---
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

# --- 6. ОБРАБОТКА ДАННЫХ ---
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
        wav_16k = F.pad(wav_16k, (0, target - wav_16k.shape[-1]))
    with torch.no_grad():
        conf = vad_model(wav_16k, 16000).item()
    return conf < WAKE_UP_THRESHOLD

def identify_person(face_emb, voice_emb=None):
    db = load_db()
    best_id = "Unknown"
    max_face_score = 0
    
    # Сначала ищем по лицу
    for uid, data in db.items():
        score = np.dot(face_emb, np.array(data["face_vec"]))
        if score > max_face_score:
            max_face_score = score
            if score > FACE_SIM_THRESHOLD:
                best_id = uid
    
    # Если есть голос, проверяем его для подтверждения
    if voice_emb is not None and best_id != "Unknown":
        saved_voice = db[best_id].get("voice_vec")
        if saved_voice:
            voice_score = np.dot(voice_emb, np.array(saved_voice))
            return best_id, max_face_score, voice_score
            
    return best_id, max_face_score, 0.0

def is_waving(frame):
    try:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands_detector.process(rgb)
        if res.multi_hand_landmarks:
            lms = res.multi_hand_landmarks[0].landmark
            return lms[8].y < lms[0].y
    except:
        pass
    return False

# --- 7. РЕГИСТРАЦИЯ ---
def run_registration():
    print("\n" + "="*40)
    print(">>> [РЕГИСТРАЦИЯ] ОБНАРУЖЕН ЖЕСТ!")
    print(">>> Робот: 'Я вас не знаю. Встаньте прямо и скажите имя.'")
    time.sleep(2.0)
    
    new_id = get_next_id()
    print(f">>> ID {new_id}: ГОВОРИТЕ (4 сек)...")
    
    # 1. Голос
    try:
        rec_voice = sd.rec(int(4 * NATIVE_RATE), samplerate=NATIVE_RATE, channels=1, blocking=True)
        voice_emb = get_voice_embedding(rec_voice)
    except Exception as e:
        print(f"!!! Ошибка звука: {e}")
        return

    # 2. Лицо
    print(">>> Фотографирую...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return
    faces = face_app.get(frame)
    if not faces:
        print("!!! Лицо не найдено. Попробуйте еще раз.")
        return
        
    face_emb = faces[0].normed_embedding.tolist()
    
    # 3. Сохранение
    db = load_db()
    db[new_id] = {
        "face_vec": face_emb,
        "voice_vec": voice_emb,
        "created_at": time.time()
    }
    save_db(db)
    
    print(f">>> [УСПЕХ] Вы записаны как {new_id}!")
    print("="*40 + "\n")

# --- 8. ГЛАВНЫЙ ЦИКЛ ---
def main():
    print("\n>>> [SYSTEM] РОБОТ ЗАПУЩЕН.")
    
    ratio = NATIVE_RATE / 16000
    block_size = int(np.ceil(512 * ratio))
    
    state = "SLEEP"
    last_activity = time.time()
    cap = None
    
    while True:
        # === СПИМ ===
        if state == "SLEEP":
            print("\r💤 [СОН] Тишина...", end="")
            try:
                with sd.InputStream(samplerate=NATIVE_RATE, channels=1, dtype='float32', blocksize=block_size) as stream:
                    while True:
                        chunk, _ = stream.read(block_size)
                        if not is_silence(chunk):
                            print("\n👂 [ПРОБУЖДЕНИЕ] Звук!")
                            state = "AWAKE"
                            last_activity = time.time()
                            break
            except Exception as e:
                print(f"Ошибка микрофона: {e}")
                time.sleep(1)
        
        # === РАБОТАЕМ ===
        elif state == "AWAKE":
            # 1. Включаем камеру (если надо)
            if cap is None or not cap.isOpened():
                print(">>> [КАМЕРА] Старт...")
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    cap = cv2.VideoCapture(1)
                cap.set(3, 320)
                cap.set(4, 240)
            
            ret, frame = cap.read()
            if not ret:
                continue
            
            # 2. Ищем лицо
            faces = face_app.get(frame)
            
            if faces:
                last_activity = time.time()
                face = faces[0]
                
                # Идентификация только по лицу (быстро)
                fid, fscore, _ = identify_person(face.normed_embedding)
                
                if fscore > FACE_SIM_THRESHOLD:
                    msg = RESPONSES.get(fid, RESPONSES["default"])
                    print(f"👀 Вижу: {fid} ({int(fscore*100)}%) -> {msg}")
                else:
                    print(f"👀 Незнакомец. Жду жеста...")
                    if is_waving(frame):
                        if cap.isOpened():
                            cap.release()
                        run_registration()
                        state = "AWAKE" # Сброс состояния
                        continue
            
            # 3. Таймер сна
            if time.time() - last_activity > SLEEP_TIMEOUT:
                print("\n💤 [ТАЙМ-АУТ] Засыпаю.")
                if cap and cap.isOpened():
                    cap.release()
                state = "SLEEP"

if __name__ == "__main__":
    main()