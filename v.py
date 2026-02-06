import os
import shutil
import torch
import torchaudio
import json
import time
import numpy as np
import sounddevice as sd
import torchaudio.transforms as T
import torch.nn.functional as F

# --- !!! ГЛАВНЫЙ ФИКС (ВСТАВИТЬ В НАЧАЛО) !!! ---
# Без этого блока SpeechBrain будет вылетать с ошибкой
if not hasattr(torchaudio, 'list_audio_backends'):
    torchaudio.list_audio_backends = lambda: ['soundfile']
if not hasattr(torchaudio, 'get_audio_backend'):
    torchaudio.get_audio_backend = lambda: 'soundfile'
# ------------------------------------------------

# Теперь можно импортировать SpeechBrain
from speechbrain.inference.speaker import SpeakerRecognition

# --- НАСТРОЙКИ ---
DB_FILE = "voice_id_db.json"
VOICE_THRESHOLD = 0.35       # Порог похожести (0.35 - оптимально)
RECORD_SECONDS = 4           # Время записи

# --- АУДИО ПАРАМЕТРЫ ---
try:
    dev_info = sd.query_devices(kind='input')
    NATIVE_RATE = int(dev_info['default_samplerate'])
    print(f">>> Микрофон: {dev_info['name']} ({NATIVE_RATE} Hz)")
except:
    print(">>> Микрофон не найден, ставим 44100")
    NATIVE_RATE = 44100

resampler = T.Resample(NATIVE_RATE, 16000)

# --- ЗАГРУЗКА МОДЕЛЕЙ ---
print(">>> [1/2] Загрузка VAD (Слух)...")
vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', trust_repo=True)

print(">>> [2/2] Загрузка SpeechBrain (Голос)...")
speaker_model = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="tmp_model")

# --- БАЗА ДАННЫХ (ID) ---
def load_db():
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, "r") as f: return json.load(f)
        except: pass
    return {}

def get_next_id():
    """Генерирует ID: 00001, 00002..."""
    db = load_db()
    return f"{len(db) + 1:05d}"

def save_new_voice(vector):
    """Сохраняет новый голос"""
    new_id = get_next_id()
    db = load_db()
    db[new_id] = vector
    with open(DB_FILE, "w") as f: json.dump(db, f)
    print(f"\n[+++] БАЗА ОБНОВЛЕНА: Новый ID {new_id}")
    return new_id

def identify_speaker(vector):
    db = load_db()
    
    # Если база пустая — создаем первого
    if not db:
        return None, 0.0, True
    
    best_id = None
    max_score = -1.0
    
    for uid, saved_vec in db.items():
        score = np.dot(vector, np.array(saved_vec))
        if score > max_score:
            max_score = score
            best_id = uid
            
    if max_score > VOICE_THRESHOLD:
        return best_id, max_score, False # Узнали
    else:
        return None, max_score, True     # Новый

# --- ОБРАБОТКА ---
def convert_audio(audio_np):
    waveform = torch.from_numpy(audio_np).float()
    if len(waveform.shape) == 1: waveform = waveform.unsqueeze(0)
    elif waveform.shape[0] != 1: waveform = waveform.t()
    return resampler(waveform)

def get_embedding(audio_data):
    wav_16k = convert_audio(audio_data)
    emb = speaker_model.encode_batch(wav_16k)
    return (emb.squeeze().cpu().numpy() / np.linalg.norm(emb.squeeze().cpu().numpy())).tolist()

def is_talking(audio_chunk):
    wav_16k = convert_audio(audio_chunk)
    target = 512
    if wav_16k.shape[-1] > target: wav_16k = wav_16k[..., :target]
    elif wav_16k.shape[-1] < target: wav_16k = F.pad(wav_16k, (0, target - wav_16k.shape[-1]))
    with torch.no_grad(): conf = vad_model(wav_16k, 16000).item()
    return conf > 0.3

# --- ГЛАВНЫЙ ЦИКЛ ---
def main():
    print("\n=== VOICE ID SYSTEM (AUTO) ===")
    print(">>> Ожидание голоса...")
    
    ratio = NATIVE_RATE / 16000
    block_size = int(np.ceil(512 * ratio))
    
    while True:
        voice_detected = False
        
        # 1. Слушаем эфир
        try:
            with sd.InputStream(samplerate=NATIVE_RATE, channels=1, dtype='float32', blocksize=block_size) as stream:
                while True:
                    chunk, _ = stream.read(block_size)
                    if is_talking(chunk):
                        print("\n🎤 ГОЛОС! Запись...", end="", flush=True)
                        voice_detected = True
                        break
        except Exception as e:
            print(f"Ошибка микрофона: {e}")
            time.sleep(1)
            continue

        # 2. Запись и анализ
        if voice_detected:
            try:
                # Пишем 4 секунды
                rec = sd.rec(int(RECORD_SECONDS * NATIVE_RATE), samplerate=NATIVE_RATE, channels=1, blocking=True)
                print(" Обработка...")
                
                # Вектор
                current_vec = get_embedding(rec)
                
                # Поиск
                uid, score, is_new = identify_speaker(current_vec)
                
                print("-" * 30)
                if is_new:
                    print(f"🆕 НЕЗНАКОМЫЙ ГОЛОС (Сходство: {int(score*100)}%)")
                    new_id = save_new_voice(current_vec)
                    print(f"   >>> ПРИСВОЕН ID: {new_id}")
                else:
                    print(f"✅ УЗНАЛ ID: {uid}")
                    print(f"   >>> Вероятность: {int(score*100)}%")
                print("-" * 30)
                
                time.sleep(1.5)
                print("👂 Слушаю дальше...")
                
            except Exception as e:
                print(f"Ошибка: {e}")

if __name__ == "__main__":
    main()