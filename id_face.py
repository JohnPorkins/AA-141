import cv2
import numpy as np
import os
import json
import time
import threading
from flask import Flask, Response
from insightface.app import FaceAnalysis

# --- НАСТРОЙКИ ЭКОНОМИИ ---
DB_FILE = "faces_memory.json"
SIMILARITY_THRESHOLD = 0.5 
FRAME_WIDTH = 320           # Минимальное разрешение
FRAME_HEIGHT = 240
GATHER_TIME = 3.0           # Секунд на сбор данных о лице
CROWD_LIMIT = 3             # Если лиц больше этого числа -> включаем режим толпы
CROWD_COOLDOWN = 30.0       # Секунд отдыха после толпы

# Глобальные переменные
outputFrame = None
lock = threading.Lock()
app = Flask(__name__)

# Состояние робота
state = {
    "collecting": [],       # Накопленные векторы за 3 сек
    "start_collect": 0,     # Время начала сбора
    "last_result_text": "", # Что показывать на экране
    "last_result_color": (255, 255, 255),
    "crowd_sleep_until": 0  # Время, до которого робот "спит"
}

# --- 1. ЗАГРУЗКА МОДЕЛИ ---
print(">>> Загрузка InsightFace (Buffalo_L)...")
# buffalo_l тяжелее, но точнее на низком разрешении. 
# Если будет тормозить - замените на 'buffalo_s'
face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=-1, det_size=(320, 320)) # Оптимизация под размер кадра

# --- 2. БАЗА ДАННЫХ ---
def load_db():
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, "r") as f: return json.load(f)
        except: pass
    return {}

def get_next_id():
    db = load_db()
    return f"{len(db) + 1:05d}" # Формат 00001

def save_user(embedding):
    new_id = get_next_id()
    db = load_db()
    db[new_id] = {"face_vec": embedding.tolist(), "created_at": time.time()}
    with open(DB_FILE, "w") as f: json.dump(db, f)
    return new_id

def identify_user(embedding):
    db = load_db()
    max_score = 0
    best_id = None
    for uid, data in db.items():
        score = np.dot(embedding, np.array(data["face_vec"]))
        if score > max_score:
            max_score = score
            best_id = uid
    
    if max_score > SIMILARITY_THRESHOLD:
        return best_id, max_score, False # False = не новый
    else:
        # Авто-регистрация нового
        new_id = save_user(embedding)
        return new_id, 1.0, True # True = новый

# --- 3. ЛОГИКА ЗРЕНИЯ ---
def camera_loop():
    global outputFrame, state
    
    print(">>> ЗАПУСК КАМЕРЫ (Low Res Mode)...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): cap = cv2.VideoCapture(1)
    
    # Жестко ставим низкое разрешение
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)

    while True:
        ret, frame = cap.read()
        if not ret: time.sleep(0.1); continue

        current_time = time.time()
        
        # --- ПРОВЕРКА: РЕЖИМ СНА (ТОЛПА) ---
        if current_time < state["crowd_sleep_until"]:
            # Робот отдыхает, просто показываем видео и таймер
            remaining = int(state["crowd_sleep_until"] - current_time)
            cv2.putText(frame, f"COOLING DOWN: {remaining}s", (10, 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            
            with lock: outputFrame = frame.copy()
            time.sleep(0.05) # Экономим CPU
            continue

        # --- АНАЛИЗ (НЕЙРОСЕТЬ) ---
        # Запускаем распознавание
        faces = face_app.get(frame)
        
        if not faces:
            # Лиц нет - сбрасываем накопление
            state["collecting"] = []
            state["start_collect"] = 0
            state["last_result_text"] = "WAITING FOR FACE..."
            state["last_result_color"] = (100, 100, 100)
        
        else:
            # Лица есть
            faces.sort(key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)
            main_face = faces[0]
            bbox = main_face.bbox.astype(int)
            
            # 1. ПРОВЕРКА НА ТОЛПУ
            if len(faces) > CROWD_LIMIT:
                # Если лиц слишком много
                cv2.putText(frame, "CROWD DETECTED!", (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # Даем поработать еще 4 секунды (чтобы успеть кого-то узнать), потом в сон
                if state["start_collect"] == 0:
                     state["start_collect"] = current_time # Начинаем отсчет 4 сек
                elif current_time - state["start_collect"] > 4.0:
                     state["crowd_sleep_until"] = current_time + CROWD_COOLDOWN
                     state["start_collect"] = 0
                     print(f">>> ТОЛПА! Ухожу в сон на {CROWD_COOLDOWN} сек.")
            
            # 2. СБОР ДАННЫХ (3 СЕКУНДЫ)
            if state["start_collect"] == 0:
                state["start_collect"] = current_time
                state["collecting"] = []
                state["last_result_text"] = "ANALYZING..."
                state["last_result_color"] = (255, 255, 0) # Cyan
            
            # Добавляем вектор в копилку
            state["collecting"].append(main_face.normed_embedding)
            
            elapsed = current_time - state["start_collect"]
            
            # Рисуем прогресс бар
            bar_width = int((elapsed / GATHER_TIME) * (bbox[2] - bbox[0]))
            cv2.rectangle(frame, (bbox[0], bbox[1]-10), (bbox[0]+bar_width, bbox[1]-5), (0, 255, 255), -1)

            # 3. ВРЕМЯ ПРИШЛО (ПРОШЛО 3 СЕК)
            if elapsed >= GATHER_TIME:
                # Усредняем
                if len(state["collecting"]) > 0:
                    avg_vec = np.mean(state["collecting"], axis=0)
                    avg_vec = avg_vec / np.linalg.norm(avg_vec)
                    
                    # ИДЕНТИФИКАЦИЯ
                    uid, score, is_new = identify_user(avg_vec)
                    
                    if is_new:
                        state["last_result_text"] = f"NEW ID: {uid}"
                        state["last_result_color"] = (255, 0, 255) # Магента
                    else:
                        state["last_result_text"] = f"ID: {uid} ({int(score*100)}%)"
                        state["last_result_color"] = (0, 255, 0) # Зеленый
                    
                    print(f">>> РЕЗУЛЬТАТ: {state['last_result_text']}")
                
                # Сброс таймера, чтобы начать проверку заново (или держать результат)
                state["collecting"] = []
                state["start_collect"] = current_time # Начинаем следующий цикл сразу
            
            # Рисуем результат
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), state["last_result_color"], 2)
            cv2.putText(frame, state["last_result_text"], (bbox[0], bbox[1]-15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, state["last_result_color"], 2)

        # Отправка в веб
        with lock: outputFrame = frame.copy()
        
        # Небольшая пауза, чтобы не грузить CPU на 100%
        time.sleep(0.03)

# --- 4. ВЕБ СЕРВЕР ---
@app.route("/")
def index():
    return """
    <html>
    <body style="background:black; color:white; text-align:center; font-family:monospace;">
        <h1>ROBOT OPTIMIZED VISION</h1>
        <img src="/video_feed" style="width:640px; height:480px; border:2px solid gray; image-rendering: pixelated;">
        <p>Resolution: 320x240 | Analysis: 3 sec avg | Crowd Mode: ON</p>
    </body>
    </html>
    """

def generate():
    global outputFrame
    while True:
        with lock:
            if outputFrame is None: continue
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
            if not flag: continue
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')
        time.sleep(0.05)

@app.route("/video_feed")
def video_feed(): return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    t = threading.Thread(target=camera_loop)
    t.daemon = True
    t.start()
    app.run(host="0.0.0.0", port=5000, debug=False)