import cv2
import time
import threading
import numpy as np
import mediapipe as mp
from flask import Flask, Response
from collections import deque # Для хранения истории движений

# --- НАСТРОЙКИ ---
HISTORY_LENGTH = 15      # Сколько кадров помнить (примерно 0.5 - 1 сек)
WAVE_THRESHOLD = 0.05    # Насколько сильно нужно махать (чувствительность)
HAND_UP_THRESHOLD = 0.1  # Насколько рука должна быть выше плеча

# Глобальные переменные
outputFrame = None
lock = threading.Lock()
app = Flask(__name__)

# --- MEDIAPIPE ---
print(">>> Загрузка модели Pose...")
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1)

# --- ИСТОРИЯ ДВИЖЕНИЙ ---
# Deque - это список с фиксированной длиной. Когда он заполняется, старые данные выпадают.
history_left_x = deque(maxlen=HISTORY_LENGTH)
history_right_x = deque(maxlen=HISTORY_LENGTH)

def analyze_gesture(landmarks):
    """Анализирует махание и типы приветствий"""
    
    # Координаты (11/12 - плечи, 15/16 - запястья)
    l_wrist = landmarks[15]
    r_wrist = landmarks[16]
    l_shldr = landmarks[11]
    r_shldr = landmarks[12]
    
    # 1. Обновляем историю X координат (движение влево-вправо)
    history_left_x.append(l_wrist.x)
    history_right_x.append(r_wrist.x)
    
    # Пропускаем, если истории мало
    if len(history_left_x) < HISTORY_LENGTH:
        return "GATHERING DATA...", (100, 100, 100)

    # 2. Вычисляем "Энергию махания" (дисперсия координат X)
    # Если рука двигается туда-сюда, дисперсия будет большой.
    # Если рука стоит - маленькой.
    left_energy = np.std(history_left_x)
    right_energy = np.std(history_right_x)
    
    # 3. Проверка: Рука поднята? (Запястье выше плеча + запас)
    left_up = l_wrist.y < (l_shldr.y - HAND_UP_THRESHOLD)
    right_up = r_wrist.y < (r_shldr.y - HAND_UP_THRESHOLD)
    
    # --- ЛОГИКА ЖЕСТОВ ---
    
    # ВАРИАНТ 1: ДВЕ РУКИ (Энтузиазм)
    if left_up and right_up:
        if left_energy > WAVE_THRESHOLD or right_energy > WAVE_THRESHOLD:
            return "DOUBLE WAVE! \o/", (255, 0, 255) # Магента
        else:
            return "HANDS UP (PEACE)", (0, 255, 255) # Желтый

    # ВАРИАНТ 2: ЛЕВАЯ РУКА
    if left_up:
        if left_energy > WAVE_THRESHOLD:
            return "WAVING (Left) 👋", (0, 255, 0) # Зеленый
        else:
            return "STATIC HI (Left) ✋", (200, 200, 0) # Тускло-желтый

    # ВАРИАНТ 3: ПРАВАЯ РУКА
    if right_up:
        if right_energy > WAVE_THRESHOLD:
            return "WAVING (Right) 👋", (0, 255, 0) # Зеленый
        else:
            return "STATIC HI (Right) ✋", (200, 200, 0) # Тускло-желтый

    return "...", (50, 50, 50) # Ничего

# --- КАМЕРА ---
def camera_loop():
    global outputFrame
    
    print(">>> ЗАПУСК КАМЕРЫ...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): cap = cv2.VideoCapture(1)
    
    cap.set(3, 640)
    cap.set(4, 480)

    while True:
        ret, frame = cap.read()
        if not ret: time.sleep(0.1); continue
            
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        
        status = "NO BODY"
        color = (100, 100, 100)
        
        if results.pose_landmarks:
            # Рисуем скелет
            mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Анализ
            status, color = analyze_gesture(results.pose_landmarks.landmark)
        
        # Рисуем статус
        cv2.rectangle(frame, (0, 0), (640, 50), (0,0,0), -1)
        cv2.putText(frame, status, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        with lock: outputFrame = frame.copy()

# --- ВЕБ ---
@app.route("/")
def index():
    return """
    <html>
    <head><title>Gesture ID</title></head>
    <body style="background:#111; color:white; text-align:center; font-family:monospace;">
        <h1>GREETING DETECTOR</h1>
        <img src="/video_feed" style="width:90%; border:4px solid #333;">
        <h3>Попробуйте:</h3>
        <p>1. Просто поднять руку (Static)</p>
        <p>2. Помахать рукой влево-вправо (Wave)</p>
        <p>3. Поднять две руки (Double)</p>
    </body>
    </html>
    """

def gen():
    global outputFrame
    while True:
        with lock:
            if outputFrame is None: continue
            (flag, enc) = cv2.imencode(".jpg", outputFrame)
            if not flag: continue
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(enc) + b'\r\n')
        time.sleep(0.04)

@app.route("/video_feed")
def video_feed(): return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    t = threading.Thread(target=camera_loop)
    t.daemon = True
    t.start()
    app.run(host="0.0.0.0", port=5000, debug=False)