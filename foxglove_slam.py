import serial
import time
import threading
import math
import cv2
import numpy as np
import mediapipe as mp
import requests
from flask import Flask, Response

# ==========================================
# ⚙️ НАЛАШТУВАННЯ СИСТЕМИ
# ==========================================
CAMERA_FOV = 60 
MAX_RADAR_DIST_MM = 4000 
ARDUINO_URL = "http://192.168.4.1" 

app = Flask(__name__)
current_frame = None
latest_scan = [0] * 360  
current_robot_command = 'S' 

# ==========================================
# 🛠 ДОПОМІЖНІ ФУНКЦІЇ
# ==========================================
def send_command(cmd):
    """Відправка команд на Arduino (тільки якщо команда змінилася)"""
    global current_robot_command
    if cmd != current_robot_command:
        try:
            requests.get(f"{ARDUINO_URL}/{cmd}", timeout=0.1)
            current_robot_command = cmd
            print(f"🤖 ВІДПРАВЛЕНО: {cmd}")
        except:
            pass 

def create_tracker():
    """Створює OpenCV трекер (сумісність з різними версіями OpenCV)"""
    try: return cv2.TrackerKCF_create()
    except AttributeError:
        try: return cv2.legacy.TrackerKCF_create()
        except: return cv2.TrackerMIL_create()

def get_robust_distance(scan, target_angle, window_size=5):
    """
    ФІЛЬТР 1 (Просторовий): Бере медіану з сусідніх променів (±5 градусів).
    Прибирає помилкові нулі та "дірки" в даних лідара.
    """
    valid_dists = []
    for i in range(-window_size, window_size + 1):
        angle = int(target_angle + i) % 360
        dist = scan[angle]
        if 50 < dist < 8000: 
            valid_dists.append(dist)
            
    if not valid_dists: return None
    valid_dists.sort()
    return valid_dists[len(valid_dists) // 2] # Повертаємо медіану

def draw_radar_map(scan, target_angle=None, target_dist=None):
    """Малює чорний квадрат з радарною обстановкою"""
    size = 400 
    center = (size // 2, size // 2)
    scale = (size / 2) / MAX_RADAR_DIST_MM 
    radar_img = np.zeros((size, size, 3), dtype=np.uint8)

    # Круги дальності
    for r in [1000, 2000, 3000, 4000]:
        r_px = int(r * scale)
        cv2.circle(radar_img, center, r_px, (40, 40, 40), 1)

    # Зона видимості камери (синій сектор)
    fov_half = CAMERA_FOV / 2.0
    cv2.ellipse(radar_img, center, (int(MAX_RADAR_DIST_MM*scale), int(MAX_RADAR_DIST_MM*scale)), 
                -90, -fov_half, fov_half, (20, 20, 60), -1)

    cv2.line(radar_img, center, (center[0], 0), (0, 255, 0), 2) # Ніс робота

    # Малюємо перешкоди (білі крапки)
    for angle in range(360):
        dist = scan[angle]
        if 50 < dist < MAX_RADAR_DIST_MM:
            rad = math.radians(angle - 90)
            x = int(center[0] + dist * scale * math.cos(rad))
            y = int(center[1] + dist * scale * math.sin(rad))
            cv2.circle(radar_img, (x, y), 2, (255, 255, 255), -1)

    # Малюємо ціль (червона крапка)
    if target_angle is not None and target_dist is not None and target_dist > 0:
        rad = math.radians(target_angle - 90)
        x = int(center[0] + target_dist * scale * math.cos(rad))
        y = int(center[1] + target_dist * scale * math.sin(rad))
        cv2.line(radar_img, center, (x, y), (0, 0, 255), 1)
        cv2.circle(radar_img, (x, y), 6, (0, 0, 255), -1)

    cv2.circle(radar_img, center, 8, (0, 255, 255), -1) # Сам робот
    return radar_img

# ==========================================
# 📡 ПОТІК 1: ЧИТАННЯ ЛІДАРУ (Без SLAM - дуже швидко!)
# ==========================================
def lidar_thread():
    global latest_scan
    PORT_NAME = '/dev/ttyACM0'
    BAUDRATE = 230400
    try: lidar_serial = serial.Serial(PORT_NAME, BAUDRATE, timeout=1)
    except: return

    scan_data = [0] * 360 
    last_angle = 0.0
    buffer = bytearray() 

    while True:
        try:
            if lidar_serial.in_waiting > 0: buffer.extend(lidar_serial.read(lidar_serial.in_waiting))
            else: buffer.extend(lidar_serial.read(1))
                
            while len(buffer) >= 47:
                if buffer[0] == 0x54 and buffer[1] == 0x2C:
                    packet = buffer[:47]
                    del buffer[:47]
                    start_angle = (packet[4] + packet[5] * 256) / 100.0
                    end_angle = (packet[42] + packet[43] * 256) / 100.0
                    if start_angle > 360.0 or end_angle > 360.0: continue
                    step = (end_angle - start_angle)
                    if step < 0: step += 360.0
                    step /= 11.0
                    
                    if start_angle < last_angle - 180:
                        latest_scan = scan_data.copy()
                        scan_data = [0] * 360 
                        
                    for i in range(12):
                        distance = packet[6 + i*3] + packet[7 + i*3] * 256
                        if 50 < distance < 8000:
                            scan_data[int((start_angle + i * step) + 0.5) % 360] = distance
                    last_angle = start_angle
                else: del buffer[0:1]
        except: time.sleep(0.01)

# ==========================================
# 🎥 ПОТІК 2: КАМЕРА, ТРЕКІНГ ТА РУХ
# ==========================================
def camera_thread():
    global current_frame, latest_scan, current_robot_command
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    cap = cv2.VideoCapture(0)
    cap.set(3, 320); cap.set(4, 240)

    # Змінні Трекера та Згладжування
    tracking_active = False
    tracker = None
    lock_counter = 0
    smoothed_human_distance = None # Змінна для фільтру відстані

    while cap.isOpened():
        success, image = cap.read()
        if not success: continue
        h, w = image.shape[:2]

        target_angle = None
        human_distance = None
        angle_offset = 0
        cmd = 'S'

        # --- МАЛЮЄМО СІТКУ (HUD) ---
        cv2.line(image, (w//2, 0), (w//2, h), (255, 255, 255), 1) # Центр
        x_m15 = int(w/2 - (15.0 / CAMERA_FOV) * w)
        x_p15 = int(w/2 + (15.0 / CAMERA_FOV) * w)
        cv2.line(image, (x_m15, 0), (x_m15, h), (100, 255, 100), 1) 
        cv2.line(image, (x_p15, 0), (x_p15, h), (100, 255, 100), 1) 
        cv2.putText(image, "0", (w//2 + 5, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        cv2.putText(image, "-15", (x_m15 - 25, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100,255,100), 1)
        cv2.putText(image, "+15", (x_p15 + 5, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100,255,100), 1)

        # ==================================================
        # РЕЖИМ 1: ПОШУК (MediaPipe)
        # ==================================================
        if not tracking_active:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            
            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                pts = [lm[11], lm[12], lm[23], lm[24]] # Плечі та стегна
                xs, ys = [p.x * w for p in pts], [p.y * h for p in pts]
                x_min, x_max = int(min(xs)), int(max(xs))
                y_min, y_max = int(min(ys)), int(max(ys))
                
                pad_x, pad_y = int((x_max - x_min) * 0.2), int((y_max - y_min) * 0.2)
                bx, by = max(0, x_min - pad_x), max(0, y_min - pad_y)
                bw, bh = min(w, x_max - x_min + 2*pad_x), min(h, y_max - y_min + 2*pad_y)
                
                center_x = bx + bw/2.0
                angle_offset = (center_x / w - 0.5) * CAMERA_FOV
                target_angle = int(angle_offset) % 360
                
                # Отримуємо сиру відстань з просторовим фільтром
                raw_dist = get_robust_distance(latest_scan, target_angle, window_size=5)
                
                # ФІЛЬТР 2 (Часовий): Плавне згладжування
                if raw_dist is not None:
                    if smoothed_human_distance is None: smoothed_human_distance = raw_dist
                    else: smoothed_human_distance = int(0.2 * raw_dist + 0.8 * smoothed_human_distance)
                
                human_distance = smoothed_human_distance if smoothed_human_distance else 0

                # Малюємо ЖОВТУ рамку (Пошук) + підпис з відстанню
                cv2.rectangle(image, (bx, by), (bx+bw, by+bh), (0, 255, 255), 2)
                if human_distance and human_distance > 0:
                    label = f"SEARCH {human_distance} mm"
                else:
                    label = "SEARCHING..."
                cv2.putText(image, label, (bx, by-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                # Логіка захоплення (Центр + 2 секунди)
                if -10 <= angle_offset <= 10 and 400 < human_distance < 2000:
                    lock_counter += 1
                    cv2.circle(image, (w//2, h//2), lock_counter*2, (0, 255, 255), 2)
                    if lock_counter > 20: 
                        tracker = create_tracker()
                        tracker.init(image, (bx, by, bw, bh))
                        tracking_active = True
                else: lock_counter = 0 
            else:
                smoothed_human_distance = None # Скидаємо згладжування, якщо нікого немає

        # ==================================================
        # РЕЖИМ 2: ТРЕКІНГ ТА РУХ (OpenCV)
        # ==================================================
        else:
            success, bbox = tracker.update(image)
            
            if success:
                bx, by, bw, bh = [int(v) for v in bbox]
                center_x = bx + bw/2.0
                angle_offset = (center_x / w - 0.5) * CAMERA_FOV
                target_angle = int(angle_offset) % 360
                
                # Застосовуємо ті самі два фільтри для стабільності
                raw_dist = get_robust_distance(latest_scan, target_angle, window_size=5)
                if raw_dist is not None:
                    if smoothed_human_distance is None: smoothed_human_distance = raw_dist
                    else: smoothed_human_distance = int(0.2 * raw_dist + 0.8 * smoothed_human_distance)
                human_distance = smoothed_human_distance if smoothed_human_distance else 0

                # Малюємо ЗЕЛЕНУ рамку (Залочено) + підпис з відстанню
                cv2.rectangle(image, (bx, by), (bx+bw, by+bh), (0, 255, 0), 3)
                if human_distance and human_distance > 0:
                    label = f"LOCKED {human_distance} mm"
                else:
                    label = "LOCKED"
                cv2.putText(image, label, (bx, by-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # 🧠 ЛОГІКА РУХУ
                if human_distance > 100:
                    if human_distance < 600: cmd = 'B'
                    elif human_distance < 1000: cmd = 'S'
                    else:
                        if angle_offset < -15: cmd = 'L'
                        elif angle_offset > 15: cmd = 'R'
                        else: cmd = 'F'
            else:
                tracking_active = False
                lock_counter = 0
                smoothed_human_distance = None
                cmd = 'S'
                cv2.putText(image, "TARGET LOST!", (w//2-70, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        send_command(cmd)

        # Телеметрія на екран
        cv2.putText(image, f"Angle: {int(angle_offset)} deg", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        dist_text = f"Dist: {human_distance} mm" if human_distance else "Dist: LOST"
        cv2.putText(image, dist_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        cv2.putText(image, f"CMD: {current_robot_command}", (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        # Формування єдиної картинки (Камера + Радар)
        new_w = int(w * (400 / h))
        cam_resized = cv2.resize(image, (new_w, 400))
        radar_img = draw_radar_map(latest_scan, target_angle, human_distance)
        
        combined_img = np.hstack((cam_resized, radar_img))
        ret, buffer = cv2.imencode('.jpg', combined_img)
        if ret: current_frame = buffer.tobytes()

# ==========================================
# 🌐 WEB СЕРВЕР FLASK
# ==========================================
def generate_frames():
    global current_frame
    while True:
        if current_frame is not None: yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + current_frame + b'\r\n')
        else: time.sleep(0.05)

@app.route('/')
def index():
    return '''<!DOCTYPE html><html><head><title>Robot Follow System</title></head>
    <body style="background:#111; text-align:center; color:white; font-family:sans-serif;">
        <h3>Зліва: Camera Tracker (HUD) &nbsp;|&nbsp; Справа: 2D Радар</h3>
        <img src="/video_feed" style="max-width: 100%; border: 3px solid #00ffcc; border-radius: 5px;">
    </body></html>'''

@app.route('/video_feed')
def video_feed(): return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    threading.Thread(target=lidar_thread, daemon=True).start()
    threading.Thread(target=camera_thread, daemon=True).start()
    print("🚀 СИСТЕМА ЗАПУЩЕНА! Відкрийте http://<IP_RASPBERRY>:5000 у браузері.")
    app.run(host='0.0.0.0', port=5000, threaded=True)