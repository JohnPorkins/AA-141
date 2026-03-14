import cv2
import requests
import time
import threading
import numpy as np
import platform
import glob
import math
import os
from flask import Flask, Response, request, jsonify
import asyncio
from dotenv import load_dotenv

# Try to import Realtime streaming lib
try:
    from openai_realtime_client import RealtimeClient, AudioHandler, TurnDetectionMode
except ImportError:
    RealtimeClient = None

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

try:
    import serial
except ImportError:
    serial = None

# --- Configuration ---
SERVER_IP = "207.154.194.192"
URL_VISION = f"http://{SERVER_IP}:5000/process_vision"
URL_FACT = f"http://{SERVER_IP}:5000/save_fact"
URL_REGISTER = f"http://{SERVER_IP}:5000/register"

RASPBERRY_URL = "http://192.168.4.1" # The control endpoint
MAX_FRAME_DIM = 320 # Resize for faster transmission
NATIVE_RATE = 44100

# --- State ---
outputFrame = None
foxglove_current_frame = None
latest_scan = [0] * 360
latest_analysis = {"people": [], "is_waving": False, "tracking": None, "speaker_id": None, "transcription": None, "ai_response": None}
robot_state = {"status": "ONLINE", "subtext": "System Ready", "color": (0, 255, 0)}
chat_history = []  # List of {"role": "User/M141", "text": "...", "id": "..."}
last_command = "S"
frame_lock = threading.Lock()
registration_lock = threading.Lock()
vision_busy = False

# --- Audio Config (Now handled by RealtimeClient) ---
# We keep these empty to maintain vision backward compatibility 
def get_last_audio(duration_s=0.7):
    return None

app = Flask(__name__)

# --- Hardware I/O ---
def open_camera(width=640, height=480):
    is_linux = platform.system().lower() == "linux"
    backend = cv2.CAP_V4L2 if is_linux and hasattr(cv2, "CAP_V4L2") else None
    cap = cv2.VideoCapture(0, backend) if backend is not None else cv2.VideoCapture(0)
    if cap and cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Reduce lag
        return cap
    return None

def lidar_thread():
    global latest_scan
    if serial is None: return
    BAUDRATE = 230400
    port_candidates = ["/dev/ttyUSB0", "/dev/ttyACM0", "/dev/ttyAMA10"]
    ser = None
    for p in port_candidates:
        try:
            print("Trying", p)
            ser = serial.Serial(p, BAUDRATE, timeout=1)
            print(f"LiDAR connected on {p}")
            break
        except: continue
    if not ser: return
    
    buffer = bytearray()
    scan_data = [0] * 360
    last_angle = 0.0
    while True:
        try:
            if ser.in_waiting:
                buffer.extend(ser.read(ser.in_waiting))
                while len(buffer) >= 47:
                    if buffer[0] == 0x54 and buffer[1] == 0x2C:
                        packet = buffer[:47]; del buffer[:47]
                        start_angle = (packet[4] + packet[5] * 256) / 100.0
                        if start_angle > 360: continue
                        if start_angle < last_angle - 180:
                            latest_scan = scan_data.copy()
                            scan_data = [0] * 360
                        for i in range(12):
                            dist = packet[6 + i*3] + packet[7 + i*3] * 256
                            if 50 < dist < 8000:
                                scan_data[int((start_angle + i * 0.3) % 360)] = dist
                        last_angle = start_angle
                    else: del buffer[0:1]
            time.sleep(0.01)
        except: time.sleep(1)

def send_raspberry_command(cmd):
    global last_command
    if cmd != last_command:
        try:
            requests.get(f"{RASPBERRY_URL}/{cmd}", timeout=0.1)
            last_command = cmd
            print(f"🤖 CMD Sent: {cmd}")
        except: pass

def run_registration(frame_small):
    """Voice + Face registration triggered by waving."""
    global robot_state

    with registration_lock:
        print(">>> [REGISTRATION] Starting...")
        robot_state.update({"status": "REGISTRATION", "subtext": "ОБЛИЧЧЯ ЗНАЙДЕНО", "color": (255, 0, 255)})
        
        # 1. Package and send to server
        try:
            _, enc = cv2.imencode('.jpg', frame_small)
            # Only send the face frame; voice can be captured later or via realtime stream.
            files = {
                'frame': ('f.jpg', enc.tobytes(), 'image/jpeg')
            }
            r = requests.post(URL_REGISTER, files=files, timeout=10)
            if r.status_code == 200:
                new_id = r.json().get("id")
                print(f">>> [REGISTRATION] Success! Assigned ID: {new_id}")
                robot_state.update({"status": "SAVED", "subtext": f"Привіт {new_id}!", "color": (0, 255, 0)})
            else:
                print(f">>> [REGISTRATION] Failed! Server returned Code: {r.status_code}, Response: {r.text}")
                robot_state.update({"status": "ERROR", "subtext": "Registration failed", "color": (0, 0, 255)})
        except Exception as e:
            print(f"Reg err: {e}")
            robot_state.update({"status": "ERROR", "subtext": "Network err", "color": (0, 0, 255)})
        
        time.sleep(3)
        robot_state.update({"status": "ONLINE", "subtext": "System Ready", "color": (0, 255, 0)})

# --- Visualization Helpers ---
RADAR_MAX_DIST = 4000
CAMERA_FOV = 60

def draw_radar_map(scan, target_angle=None, target_dist=None):
    size = 400
    center = (size // 2, size // 2)
    scale = (size / 2) / RADAR_MAX_DIST
    radar_img = np.zeros((size, size, 3), dtype=np.uint8)
    
    # Draw range circles
    for r in [1000, 2000, 3000, 4000]:
        r_px = int(r * scale)
        cv2.circle(radar_img, center, r_px, (40, 40, 40), 1)
        
    # Draw FOV cone
    fov_half = CAMERA_FOV / 2.0
    cv2.ellipse(radar_img, center, (int(RADAR_MAX_DIST*scale), int(RADAR_MAX_DIST*scale)),
                -90, -fov_half, fov_half, (20, 20, 60), -1)
                
    # Center line
    cv2.line(radar_img, center, (center[0], 0), (0, 255, 0), 2)
    
    # Draw LiDAR points
    for angle in range(360):
        try:
            dist = scan[angle]
            if 50 < dist < RADAR_MAX_DIST:
                rad = math.radians(angle - 90)
                x = int(center[0] + dist * scale * math.cos(rad))
                y = int(center[1] + dist * scale * math.sin(rad))
                cv2.circle(radar_img, (x, y), 2, (255, 255, 255), -1)
        except:
            continue
            
    # Draw target if available
    if target_angle is not None and target_dist is not None and target_dist > 0:
        rad = math.radians(target_angle - 90)
        x = int(center[0] + target_dist * scale * math.cos(rad))
        y = int(center[1] + target_dist * scale * math.sin(rad))
    cv2.circle(radar_img, center, 8, (0, 255, 255), -1)
    return radar_img

def async_vision_worker(frame_small):
    """Network request in a separate thread to avoid blocking HUD/Control."""
    global latest_analysis, vision_busy
    try:
        _, enc = cv2.imencode('.jpg', frame_small)
        data = {'frame': ('f.jpg', enc.tobytes(), 'image/jpeg')}
            
        r = requests.post(URL_VISION, files=data, timeout=5.0)
        if r.status_code == 200:
            res = r.json()
            latest_analysis.update(res)
            
            # Transcription is now handled locally by RealtimeClient
            if res.get('is_waving') and robot_state["status"] == "ONLINE":
                threading.Thread(target=run_registration, args=(frame_small,), daemon=True).start()
    except Exception as e:
        print(f"Vision worker err: {e}")
    finally:
        vision_busy = False

# --- Vision Worker ---
def vision_worker():
    global outputFrame, foxglove_current_frame, latest_analysis, robot_state, vision_busy
    cap = open_camera()
    if not cap: return
    
    while True:
        ret, frame = cap.read()
        if not ret: continue
        
        h, w = frame.shape[:2]
        # Calculate scale to ensure max dimension is MAX_FRAME_DIM
        scale = MAX_FRAME_DIM / float(max(h, w))
        frame_small = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        # 1. Vision Processing (Non-blocking)
        if not vision_busy and robot_state["status"] == "ONLINE":
            vision_busy = True
            threading.Thread(target=async_vision_worker, args=(frame_small,), daemon=True).start()
            
        # 3. Drawing
        hud = frame.copy()
        inv_scale = 1.0 / scale
        people = latest_analysis.get('people', [])
        speaker_id = latest_analysis.get('speaker_id')

        for p in people:
            b = [int(v * inv_scale) for v in p['bbox']]
            role = "Говорить" if p['id'] == speaker_id else "Мовчить"
            color = (0, 255, 0) if role == "Говорить" else (0, 255, 255) if p['id'] != "Unknown" else (0, 0, 255)
            cv2.rectangle(hud, (b[0], b[1]), (b[2], b[3]), color, 2)
            label = f"{p['id']} [{role}]" if p['id'] != "Unknown" else "Незнайомець"
            cv2.putText(hud, label, (b[0], b[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Status Bar
        cv2.rectangle(hud, (0, h-30), (w, h), (0, 0, 0), -1)
        cv2.putText(hud, f"{robot_state['status']}: {robot_state['subtext']}", (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, robot_state["color"], 1)

        with frame_lock:
            outputFrame = hud
        
        # 4. Control Logic & Foxglove
        cmd = "S"
        tracking = latest_analysis.get('tracking')
        fox_image = frame.copy()
        if tracking:
            angle = tracking['angle']
            cmd = "L" if angle < -15 else "R" if angle > 15 else "F"
            tb = [int(v * inv_scale) for v in tracking['bbox']]
            cv2.rectangle(fox_image, (tb[0], tb[1]), (tb[2], tb[3]), (0, 255, 255), 2)
        
        # Smooth command sending
        if robot_state["status"] == "ONLINE":
            send_raspberry_command(cmd)

        # Foxglove stack
        target_angle = tracking['angle'] if tracking else None
        radar = draw_radar_map(latest_scan, target_angle=target_angle)
        combined = np.hstack((cv2.resize(fox_image, (400, 400)), radar))
        _, enc_combined = cv2.imencode(".jpg", combined)
        foxglove_current_frame = enc_combined.tobytes()

        time.sleep(0.01)

# --- Flask Routes ---
@app.route("/state")
def get_state():
    return jsonify({
        "robot": robot_state,
        "chat": chat_history,
        "analysis": {
            "people_count": len(latest_analysis.get('people', [])),
            "speaker": latest_analysis.get('speaker_id')
        }
    })

@app.route("/")
def index():
    return """
    <html>
    <head>
        <title>M141 Робот-Контроль</title>
        <style>
            body { background: #080808; color: #00ff00; font-family: 'Segoe UI', system-ui, sans-serif; margin: 0; padding: 20px; overflow: hidden; }
            .container { display: flex; gap: 20px; height: 95vh; }
            .feed-container { flex: 7; background: #000; padding: 10px; border: 1px solid #1a1a1a; display: flex; flex-direction: column; }
            .sidebar { flex: 3; background: #0a0a0a; border: 1px solid #1a1a1a; padding: 15px; display: flex; flex-direction: column; }
            .title-bar { display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid #004400; padding-bottom: 10px; margin-bottom: 15px; }
            .status-badge { font-size: 0.8em; padding: 3px 8px; border-radius: 3px; background: #002200; }
            .chat-box { flex-grow: 1; overflow-y: auto; padding-right: 10px; font-family: 'Courier New', monospace; }
            .message { margin-bottom: 12px; border-bottom: 1px solid #111; padding-bottom: 8px; }
            .role { font-weight: bold; color: #00ffaa; font-size: 0.9em; }
            .text { color: #ccc; font-size: 0.95em; margin-top: 2px; }
            .status-led { display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #00ff00; margin-right: 5px; }
            img { width: 100%; height: auto; border: 1px solid #222; }
            ::-webkit-scrollbar { width: 4px; }
            ::-webkit-scrollbar-track { background: #000; }
            ::-webkit-scrollbar-thumb { background: #003300; }
            .fox-link { display: block; margin-top: 10px; color: #008888; text-decoration: none; font-size: 0.8em; text-align: center; border: 1px dashed #008888; padding: 5px; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="feed-container">
                <div class="title-bar">
                    <div style="font-weight:bold; letter-spacing:2px;">M141_ULTRA_VISION</div>
                    <div id="robot-status">
                        <span class="status-led"></span><span id="status-text">ІНІЦІАЛІЗАЦІЯ...</span>
                    </div>
                </div>
                <img src="/video_feed">
            </div>
            <div class="sidebar">
                <div class="title-bar"><h3>ЛОГ_ДІАЛОГУ</h3></div>
                <div id="chat-box" class="chat-box"></div>
                <a href="/foxglove" class="fox-link">ВІДКРИТИ FOXGLOVE SLAM</a>
            </div>
        </div>

        <script>
            async function updateState() {
                try {
                    const res = await fetch('/state');
                    const data = await res.json();
                    
                    document.getElementById('status-text').innerText = `${data.robot.status}: ${data.robot.subtext}`;
                    const led = document.querySelector('.status-led');
                    led.style.background = `rgb(${data.robot.color[0]}, ${data.robot.color[1]}, ${data.robot.color[2]})`;

                    const chatBox = document.getElementById('chat-box');
                    const newHtml = data.chat.map(m => `
                        <div class="message">
                            <div class="role">[${m.role}]</div>
                            <div class="text">${m.text}</div>
                        </div>
                    `).join('');
                    
                    if (chatBox.innerHTML !== newHtml) {
                        chatBox.innerHTML = newHtml;
                        chatBox.scrollTop = chatBox.scrollHeight;
                    }
                } catch (e) {}
            }
            setInterval(updateState, 800);
        </script>
    </body>
    </html>
    """

@app.route("/video_feed")
def video_feed():
    def gen():
        while True:
            with frame_lock:
                if outputFrame is not None:
                    _, enc = cv2.imencode(".jpg", outputFrame)
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + enc.tobytes() + b'\r\n')
            time.sleep(0.1)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/foxglove")
def foxglove_index(): return '<html><body style="background:#111;text-align:center;"><img src="/foxglove/video_feed"></body></html>'

@app.route("/foxglove/video_feed")
def foxglove_video_feed():
    def gen():
        while True:
            if foxglove_current_frame:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + foxglove_current_frame + b'\r\n')
            time.sleep(0.1)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

def run_flask():
    app.run(host="0.0.0.0", port=5001, use_reloader=False)

async def start_realtime_stream():
    if RealtimeClient is None:
        print("RealtimeClient library not found! Audio streaming disabled.")
        return

    if not OPENAI_API_KEY:
        print("OPENAI_API_KEY not found in .env! Audio disabled.")
        return

    audio_handler = AudioHandler()
    
    # State to handle M141 chunks continuously without splitting lines
    def on_input_transcript(transcript):
        global chat_history
        if len(transcript.strip()) > 0:
            print(f">>> [SPEECH] You: {transcript}")
            chat_history.append({"role": "You", "text": transcript})
            if len(chat_history) > 12: chat_history = chat_history[-12:]

    def on_text_delta(text):
        pass # Some wrappers use this, but we handle audio transcript

    def on_output_transcript(transcript):
        global chat_history
        if len(transcript) > 0:
            print(f">>> [SPEECH] M141: {transcript}")
            # If the last message was M141, append to it instead of creating a new line
            if len(chat_history) > 0 and chat_history[-1]["role"] == "M141":
                chat_history[-1]["text"] += transcript
            else:
                chat_history.append({"role": "M141", "text": transcript})
            if len(chat_history) > 12: chat_history = chat_history[-12:]

    def on_audio_delta(audio):
        audio_handler.play_audio(audio)

    realtime_client = RealtimeClient(
        api_key=OPENAI_API_KEY,
        model="gpt-4o-realtime-preview",
        instructions="Ти — M141, маленький доброзичливий робот. АБСОЛЮТНО ВАЖЛИВО: Завжди розмовляй ТІЛЬКИ українською мовою. Користувач говорить з тобою українською. Ти ПОВИНЕН розпізнавати його мову як українську і відкидати будь-які російські транскрипції. Відповідай дуже коротко, мило і весело.",
        on_text_delta=on_text_delta,
        on_audio_delta=on_audio_delta,
        on_input_transcript=on_input_transcript,
        on_output_transcript=on_output_transcript,
        turn_detection_mode=TurnDetectionMode.SERVER_VAD,
    )

    try:
        await realtime_client.connect()
        message_handler = asyncio.create_task(realtime_client.handle_messages())
        streaming_task = asyncio.create_task(audio_handler.start_streaming(realtime_client))
        print(">>> [AUDIO] Connected to OpenAI Realtime API. Streaming active!")
        await asyncio.gather(message_handler, streaming_task)
    except Exception as e:
        print("Streaming Error:", e)

if __name__ == "__main__":
    # Start Vision & SLAM
    threading.Thread(target=vision_worker, daemon=True).start()
    threading.Thread(target=lidar_thread, daemon=True).start()
    
    # Start Web Server
    threading.Thread(target=run_flask, daemon=True).start()

    # Start Realtime OpenAI Streaming
    print(">>> Starting system with Realtime Voice Streaming (Option 2)")
    try:
        asyncio.run(start_realtime_stream())
    except KeyboardInterrupt:
        pass

