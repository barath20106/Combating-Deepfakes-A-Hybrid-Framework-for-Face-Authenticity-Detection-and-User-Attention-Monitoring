from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import threading
import numpy as np
import imutils
import os
import time
from utils.attention import AttentionTracker
from utils.authenticity import check_face_authenticity

app = Flask(__name__)

tracker = AttentionTracker()
calibrated = False
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

camera_index = 0
camera = None
camera_on = True  # Track camera state


# ---------- Threaded Camera ----------
class VideoCamera:
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.frame = None
        self.lock = threading.Lock()
        threading.Thread(target=self.update, daemon=True).start()

    def update(self):
        while True:
            if camera_on:
                ret, frame = self.capture.read()
                if ret:
                    frame = imutils.resize(frame, width=480)
                    with self.lock:
                        self.frame = frame
            else:
                time.sleep(0.05)  # reduce CPU load if camera off
            time.sleep(0.005)

    def get_frame(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def set_src(self, src):
        self.capture.release()
        self.capture = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


def get_camera():
    global camera, camera_index
    if camera is None:
        camera = VideoCamera(camera_index)
    return camera


# ---------- Video Frame Generator ----------
def gen_frames():
    global calibrated, camera_on
    cam = get_camera()
    last_pred = "REAL"
    last_pred_time = 0
    fps_time = time.time()
    frame_count = 0
    fps = 0

    while True:
        frame = cam.get_frame()
        if frame is None or not camera_on:
            blank = np.zeros((360, 480, 3), dtype=np.uint8)
            cv2.putText(blank, "Camera is OFF", (120, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            ret, buffer = cv2.imencode('.jpg', blank)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.05)
            continue

        # --- Enhance clarity ---
        frame = cv2.convertScaleAbs(frame, alpha=1.4, beta=30)
        frame = cv2.detailEnhance(frame, sigma_s=8, sigma_r=0.2)

        # Deepfake prediction every 1 sec
        if time.time() - last_pred_time > 1.0:
            last_pred = check_face_authenticity(frame)
            last_pred_time = time.time()

        if not calibrated:
            cv2.putText(frame, "Calibration in Progress...", (40, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            tracker.calibrate(frame, samples=25)
            calibrated = True

        attention, _ = tracker.get_attention(frame)

        # Overlay status
        color = (0, 255, 0) if last_pred == "REAL" else (0, 0, 255)
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 60), (280, 140), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

        cv2.putText(frame, f"Attention: {attention}", (15, 90),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Authenticity: {last_pred}", (15, 125),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)

        cv2.rectangle(frame, (0, 0), (190, 40), color, -1)
        cv2.putText(frame, f"{last_pred} FACE", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # FPS counter
        frame_count += 1
        if frame_count >= 10:
            fps = 10 / (time.time() - fps_time)
            fps_time = time.time()
            frame_count = 0
        cv2.putText(frame, f"FPS: {int(fps)}", (400, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


# ---------- Web Routes ----------
@app.route("/", methods=["GET", "POST"])
def index():
    global camera_index, camera, calibrated, camera_on
    authenticity = None
    uploaded_image = None
    selected_camera = camera_index
    error = None

    if request.method == "POST":
        if "calibrate" in request.form:
            calibrated = False

        elif "camera_index" in request.form:
            new_index = int(request.form["camera_index"])
            if new_index != camera_index:
                camera_index = new_index
                if camera is not None:
                    camera.set_src(new_index)
                calibrated = False

        elif "toggle_camera" in request.form:
            camera_on = not camera_on

        elif "file" in request.files:
            file = request.files["file"]
            if file.filename == "":
                error = "Please select an image to upload."
            else:
                npimg = np.frombuffer(file.read(), np.uint8)
                img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
                upload_path = os.path.join(UPLOAD_FOLDER, "uploaded_image.jpg")
                cv2.imwrite(upload_path, img)
                uploaded_image = "uploads/uploaded_image.jpg"
                prob = check_face_authenticity(img, return_prob=True)
                authenticity = "REAL" if prob >= 0.5 else "FAKE"

    return render_template("index.html",
                           authenticity=authenticity,
                           uploaded_image=uploaded_image,
                           error=error,
                           selected_camera=selected_camera,
                           camera_on=camera_on)


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False, threaded=True)
