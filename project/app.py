import os
import time
import threading
from collections import Counter

import cv2
import numpy as np
import requests
from flask import Flask, render_template, Response, request, jsonify, send_from_directory
from ultralytics import YOLO
import pytesseract
from gtts import gTTS

app = Flask(__name__)

# -------------------- Global State --------------------
STATE = {
    "source_base": "",          # e.g. http://192.168.1.45
    "use_stream": True,         # True -> /stream, False -> /capture
    "ocr_enabled": True,
    "tts_enabled": True,
    "tts_mode": "both",         # "objects", "ocr", "both"
    "inference_stride": 3,      # run YOLO/OCR every Nth frame to save CPU
}

# last TTS update time (epoch seconds)
tts_last_update = 0
tts_lock = threading.Lock()

# YOLOv8n model (first run will auto-download weights)
model = YOLO("yolov8n.pt")  # nano model for speed

# Optional: If Tesseract is not in PATH, set the absolute path below
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# -------------------- Video Source Helpers --------------------
def build_urls():
    base = STATE["source_base"].rstrip("/")
    if not base.startswith("http"):
        return None, None
    stream_url = f"{base}/stream"
    capture_url = f"{base}/capture"
    return stream_url, capture_url

def mjpeg_frames_via_requests(stream_url):
    """
    Read multipart/x-mixed-replace MJPEG from ESP32 `/stream` endpoint with requests.
    Yields OpenCV BGR frames.
    """
    with requests.get(stream_url, stream=True, timeout=5) as r:
        r.raise_for_status()
        bytes_buf = b""
        for chunk in r.iter_content(chunk_size=1024):
            if chunk is None:
                continue
            bytes_buf += chunk
            a = bytes_buf.find(b'\xff\xd8')  # JPEG start
            b = bytes_buf.find(b'\xff\xd9')  # JPEG end
            if a != -1 and b != -1 and b > a:
                jpg = bytes_buf[a:b+2]
                bytes_buf = bytes_buf[b+2:]
                frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    yield frame

def snapshot_frames_via_requests(capture_url, fps=8):
    """
    Poll `/capture` endpoint periodically and yield frames.
    """
    delay = 1.0 / max(1, fps)
    while True:
        try:
            resp = requests.get(capture_url, timeout=3)
            img = cv2.imdecode(np.frombuffer(resp.content, np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                yield img
        except Exception:
            pass
        time.sleep(delay)

# -------------------- TTS --------------------
def speak_text(text):
    """
    Generate TTS file (static/tts.mp3) from text. Overwrites file and bumps timestamp.
    """
    global tts_last_update
    if not text.strip():
        return
    try:
        tts = gTTS(text=text, lang="en")
        out_path = os.path.join("static", "tts.mp3")
        tts.save(out_path)
        with tts_lock:
            tts_last_update = int(time.time())
    except Exception as e:
        print("TTS error:", e)

# -------------------- Streaming & Inference --------------------
def annotate_and_optionally_tts(frame, frame_index):
    """
    Runs YOLO every Nth frame; optionally runs OCR; returns annotated frame.
    Also triggers TTS if enabled and new content is found.
    """
    run_this_frame = (frame_index % max(1, STATE["inference_stride"]) == 0)

    detected_labels = []
    ocr_text = ""

    annotated = frame

    if run_this_frame:
        # YOLOv8 inference
        try:
            res = model(frame, verbose=False)
            annotated = res[0].plot()
            # Collect detected class names with confidence thresholding
            det = res[0].boxes
            if det is not None and det.cls is not None:
                names = res[0].names
                confs = det.conf.cpu().numpy() if det.conf is not None else []
                clss = det.cls.cpu().numpy().astype(int)
                for c, p in zip(clss, confs):
                    if p >= 0.35:  # tune threshold
                        detected_labels.append(names.get(c, str(c)))
        except Exception as e:
            print("YOLO error:", e)

        # OCR
        try:
            if STATE["ocr_enabled"]:
                ocr_text = pytesseract.image_to_string(frame)
                # quick normalize
                ocr_text = " ".join(ocr_text.split())
        except Exception as e:
            print("OCR error:", e)

        # TTS trigger
        if STATE["tts_enabled"]:
            to_say = []
            if STATE["tts_mode"] in ("objects", "both"):
                if detected_labels:
                    top = [f"{k} x{v}" for k, v in Counter(detected_labels).most_common()]
                    to_say.append("Detected: " + ", ".join(top))
            if STATE["tts_mode"] in ("ocr", "both"):
                if ocr_text and len(ocr_text) >= 3:
                    to_say.append("Text: " + ocr_text[:250])  # cap length

            msg = ". ".join(to_say)
            if msg:
                speak_text(msg)

    return annotated

def generate_mjpeg():
    """
    Flask generator that yields annotated JPEG frames as multipart MJPEG.
    """
    stream_url, capture_url = build_urls()
    if not stream_url:
        # blank frame with instruction
        blank = np.zeros((360, 640, 3), dtype=np.uint8)
        cv2.putText(blank, "Set ESP32 IP on the page", (30, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        _, buf = cv2.imencode(".jpg", blank)
        while True:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
            time.sleep(0.5)

    frame_iter = None
    if STATE["use_stream"]:
        try:
            frame_iter = mjpeg_frames_via_requests(stream_url)
        except Exception as e:
            print("Stream open error:", e)
    if frame_iter is None:
        frame_iter = snapshot_frames_via_requests(capture_url or "", fps=8)

    idx = 0
    for frame in frame_iter:
        idx += 1
        annotated = annotate_and_optionally_tts(frame, idx)
        ok, buffer = cv2.imencode(".jpg", annotated)
        if not ok:
            continue
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

# -------------------- Routes --------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(generate_mjpeg(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/set_source", methods=["POST"])
def set_source():
    """
    Body JSON:
    {
      "base": "http://192.168.1.45",
      "use_stream": true
    }
    """
    data = request.get_json(silent=True) or {}
    base = str(data.get("base", "")).strip()
    use_stream = bool(data.get("use_stream", True))
    STATE["source_base"] = base
    STATE["use_stream"] = use_stream
    return jsonify({"ok": True})

@app.route("/set_options", methods=["POST"])
def set_options():
    """
    Body JSON:
    {
      "ocr_enabled": true,
      "tts_enabled": true,
      "tts_mode": "both",      # "objects" | "ocr" | "both"
      "inference_stride": 3
    }
    """
    data = request.get_json(silent=True) or {}
    if "ocr_enabled" in data: STATE["ocr_enabled"] = bool(data["ocr_enabled"])
    if "tts_enabled" in data: STATE["tts_enabled"] = bool(data["tts_enabled"])
    if "tts_mode" in data:    STATE["tts_mode"] = data["tts_mode"] if data["tts_mode"] in ("objects","ocr","both") else "both"
    if "inference_stride" in data:
        try:
            s = int(data["inference_stride"])
            STATE["inference_stride"] = max(1, min(10, s))
        except Exception:
            pass
    return jsonify({"ok": True, "state": STATE})

@app.route("/tts_timestamp")
def tts_timestamp():
    """Return last TTS modification time so the client can refresh audio when it changes."""
    with tts_lock:
        ts = tts_last_update
    return jsonify({"ts": ts})

@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory("static", filename, as_attachment=False)

if __name__ == "__main__":
    os.makedirs("static", exist_ok=True)
    # create empty TTS file if not exists
    tts_path = os.path.join("static", "tts.mp3")
    if not os.path.exists(tts_path):
        # generate a tiny silent file to avoid 404 on first load
        try:
            tts = gTTS(text="ready", lang="en")
            tts.save(tts_path)
        except Exception:
            # fallback: write empty file
            open(tts_path, "wb").close()
    app.run(host="0.0.0.0", port=5000, debug=True)
