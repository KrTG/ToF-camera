import subprocess
import os
import glob
from flask import Flask, make_response, redirect, render_template, Response, url_for

from src.web import streamer

PORT = 5000

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/odometry")
def odometry():
    return render_template("odometry.html")

@app.get("/odometry_stream")
def odometry_stream():
    return Response(streamer.stream_odometry(), mimetype="text/event-stream")

@app.post("/odometry_reset")
def reset_odometry():
    with open("/tmp/reset", "w"):
        pass
    return make_response()

@app.route("/recording")
def recording():
    return render_template("recording.html")

@app.get("/recording_stream")
def recording_stream():
    return Response(streamer.stream_recording(), mimetype="text/event-stream")

@app.post("/recording/start")
def start_recording():
    if streamer.recorder_thread is not None:
        streamer.recorder_thread.start_recording()
    return make_response()

@app.post("/recording/stop")
def stop_recording():
    if streamer.recorder_thread is not None:
        streamer.recorder_thread.stop_recording()
    return make_response()


@app.route("/playbacks")
def list_playbacks():
    recordings = []
    for base in ["out", "test"]:
        if os.path.exists(base):
            for root, dirs, files in os.walk(base):
                for f in files:
                    if f.endswith(".replay"):
                        full_path = os.path.join(root, f)
                        recordings.append(full_path)
    recordings.sort()
    return render_template("playbacks.html", recordings=recordings)

@app.route("/playback/<path:filename>")
def playback(filename):
    return render_template("playback.html", filename=filename)

@app.route("/playback_amplitude/<path:filename>")
def playback_amplitude(filename):
    return Response(
        streamer.stream_playback_frames(filename, "amplitude"), mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/playback_depth/<path:filename>")
def playback_depth(filename):
    return Response(
        streamer.stream_playback_frames(filename, "depth"), mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/video")
def video():
    return render_template("video.html")

@app.route("/amplitude_video")
def amplitude_video():
    return Response(
        streamer.stream_frames("amplitude"), mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/depth_video")
def depth_video():
    return Response(
        streamer.stream_frames("depth"), mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.post("/shutdown")
def shutdown_system():
    try:
        subprocess.run(['sudo', '-n', 'shutdown', '-h', 'now'], check=True)
        return make_response("Shutting down...", 200)
    except Exception as e:
        return make_response(str(e), 500)

@app.post("/reload")
def reload_service():
    try:
        subprocess.run(['setsid', 'sudo', 'systemctl', 'restart', 'cam'], check=True)
        return redirect(url_for("index"))
    except Exception as e:
        return make_response(str(e), 500)

@app.post("/mav/start")
def start_mav_service():
    try:
        subprocess.run(['sudo', 'systemctl', 'start', 'mav'], check=True)
        return make_response("MAVLink service started", 200)
    except Exception as e:
        return make_response("unavailable", 200)

@app.post("/mav/stop")
def stop_mav_service():
    try:
        subprocess.run(['sudo', 'systemctl', 'stop', 'mav'], check=True)
        return make_response("MAVLink service stopped", 200)
    except Exception as e:
        return make_response("unavailable", 200)

@app.get("/mav/status")
def get_mav_service_status():
    try:
        result = subprocess.run(["systemctl", "is-active", "mav"], capture_output=True, text=True, check=False)
        status_output = result.stdout.strip()
        return make_response(status_output, 200)
    except Exception as e:
        return make_response("unavailable", 200)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)
