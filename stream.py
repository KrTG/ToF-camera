import subprocess

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
    return Response(streamer.stream_odometry(), mimetype='text/event-stream')

@app.post("/odometry_reset")
def reset_odometry():
    with open("/tmp/reset", "w"):
        pass
    return make_response()

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
        subprocess.run(['sudo', 'systemctl', 'restart', 'cam'], check=True)
        return redirect(url_for("index"))
    except Exception as e:
        return make_response(str(e), 500)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)
