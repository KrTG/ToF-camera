from flask import Flask, render_template, Response

from src.web import stream_frames

PORT = 5000

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/odometry")
def odometry():
    return render_template("odometry.html")

@app.route("/video")
def video():
    return render_template("video.html")

@app.route("/amplitude_video")
def amplitude_video():
    return Response(
        stream_frames("amplitude"), mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/depth_video")
def depth_video():
    return Response(
        stream_frames("depth"), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)
