from flask import Flask, render_template, Response

from src.tof_camera import stream_frames

PORT = 5000

app = Flask(__name__)


@app.route("/vid0")
def vid0():
    return Response(
        stream_frames("amplitude"), mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/vid1")
def vid1():
    return Response(
        stream_frames("depth"), mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)
