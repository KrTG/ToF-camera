from flask import Flask, render_template, Response

from tof_camera import stream_frames

PORT = 5000

app = Flask(__name__)
cam = None


@app.route("/vid")
def vid():
    return Response(
        stream_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
