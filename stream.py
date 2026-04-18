from flask import Flask, render_template, Response, url_for

from regular_camera import get_frame
from tof_camera import TofCamera

PORT = 5000

app = Flask(__name__)
cam = TofCamera()
cam.start()


@app.route("/vid")
def vid():
    return Response(
        cam.get_frame(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80, debug=True, threaded=True)
