from flask import Flask, render_template, Response, url_for

from tof_camera import get_frame

PORT = 5000

app = Flask(__name__)
cam = None


@app.route("/vid")
def vid():
    return Response(
        get_frame(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=False)
