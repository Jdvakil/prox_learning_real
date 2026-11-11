"""Live browser dashboard while collecting: joints, gripper, cameras and the 40 proximity sensors.

The recorders start it by default and print its address (open it in a browser on the robot PC,
or from another machine on the lab network). --no-dashboard turns it off, --dashboard-port
moves it. It can also run on its own, without recording, to check sensors and cameras:

    python data.py live --wrist-serial <serial> --exo-serial <serial>     # -> http://<host>:8050
    python data.py live --robot                                          # also read the arm (read-only)

Everything is read on request from objects the recorder already has (robot state, gripper
reader, sensor listener, camera threads), so nothing runs while no browser is open. Cameras are
MJPEG streams, everything else is one small JSON poll at 10 Hz. Anything missing (no robot, no
cameras, silent sensors) is shown as missing, never an error.
"""

import argparse
import base64
import json
import os
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from .kinesthetic import Q_MAX, Q_MIN
from .sensor_layout import NO_TARGET_MM, SENSORS_PER_LINK

SENSORS = [(name, sensor_id) for name, count in SENSORS_PER_LINK.items() for sensor_id in range(count)]
PAGE = Path(__file__).resolve().parent / "dashboard_assets" / "live.html"
STREAM_FPS = 10


def encode_jpeg(rgb, quality=70):
    try:
        import cv2

        ok, data = cv2.imencode(".jpg", np.ascontiguousarray(rgb[:, :, ::-1]), [cv2.IMWRITE_JPEG_QUALITY, quality])
        return data.tobytes() if ok else None
    except ImportError:
        import io

        from PIL import Image

        out = io.BytesIO()
        Image.fromarray(rgb).save(out, format="JPEG", quality=quality)
        return out.getvalue()


class LiveDashboard:
    """Serves the page; collectors call publish(record) at every logged tick (cheap: one reference)."""

    def __init__(self, port=8050, robot=None, gripper=None, listener=None, cameras=(), phases=None, title=""):
        self.robot, self.gripper, self.listener = robot, gripper, listener
        self.cameras = {cam.name: cam for cam in cameras}
        self.phases, self.title = phases, title
        self.record, self.record_time = None, 0.0
        self.server = ThreadingHTTPServer(("0.0.0.0", port), self._handler())
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def start(self):
        self.thread.start()
        host = socket.gethostname()
        print(f"live dashboard: http://localhost:{self.server.server_port}  (or http://{host}:{self.server.server_port})", flush=True)
        return self

    def stop(self):
        self.server.shutdown()
        self.server.server_close()

    def publish(self, record):
        self.record, self.record_time = record, time.monotonic()

    # ---- state ---------------------------------------------------------------------------
    def state(self):
        now = time.monotonic()
        out = dict(title=self.title, time=time.time(), q_min=list(Q_MIN), q_max=list(Q_MAX),
                   recording=False, phase=None, t=None, q=None, dq=None, target=None, width=None)
        record = self.record
        if record is not None and now - self.record_time < 1.0:     # a collector is logging
            out.update(recording=True, t=float(record["t"]), q=record["q"].tolist(), dq=record["dq"].tolist(),
                       target=np.asarray(record["target_q"], dtype=float).tolist())
            phase = int(record["phase"])
            out["phase"] = self.phases[phase] if self.phases and 0 <= phase < len(self.phases) else str(phase)
        elif self.robot is not None:                                  # idle: read the arm directly
            try:
                from .record_dataset import read_robot_state

                state = read_robot_state(self.robot)
                out.update(q=state["q"].tolist(), dq=state["dq"].tolist())
            except Exception as exc:
                out["robot_error"] = str(exc)
        if self.gripper is not None:
            width = getattr(self.gripper, "width", float("nan"))
            out["width"] = None if not np.isfinite(width) else float(width)
        out["cameras"] = list(self.cameras)
        out["sensors"] = self.sensors(now)
        return out

    def sensors(self, now):
        names = [f"{name} {sensor_id}" for name, sensor_id in SENSORS]
        depth = np.full((len(SENSORS), 128), NO_TARGET_MM, dtype=np.uint16)
        age = [None] * len(SENSORS)
        if self.listener is not None:
            packets = self.listener.snapshot()
            for i, key in enumerate(SENSORS):
                if key in packets:
                    depth[i], arrived, _ = packets[key]
                    age[i] = round((now - arrived) * 1000.0)
        return dict(names=names, links=[name for name, _ in SENSORS], age_ms=age, no_target=NO_TARGET_MM,
                    depth=base64.b64encode(depth.astype("<u2").tobytes()).decode("ascii"))

    def frame(self, name):
        cam = self.cameras.get(name)
        if cam is None:
            return None, None
        try:
            frame, count = cam.latest()
        except Exception:
            return None, None
        return (None, None) if frame is None else (frame[0], count)

    # ---- http ----------------------------------------------------------------------------
    def _handler(self):
        dashboard = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def send(self, body, kind, code=200):
                self.send_response(code)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                try:
                    if self.path in ("/", "/index.html"):
                        self.send(PAGE.read_bytes(), "text/html; charset=utf-8")
                    elif self.path == "/state.json":
                        self.send(json.dumps(dashboard.state()).encode(), "application/json")
                    elif self.path.startswith("/stream/"):
                        self.stream(self.path[len("/stream/"):])
                    else:
                        self.send(b"not found", "text/plain", 404)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def stream(self, name):
                if name not in dashboard.cameras:
                    return self.send(b"no such camera", "text/plain", 404)
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                last = None
                while True:
                    rgb, count = dashboard.frame(name)
                    if rgb is not None and count != last:
                        data = encode_jpeg(rgb)
                        if data:
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                             + str(len(data)).encode() + b"\r\n\r\n" + data + b"\r\n")
                            last = count
                    time.sleep(1.0 / STREAM_FPS)

        return Handler


def add_args(parser):
    parser.add_argument("--no-dashboard", action="store_true", help="do not serve the live browser dashboard")
    parser.add_argument("--dashboard-port", type=int, default=8050)
    parser.add_argument("--no-browser", action="store_true", help="serve the dashboard but do not open a browser")


def open_browser(url):
    """Open the page in the desktop's browser, in the background; skipped without a display."""
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        print("no display: open the dashboard address above in a browser yourself")
        return
    import webbrowser

    threading.Thread(target=webbrowser.open_new_tab, args=(url,), daemon=True).start()


def start(args, robot=None, gripper=None, listener=None, cameras=(), phases=None, title=""):
    """Start the dashboard unless --no-dashboard; a busy port is a warning, not a failure."""
    if getattr(args, "no_dashboard", False):
        return None
    try:
        dashboard = LiveDashboard(args.dashboard_port, robot, gripper, listener, cameras, phases, title).start()
    except OSError as exc:
        print(f"warning: live dashboard not started on port {args.dashboard_port} ({exc})")
        return None
    from . import record_dataset

    record_dataset.LIVE = dashboard          # base.sample() publishes every logged tick to it
    if not getattr(args, "no_browser", False):
        open_browser(f"http://localhost:{dashboard.server.server_port}")
    return dashboard


def stop(dashboard):
    from . import record_dataset

    if dashboard is not None:
        if record_dataset.LIVE is dashboard:
            record_dataset.LIVE = None
        dashboard.stop()


def main(argv=None):
    from . import record_dataset as base

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="172.16.0.2")
    p.add_argument("--robot", action="store_true", help="also show the arm's joints (read-only connection)")
    p.add_argument("--gripper", action="store_true", help="also show the gripper width")
    p.add_argument("--wrist-serial")
    p.add_argument("--exo-serial")
    p.add_argument("--camera", nargs=2, action="append", default=[], metavar=("NAME", "SERIAL"))
    p.add_argument("--no-camera-reset", action="store_true")
    p.add_argument("--cam-width", type=int, default=640)
    p.add_argument("--cam-height", type=int, default=480)
    p.add_argument("--cam-fps", type=int, default=30)
    add_args(p)
    args = p.parse_args(argv)
    args.no_dashboard = False
    sockets = base.open_sensor_sockets()
    listener = base.SensorListener(sockets)
    listener.start()
    cameras = base.open_cameras(args) if (args.wrist_serial or args.exo_serial or args.camera) else []
    robot = gripper = None
    if args.robot:
        from franky import Robot

        robot = Robot(args.host)
    if args.gripper:
        gripper = base.GripperReader(args.host)
        gripper.start()
    dashboard = start(args, robot, gripper, listener, cameras, title="live view (not recording)")
    if dashboard is None:
        return 1
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        stop(dashboard)
        listener.stop_event.set()
        for cam in cameras:
            cam.close()
