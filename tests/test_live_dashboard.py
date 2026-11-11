import base64
import json
import time
from types import SimpleNamespace
import unittest
import urllib.request

import numpy as np

from prox_real import live_dashboard as live
from prox_real import record_dataset as base


class Cam:
    name = "wrist_camera"

    def __init__(self):
        self.count = 0

    def latest(self):
        self.count += 1
        rgb = np.zeros((48, 64, 3), np.uint8)
        rgb[:, :, 0] = 200
        return (rgb, None, self.count, 0.0, time.monotonic()), self.count


class LiveDashboardTests(unittest.TestCase):
    def setUp(self):
        depth = np.full(128, 4000, np.uint16)
        depth[:64] = 321
        self.packets = {live.SENSORS[0]: (depth, time.monotonic(), 5)}
        listener = SimpleNamespace(snapshot=lambda: dict(self.packets))
        args = SimpleNamespace(no_dashboard=False, no_browser=True, dashboard_port=0)       # any free port
        self.dash = live.start(args, None, SimpleNamespace(width=0.024), listener, [Cam()], ["start", "lift"], "test")
        self.url = f"http://127.0.0.1:{self.dash.server.server_port}"

    def tearDown(self):
        live.stop(self.dash)
        self.assertIsNone(base.LIVE)

    def get(self, path):
        with urllib.request.urlopen(self.url + path, timeout=5) as response:
            return response.read()

    def test_page_and_idle_state(self):
        self.assertIn(b"Robot Live View", self.get("/"))
        state = json.loads(self.get("/state.json"))
        self.assertFalse(state["recording"])
        self.assertIsNone(state["q"])
        self.assertAlmostEqual(state["width"], 0.024)
        self.assertEqual(state["cameras"], ["wrist_camera"])
        sensors = state["sensors"]
        depth = np.frombuffer(base64.b64decode(sensors["depth"]), "<u2").reshape(len(live.SENSORS), 128)
        self.assertEqual(depth[0, 0], 321)
        self.assertEqual(depth[1, 0], 4000)
        self.assertIsNotNone(sensors["age_ms"][0])
        self.assertIsNone(sensors["age_ms"][1])                 # silent sensor shows as missing

    def test_sample_publishes_the_logged_tick(self):
        robot_state = dict(q=np.arange(7.0), dq=np.zeros(7), O_T_EE=np.eye(4), translation=np.zeros(3),
                           quaternion_xyzw=np.array([0, 0, 0, 1.0]), elbow=np.zeros(2))
        original = base.read_robot_state
        base.read_robot_state = lambda robot: dict(robot_state)
        try:
            base.sample(None, SimpleNamespace(snapshot=lambda: {}), None,
                        SimpleNamespace(phase=1, positions={1: np.ones(7)}), [], time.monotonic())
        finally:
            base.read_robot_state = original
        state = json.loads(self.get("/state.json"))
        self.assertTrue(state["recording"])
        self.assertEqual(state["phase"], "lift")
        self.assertEqual(state["q"], list(range(7)))
        self.assertEqual(state["target"], [1.0] * 7)

    def test_camera_stream_sends_jpeg_frames(self):
        with urllib.request.urlopen(self.url + "/stream/wrist_camera", timeout=5) as response:
            self.assertIn("multipart/x-mixed-replace", response.headers["Content-Type"])
            chunk = response.read(4096)
        self.assertIn(b"--frame", chunk)
        self.assertIn(b"\xff\xd8", chunk)                        # JPEG start marker

    def test_busy_port_is_a_warning(self):
        args = SimpleNamespace(no_dashboard=False, no_browser=True, dashboard_port=self.dash.server.server_port)
        self.assertIsNone(live.start(args))
        self.assertIsNone(live.start(SimpleNamespace(no_dashboard=True, dashboard_port=0)))


if __name__ == "__main__":
    unittest.main()
