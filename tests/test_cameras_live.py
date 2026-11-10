from contextlib import ExitStack
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

import cameras


def view(serial="111", depth_value=1000):
    now = time.monotonic()
    rgb = np.zeros((24, 32, 3), np.uint8)
    rgb[..., 0] = 255
    depth = np.full((24, 32), depth_value, np.uint16)
    return dict(serial=serial, name="Intel RealSense D435", now=now, fps=30.,
                frame=(rgb, depth, 42, 1000., now), error=None)


class DashboardTests(unittest.TestCase):
    def test_multiple_camera_layout_rgb_order_and_invalid_depth(self):
        canvas = cameras.live_dashboard([view(depth_value=0), view("222"), view("333")])
        self.assertEqual(canvas.shape, (694, 1328, 3))
        self.assertEqual(canvas.dtype, np.uint8)
        # OpenCV output must be BGR, even though camera input is RGB.
        np.testing.assert_array_equal(canvas[54+72+120, 8+160], [0, 0, 255])
        np.testing.assert_array_equal(canvas[54+72+120, 16+320+160], [0, 0, 0])
        self.assertTrue(np.any(canvas[54+72+120, 664+16+320+160]))

    def test_waiting_and_failed_cameras_still_have_tiles(self):
        waiting = dict(serial="111", name="camera", frame=None, now=time.monotonic())
        failed = view("222")
        failed["error"] = "camera disconnected"
        canvas = cameras.live_dashboard([waiting, failed])
        self.assertEqual(canvas.shape, (374, 1328, 3))


class ViewerTests(unittest.TestCase):
    def args(self):
        return SimpleNamespace(width=640, height=480, fps=30, reset=True, depth_max=2.)

    def gui(self, stack, keys):
        stack.enter_context(patch.dict(os.environ, {"DISPLAY": ":test"}))
        for name in ["namedWindow", "resizeWindow", "imshow", "destroyWindow"]:
            stack.enter_context(patch.object(cv2, name))
        stack.enter_context(patch.object(cv2, "waitKey", side_effect=keys))

    def test_all_cameras_open_together_and_close_on_quit(self):
        views = [view(), view("222")]
        cams = [SimpleNamespace(latest=Mock(return_value=(v["frame"], 1)), close=Mock()) for v in views]
        # A failed camera should remain visible while the other stream continues.
        cams[1].latest.side_effect = RuntimeError("camera disconnected")
        with ExitStack() as stack:
            self.gui(stack, [-1, ord("q")])
            start = stack.enter_context(patch.object(cameras, "start_cameras", return_value=cams))
            cameras.live_view([("111", "D435"), ("222", "D435")], self.args())
        start.assert_called_once_with([("111", "111"), ("222", "222")], 640, 480, 30, reset=True)
        for cam in cams:
            cam.close.assert_called_once()

    def test_viewer_failure_closes_all_pipelines(self):
        frame = view()["frame"]
        cams = [SimpleNamespace(latest=Mock(return_value=(frame, 1)), close=Mock()) for _ in range(2)]
        with ExitStack() as stack:
            self.gui(stack, [-1])
            stack.enter_context(patch.object(cameras, "start_cameras", return_value=cams))
            stack.enter_context(patch.object(cameras, "live_dashboard", side_effect=[np.zeros((100,100,3),np.uint8), RuntimeError("render failed")]))
            with self.assertRaisesRegex(RuntimeError, "render failed"):
                cameras.live_view([("111", "D435"), ("222", "D435")], self.args())
        for cam in cams:
            cam.close.assert_called_once()

    def test_live_cli_does_not_write_snapshots_or_reset_twice(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "snapshots"
            with patch.object(sys, "argv", ["cameras.py", "--live", "--reset", "--out", str(out)]),\
                 patch.object(cameras, "connected", return_value=[("111", "D435"), ("222", "D435")]),\
                 patch.object(cameras, "usb_type", return_value="3.2"),\
                 patch.object(cameras, "hardware_reset") as reset,\
                 patch.object(cameras, "live_view") as live:
                cameras.main()
            self.assertFalse(out.exists())
            reset.assert_not_called()
            self.assertTrue(live.call_args.args[1].reset)
            self.assertEqual(len(live.call_args.args[0]), 2)


if __name__ == "__main__":
    unittest.main()
