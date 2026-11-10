from contextlib import ExitStack
import os
from pathlib import Path
import sys
import tempfile
import threading
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
        views = [view(depth_value=0), view("222"), view("333")]
        canvas = cameras.live_dashboard(views)
        layout = cameras.dashboard_layout(len(views))
        self.assertEqual(canvas.shape, (layout["banner_h"] + 2 * layout["tile_h"], 1328, 3))
        self.assertEqual(canvas.dtype, np.uint8)
        self.assertGreater(layout["banner_h"], 54)
        self.assertIn("Q or Esc", " ".join(layout["lines"]))
        self.assertGreater(int(canvas[:layout["banner_h"]].max()), 100)
        y = layout["banner_h"] + 72 + 120
        # OpenCV output must be BGR, even though camera input is RGB.
        np.testing.assert_array_equal(canvas[y, 8+160], [0, 0, 255])
        np.testing.assert_array_equal(canvas[y, 16+320+160], [0, 0, 0])
        self.assertTrue(np.any(canvas[y, 664+16+320+160]))

    def test_waiting_and_failed_cameras_still_have_tiles(self):
        waiting = dict(serial="111", name="camera", frame=None, now=time.monotonic())
        failed = view("222")
        failed["error"] = "camera disconnected"
        canvas = cameras.live_dashboard([waiting, failed])
        layout = cameras.dashboard_layout(2)
        self.assertEqual(canvas.shape, (layout["banner_h"] + layout["tile_h"], 1328, 3))


class ViewerTests(unittest.TestCase):
    def args(self):
        return SimpleNamespace(width=640, height=480, fps=30, reset=True, depth_max=2.)

    def gui(self, stack, keys):
        stack.enter_context(patch.dict(os.environ, {"DISPLAY": ":test"}))
        for name in ["namedWindow", "setWindowProperty", "imshow", "destroyWindow"]:
            stack.enter_context(patch.object(cv2, name))
        stack.enter_context(patch.object(cv2, "getWindowProperty", return_value=1))
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

    def test_fullscreen_by_default_and_toggle(self):
        with ExitStack() as stack:
            self.gui(stack, [ord("f"), ord("F"), ord("q")])
            stack.enter_context(patch.object(cameras, "start_cameras", return_value=[]))
            cameras.live_view([("111", "D435")], self.args())
            self.assertEqual([call.args[2] for call in cv2.setWindowProperty.call_args_list],
                             [cv2.WINDOW_FULLSCREEN, cv2.WINDOW_NORMAL, cv2.WINDOW_FULLSCREEN])

    def test_can_quit_while_usb_startup_blocks_and_late_pipelines_close(self):
        release = threading.Event()
        entered = threading.Event()
        cam = SimpleNamespace(close=Mock())
        worker = cameras.LiveCameraStartup([("111", "D435")], self.args())

        def blocked_start(*args, **kwargs):
            entered.set()
            release.wait(timeout=5)
            return [cam]

        def quit_during_startup(_delay):
            self.assertTrue(entered.wait(timeout=1))
            self.assertFalse(worker.ready.is_set())
            return ord("q")

        try:
            with ExitStack() as stack:
                self.gui(stack, quit_during_startup)
                stack.enter_context(patch.object(cameras, "LiveCameraStartup", return_value=worker))
                stack.enter_context(patch.object(cameras, "start_cameras", side_effect=blocked_start))
                cameras.live_view([("111", "D435")], self.args())
                cv2.destroyWindow.assert_called_once()
                self.assertFalse(worker.ready.is_set())
        finally:
            release.set()
            worker.join(timeout=1)
        self.assertFalse(worker.is_alive())
        cam.close.assert_called_once()

    def test_startup_failure_is_reported_and_window_closes(self):
        worker = cameras.LiveCameraStartup([("111", "D435")], self.args())

        def wait_for_failure(_delay):
            self.assertTrue(worker.ready.wait(timeout=1))
            return -1

        with ExitStack() as stack:
            self.gui(stack, wait_for_failure)
            stack.enter_context(patch.object(cameras, "LiveCameraStartup", return_value=worker))
            stack.enter_context(patch.object(cameras, "start_cameras", side_effect=RuntimeError("USB failed")))
            with self.assertRaisesRegex(SystemExit, "Live camera startup failed: USB failed"):
                cameras.live_view([("111", "D435")], self.args())
            cv2.destroyWindow.assert_called_once()

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
