"""Intel RealSense D435 streams for the wrist and exocentric cameras.

Each camera runs in its own thread: colour (rgb8) and depth (z16) at the same size and rate,
depth aligned to the colour image, and the newest pair kept with its frame number, device
timestamp and arrival time (the 15 Hz logger in kinesthetic.py samples that newest pair).

    python cameras.py                  # list connected cameras, save one colour + depth snapshot each
    python cameras.py --out camera_check
    python cameras.py --live           # all connected cameras: live RGB + aligned depth dashboard
    python cameras.py --live --reset   # power-cycle cameras before viewing

Use the snapshots to aim the exocentric camera and check the wrist mount, and pass the serial
numbers to kinesthetic.py replay (--wrist-serial, --exo-serial).

Needs pyrealsense2 (pip install pyrealsense2).
"""

import argparse
from collections import deque
import math
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np


def connected():
    """(serial, product name) of every RealSense on USB."""
    import pyrealsense2 as rs

    return [(dev.get_info(rs.camera_info.serial_number), dev.get_info(rs.camera_info.name))
            for dev in rs.context().query_devices()]


def usb_type(serial):
    """USB link the camera enumerated on ("3.2", "2.1", ...); "?" if unknown."""
    import pyrealsense2 as rs

    try:
        for dev in rs.context().query_devices():
            if dev.get_info(rs.camera_info.serial_number) == serial:
                if dev.supports(rs.camera_info.usb_type_descriptor):
                    return dev.get_info(rs.camera_info.usb_type_descriptor)
    except Exception:                # a hung camera can fail this query too
        pass
    return "?"


def hardware_reset(serials, settle=5.0):
    """Power-cycle the cameras over USB and wait for them to re-enumerate."""
    import pyrealsense2 as rs

    for dev in rs.context().query_devices():
        if dev.get_info(rs.camera_info.serial_number) in serials:
            dev.hardware_reset()
    deadline = time.monotonic() + 20.0
    time.sleep(settle)
    while time.monotonic() < deadline:
        if set(serials) <= {s for s, _ in connected()}:
            return
        time.sleep(0.5)
    raise RuntimeError(f"cameras did not come back after reset: {serials}")


class RealSenseCamera(threading.Thread):
    """One D435: colour + aligned depth (uint16 mm), newest frame pair kept."""

    def __init__(self, name, serial, width=640, height=480, fps=30):
        super().__init__(daemon=True)
        import pyrealsense2 as rs

        self.name, self.serial, self.fps = name, serial, fps
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
        try:
            profile = self.pipeline.start(config)
        except RuntimeError as exc:
            found = ", ".join(f"{s} ({n})" for s, n in connected()) or "none"
            raise RuntimeError(f"{name}: cannot start RealSense {serial} ({exc}); connected: {found}") from exc
        self.depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.intrinsics = {"width": intr.width, "height": intr.height, "fx": intr.fx, "fy": intr.fy,
                           "ppx": intr.ppx, "ppy": intr.ppy, "model": str(intr.model),
                           "coeffs": [float(c) for c in intr.coeffs]}
        self.align = rs.align(rs.stream.color)
        self.lock = threading.Lock()
        self.frame = None            # (rgb (H, W, 3) uint8, depth (H, W) uint16 mm, frame number, device ms, arrival)
        self.count = 0
        self.error = None
        self.sink = None             # callable(frame tuple), fed every frame (record_dataset.CameraRecorder)
        self.stop_event = threading.Event()

    def run(self):
        # The first frameset can take several seconds after start; later misses are tolerated
        # until no frame has arrived for 1 s, so a recording never logs a long frozen stretch.
        first_timeout_ms, poll_ms, stall_s = 10000, 200, 1.0
        last = time.monotonic()
        try:
            while not self.stop_event.is_set():
                ok, frames = self.pipeline.try_wait_for_frames(first_timeout_ms if self.count == 0 else poll_ms)
                if not ok:
                    if self.count == 0 or time.monotonic() - last > stall_s:
                        raise RuntimeError(f"no frames from {self.serial} (USB {usb_type(self.serial)}); "
                                           "try a USB 3 port/cable, lower --fps/--width/--height, or --reset")
                    continue
                last = time.monotonic()
                frames = self.align.process(frames)
                color, depth = frames.get_color_frame(), frames.get_depth_frame()
                if not color or not depth:
                    continue
                arrived = time.monotonic()
                rgb = np.array(np.asanyarray(color.get_data()), dtype=np.uint8, copy=True)
                raw = np.asanyarray(depth.get_data())
                if abs(self.depth_scale - 0.001) < 1e-9:
                    depth_mm = np.array(raw, dtype=np.uint16, copy=True)
                else:
                    depth_mm = np.clip(np.round(raw * (self.depth_scale * 1000.0)), 0, 65535).astype(np.uint16)
                frame = (rgb, depth_mm, int(color.get_frame_number()), float(color.get_timestamp()), arrived)
                with self.lock:
                    self.frame = frame
                    self.count += 1
                sink = self.sink
                if sink is not None:
                    sink(frame)
        except Exception as exc:     # surfaced by latest() / wait_ready()
            self.error = exc

    def latest(self):
        if self.error is not None:
            raise RuntimeError(f"{self.name} camera stopped: {self.error!r}")
        with self.lock:
            return self.frame, self.count

    def wait_ready(self, frames=15, timeout=15.0):
        """Block until `frames` frames arrived (auto exposure settles over the first few)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            frame, count = self.latest()
            if count >= frames:
                return
            time.sleep(0.05)
        raise TimeoutError(f"{self.name} camera ({self.serial}): {self.count} frames in {timeout:.0f} s")

    def close(self):
        self.stop_event.set()
        if self.ident is not None:       # started
            self.join(timeout=3.0)
        try:
            self.pipeline.stop()
        except RuntimeError:
            pass


def start_cameras(specs, width=640, height=480, fps=30, reset=True, settle=3.0):
    """Open (name, serial) cameras, wait until each one streams, then let them settle.

    On the robot PC a D435 often gives no frames on a fresh start until it is power-cycled
    (python cameras.py failed the same way without --reset), so the cameras are reset first by
    default. If one still gives no frames, all are reset and started once more. Auto white
    balance needs about 2 s after a start (the first frames come out orange), so the cameras keep
    streaming for `settle` seconds before this returns.
    """
    serials = [serial for _, serial in specs]
    for attempt in (0, 1):
        if reset or attempt:
            print(f"resetting cameras {', '.join(serials)}")
            hardware_reset(serials)
        cams = []
        try:
            for name, serial in specs:
                cams.append(RealSenseCamera(name, serial, width, height, fps))
            for cam in cams:
                cam.start()
            for cam in cams:
                cam.wait_ready()
            time.sleep(settle)
            for cam in cams:
                cam.latest()             # raises if a camera stopped while settling
            return cams
        except BaseException as exc:
            for cam in cams:
                cam.close()
            if attempt or not isinstance(exc, Exception):
                raise
            print(f"cameras did not stream ({exc}); resetting and retrying once")


def save_png(path, image):
    try:
        import imageio.v2 as imageio

        imageio.imwrite(path, image)
    except ImportError:
        import cv2

        cv2.imwrite(str(path), image[..., ::-1] if image.ndim == 3 else image)


def depth_preview(depth_mm, far_mm=2000):
    """uint16 mm -> uint8 grey, near = bright, invalid = black."""
    d = depth_mm.astype(np.float32)
    out = np.where(d > 0, 255.0 * (1.0 - np.clip(d / far_mm, 0, 1)), 0)
    return out.astype(np.uint8)


def live_dashboard(views, far_mm=2000):
    """Render one RGB/depth tile per camera as an OpenCV BGR dashboard."""
    import cv2

    if not views:
        raise ValueError("the live dashboard needs at least one camera")
    preview_w, preview_h, gap = 320, 240, 8
    tile_w, tile_h, banner_h = 2 * preview_w + 3 * gap, 320, 54
    columns = min(3, math.ceil(math.sqrt(len(views))))
    rows = math.ceil(len(views) / columns)
    canvas = np.full((banner_h + rows * tile_h, columns * tile_w, 3), 22, dtype=np.uint8)
    cv2.putText(canvas, f"RealSense live RGB-D | {len(views)} cameras | Q / Esc: quit",
                (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Depth: near = red, far = purple; scale 0-{far_mm / 1000:g} m; invalid = black",
                (12, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (170, 170, 170), 1, cv2.LINE_AA)

    def preview(image):
        height, width = image.shape[:2]
        scale = min(preview_w / width, preview_h / height)
        width, height = max(1, round(width * scale)), max(1, round(height * scale))
        result = np.zeros((preview_h, preview_w, 3), dtype=np.uint8)
        x, y = (preview_w - width) // 2, (preview_h - height) // 2
        result[y:y + height, x:x + width] = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        return result

    for index, view in enumerate(views):
        x = (index % columns) * tile_w
        y = banner_h + (index // columns) * tile_h
        tile = canvas[y:y + tile_h, x:x + tile_w]
        cv2.putText(tile, f"{index + 1}. {view['serial']} | {view['name']}"[:85],
                    (gap, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (230, 230, 230), 1, cv2.LINE_AA)
        frame = view.get("frame")
        error = view.get("error")
        if frame is None:
            status = error or "Waiting for frames..."
        else:
            rgb, depth, number, _, arrived = frame
            color = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            depth_color = cv2.applyColorMap(depth_preview(depth, far_mm), cv2.COLORMAP_TURBO)
            depth_color[depth == 0] = 0
            tile[72:72 + preview_h, gap:gap + preview_w] = preview(color)
            tile[72:72 + preview_h, 2 * gap + preview_w:2 * gap + 2 * preview_w] = preview(depth_color)
            age_ms = max(0, view["now"] - arrived) * 1000
            fps = view.get("fps")
            rate = f"{fps:.1f} fps" if fps is not None else "measuring fps"
            status = (f"{rate} | frame {number} | age {age_ms:.0f} ms | "
                      f"depth valid {100 * np.mean(depth > 0):.1f}%")
            if error:
                status = "OFFLINE: " + error
            elif age_ms > 1000:
                status = "STALE: " + status
        cv2.putText(tile, status[:100], (gap, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (70, 100, 255) if error or status.startswith("STALE") else (100, 220, 150), 1, cv2.LINE_AA)
        cv2.putText(tile, "RGB", (gap, 63), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (190, 190, 190), 1, cv2.LINE_AA)
        cv2.putText(tile, "Aligned depth", (2 * gap + preview_w, 63),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.43, (190, 190, 190), 1, cv2.LINE_AA)
    return canvas


def live_view(devices, args):
    """Open all devices concurrently and display their latest RGB-D frames."""
    import cv2

    if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        raise SystemExit("--live needs a desktop display; run it in the lab PC's graphical session")
    window = "RealSense live RGB-D"
    cameras = []
    opened = False
    try:
        try:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            opened = True
        except cv2.error as exc:
            raise SystemExit("--live needs OpenCV GUI support (opencv-python) and a desktop display") from exc
        views = [dict(serial=serial, name=name, frame=None, error=None, now=time.monotonic())
                 for serial, name in devices]
        initial = live_dashboard(views, args.depth_max * 1000)
        scale = min(1, 1600 / initial.shape[1], 900 / initial.shape[0])
        cv2.resizeWindow(window, round(initial.shape[1] * scale), round(initial.shape[0] * scale))
        cv2.imshow(window, initial)
        cv2.waitKey(1)
        cameras = start_cameras([(serial, serial) for serial, _ in devices],
                                args.width, args.height, args.fps, reset=args.reset)
        histories = [deque(maxlen=60) for _ in cameras]
        print("Live RGB-D dashboard: Q, Esc, window close, or Ctrl-C to quit.")
        while True:
            now = time.monotonic()
            for cam, view, history in zip(cameras, views, histories):
                view["now"] = now
                try:
                    frame, count = cam.latest()
                    view["frame"], view["error"] = frame, None
                    if frame is not None and (not history or history[-1][1] != count):
                        history.append((frame[4], count))
                    if len(history) > 1:
                        elapsed = history[-1][0] - history[0][0]
                        view["fps"] = (history[-1][1] - history[0][1]) / elapsed if elapsed > 0 else None
                except RuntimeError as exc:
                    view["error"] = str(exc)
            cv2.imshow(window, live_dashboard(views, args.depth_max * 1000))
            key = cv2.waitKey(20) & 0xff
            if key in (27, ord("q"), ord("Q")):
                break
            try:
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:  # Window was closed by the desktop window manager.
                break
    except KeyboardInterrupt:
        pass
    finally:
        for cam in cameras:
            cam.close()
        if opened:
            try:
                cv2.destroyWindow(window)
            except cv2.error:
                pass


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("camera_check"))
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--reset", action="store_true", help="USB power-cycle every camera first")
    parser.add_argument("--live", action="store_true", help="show all connected cameras in one live RGB-D dashboard")
    parser.add_argument("--depth-max", type=float, default=2.0, help="live depth color scale far distance in meters (default 2)")
    args = parser.parse_args()
    if not np.isfinite(args.depth_max) or args.depth_max <= 0:
        parser.error("--depth-max must be positive and finite")
    devices = connected()
    if not devices:
        raise SystemExit("no RealSense camera found")
    if args.reset and not args.live:
        hardware_reset([s for s, _ in devices])
    for serial, name in devices:
        usb = usb_type(serial)
        print(f"{serial}  {name}  USB {usb}")
        if usb.startswith("2"):
            print(f"  warning: {serial} on USB {usb}; colour+depth at {args.width}x{args.height}@{args.fps} "
                  "may not stream (use a USB 3 port/cable, or e.g. --fps 15 / --width 424 --height 240)")
    if args.live:
        live_view(devices, args)
        return
    args.out.mkdir(parents=True, exist_ok=True)
    for serial, name in devices:
        cam = RealSenseCamera(serial, serial, args.width, args.height, args.fps)
        cam.start()
        try:
            cam.wait_ready(frames=30)
            (rgb, depth, number, _, _), _ = cam.latest()
        finally:
            cam.close()
        save_png(args.out / f"{serial}_rgb.png", rgb)
        save_png(args.out / f"{serial}_depth.png", depth_preview(depth))
        valid = depth[depth > 0]
        print(f"{serial}  {name}  fx {cam.intrinsics['fx']:.1f} fy {cam.intrinsics['fy']:.1f}  "
              f"depth valid {100 * valid.size / depth.size:.0f}%"
              + (f", median {np.median(valid):.0f} mm" if valid.size else "")
              + f"  -> {args.out}/{serial}_rgb.png")


if __name__ == "__main__":
    main()
