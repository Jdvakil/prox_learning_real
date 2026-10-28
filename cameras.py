"""Intel RealSense D435 streams for the wrist and exocentric cameras.

Each camera runs in its own thread: colour (rgb8) and depth (z16) at the same size and rate,
depth aligned to the colour image, and the newest pair kept with its frame number, device
timestamp and arrival time (the 15 Hz logger in kinesthetic.py samples that newest pair).

    python cameras.py                  # list connected cameras, save one colour + depth snapshot each
    python cameras.py --out camera_check

Use the snapshots to aim the exocentric camera and check the wrist mount, and pass the serial
numbers to kinesthetic.py replay (--wrist-serial, --exo-serial).

Needs pyrealsense2 (pip install pyrealsense2).
"""

import argparse
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

    for dev in rs.context().query_devices():
        if dev.get_info(rs.camera_info.serial_number) == serial:
            if dev.supports(rs.camera_info.usb_type_descriptor):
                return dev.get_info(rs.camera_info.usb_type_descriptor)
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
        self.stop_event = threading.Event()

    def run(self):
        # The first frameset can take several seconds after start; later misses are tolerated
        # until no frame has arrived for 5 s.
        first_timeout_ms, stall_s = 10000, 5.0
        last = time.monotonic()
        try:
            while not self.stop_event.is_set():
                ok, frames = self.pipeline.try_wait_for_frames(first_timeout_ms if self.count == 0 else 1000)
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
                with self.lock:
                    self.frame = (rgb, depth_mm, int(color.get_frame_number()), float(color.get_timestamp()), arrived)
                    self.count += 1
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
        self.join(timeout=3.0)
        try:
            self.pipeline.stop()
        except RuntimeError:
            pass


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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("camera_check"))
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--reset", action="store_true", help="USB power-cycle every camera first")
    args = parser.parse_args()
    devices = connected()
    if not devices:
        raise SystemExit("no RealSense camera found")
    if args.reset:
        hardware_reset([s for s, _ in devices])
    args.out.mkdir(parents=True, exist_ok=True)
    for serial, name in devices:
        usb = usb_type(serial)
        print(f"{serial}  {name}  USB {usb}")
        if usb.startswith("2"):
            print(f"  warning: {serial} on USB {usb}; colour+depth at {args.width}x{args.height}@{args.fps} "
                  "may not stream (use a USB 3 port/cable, or e.g. --fps 15 / --width 424 --height 240)")
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
