import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np

from record_dataset import CameraRecorder, camera_specs, open_cameras


def options(**overrides):
    values = dict(wrist_serial=None, exo_serial=None, camera=[], cam_width=640,
                  cam_height=480, cam_fps=30, no_camera_reset=True)
    values.update(overrides)
    return SimpleNamespace(**values)


class CameraConfigurationTests(unittest.TestCase):
    def test_multiple_cameras_without_wrist(self):
        specs = [("left", "111"), ("right", "222"), ("overhead", "333")]
        args = options(camera=specs)
        with patch("cameras.start_cameras", return_value=[]) as start:
            open_cameras(args)
        start.assert_called_once_with(specs, 640, 480, 30, reset=False)

    def test_legacy_roles_and_extra_camera(self):
        self.assertEqual(camera_specs(options(wrist_serial="111", exo_serial="222",
                                             camera=[("exo_camera_2", "333")])),
                         [("wrist_camera", "111"), ("exo_camera_1", "222"), ("exo_camera_2", "333")])
        self.assertEqual(camera_specs(options(exo_serial="222")), [("exo_camera_1", "222")])

    def test_invalid_configuration(self):
        for args in (options(), options(camera=[("../bad", "111")]),
                     options(camera=[("bad/name", "111")]),
                     options(wrist_serial="111", camera=[("wrist_camera", "222")]),
                     options(wrist_serial="111", camera=[("other", "111")])):
            with self.subTest(args=args), self.assertRaises(ValueError):
                camera_specs(args)


class CameraRecordingTests(unittest.TestCase):
    def test_three_cameras_store_independent_rgbd_and_indices(self):
        # Exercise actual recorder threads and HDF5 writes; replace only video encoding.
        with tempfile.TemporaryDirectory() as temp, patch("record_dataset.Mp4Writer") as writer:
            row = Path(temp)
            with h5py.File(row / "trajectory.h5", "w") as handle:
                recorders = []
                for i, name in enumerate(("wrist_camera", "exo_camera_1", "exo_camera_2")):
                    cam = SimpleNamespace(name=name, serial=str(i), fps=30,
                                          intrinsics={"width": 4, "height": 2}, sink=None)
                    group = handle.create_group(f"traj_0/real/cameras/{name}")
                    rec = CameraRecorder(cam, group, row, store_depth=True, store_rgb=True)
                    rec.begin()
                    recorders.append(rec)
                for n in range(2):
                    for i, rec in enumerate(recorders):
                        rgb = np.full((2, 4, 3), i + n, dtype=np.uint8)
                        depth = np.full((2, 4), 1000 * (i + 1) + n, dtype=np.uint16)
                        rec.cam.sink((rgb, depth, 100 + n, 2000 + n, 10 + n))
                        rec.tick(rgb)
                for i, rec in enumerate(recorders):
                    rec.finish()
                    self.assertIsNone(rec.error)
                    rec.write_index(10, [100, 101])
                    group = rec.group
                    np.testing.assert_array_equal(group["depth_mm"][:, 0, 0],
                                                  [1000 * (i + 1), 1000 * (i + 1) + 1])
                    np.testing.assert_array_equal(group["rgb"][:, 0, 0, 0], [i, i + 1])
                    np.testing.assert_array_equal(group["tick_frame"][:], [0, 1])
                    np.testing.assert_array_equal(group["t"][:], [0, 1])
                    self.assertEqual(group.attrs["serial"], str(i))
                    self.assertEqual(group.attrs["fps"], 30)
                    self.assertEqual(rec.ticks, 2)
                    self.assertIsNone(rec.cam.sink)
                    writer.assert_any_call(row / f"{rec.cam.name}_30fps.mp4", 30)


if __name__ == "__main__":
    unittest.main()
