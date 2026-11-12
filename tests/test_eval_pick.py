import json
from types import SimpleNamespace
import time
import unittest

import numpy as np

from prox_real import eval_pick as ev
from prox_real import pick_block as pick
from prox_real import record_dataset as base
from act_model.sensor_order import HYBRID_SKIN_SENSOR_ORDER


def eval_args(*extra):
    p = __import__("argparse").ArgumentParser()
    ev.add_args(p)
    return p.parse_args(list(extra))


class Future:
    def __init__(self, done=True, value=True):
        self.done, self.value = done, value

    def wait(self, timeout):
        return self.done

    def get(self):
        return self.value


class Gripper:
    def __init__(self):
        self.calls = []
        self.future = Future()

    def grasp_async(self, width, speed, force, epsilon_inner, epsilon_outer):
        self.calls.append(("grasp", force, epsilon_outer))
        return self.future

    def open_async(self, speed):
        self.calls.append(("open",))
        return self.future


class ObservationTests(unittest.TestCase):
    def test_qpos_is_joints_plus_half_width_per_finger(self):
        q = np.arange(7, dtype=float)
        np.testing.assert_allclose(ev.qpos_vector(q, 0.024), [0, 1, 2, 3, 4, 5, 6, 0.012, 0.012])
        np.testing.assert_allclose(ev.qpos_vector(q, float("nan"))[7:], [0, 0])

    def test_skin_frame_uses_block_one_in_metres_and_policy_order(self):
        index = ev.sensor_index(list(HYBRID_SKIN_SENSOR_ORDER))
        depth = np.full((len(base.SENSORS), 128), 4000, np.uint16)
        target = HYBRID_SKIN_SENSOR_ORDER[5]
        depth[index[5], :64] = 120             # block 1 of that sensor
        depth[index[5], 64:] = 30              # block 2 must be ignored
        depth[index[6], 0] = 0                 # invalid zero reads as no return
        frame = ev.skin_frame(depth, index)
        self.assertEqual(frame.shape, (40, 8, 8))
        np.testing.assert_allclose(frame[5], 0.12)
        self.assertEqual(frame[6, 0, 0], ev.NO_RETURN_M)
        others = np.arange(40) != 5
        self.assertTrue(np.all(frame[others] == ev.NO_RETURN_M), target)

    def test_skin_history_matches_the_training_loader_window(self):
        history = ev.SkinHistory()
        frames = [np.full((40, 8, 8), k, np.float32) for k in range(10)]
        for t, frame in enumerate(frames):
            out = history.append(frame)
            expected = np.stack([frames[max(0, k)] for k in range(t - 7, t + 1)])   # utils.py raw_causal
            np.testing.assert_array_equal(out, expected)

    def test_silent_sensors_read_no_return(self):
        listener = SimpleNamespace(snapshot=lambda: {base.SENSORS[0]: (np.full(128, 77, np.uint16), 0.0, 1)})
        depth = ev.live_packets(listener)
        self.assertEqual(depth[0, 0], 77)
        self.assertTrue(np.all(depth[1:] == 4000))


class SafetyTests(unittest.TestCase):
    def setUp(self):
        self.saved = dict(grasp_q=[0.449, 0.660, -0.076, -2.105, -0.196, 2.840, 1.352])
        f_t_ee = np.array([[0.7071, 0.7071, 0, 0], [-0.7071, 0.7071, 0, 0], [0, 0, 1, 0.1034], [0, 0, 0, 1]])
        self.saved["grasp_pose"] = (pick.flange(np.array(self.saved["grasp_q"])) @ f_t_ee).tolist()
        self.env = ev.Envelope(self.saved, eval_args())

    def test_home_and_grasp_are_inside(self):
        self.env.check(np.asarray(pick.HOME, dtype=float))
        self.env.check(np.asarray(self.saved["grasp_q"]))

    def test_below_the_floor_is_refused(self):
        low = pick.shifted(np.asarray(self.saved["grasp_pose"]), dz=-0.03)
        q = pick.ik(low, self.saved["grasp_q"], self.env.f_t_ee)
        with self.assertRaisesRegex(RuntimeError, "below the floor"):
            self.env.check(q)

    def test_joint_limit_is_refused(self):
        q = np.asarray(pick.HOME, dtype=float)
        q[3] = -3.05
        with self.assertRaisesRegex(RuntimeError, "joint 4"):
            self.env.check(q)


class GripperTests(unittest.TestCase):
    def test_close_once_open_once_with_the_collection_grasp(self):
        gripper = Gripper()
        command = ev.GripperCommand(gripper, eval_args())
        for value in (0.1, 0.2, 0.9, 0.95, 0.8, 0.3, 0.2):
            command.update(value, 0.0)
        self.assertEqual(gripper.calls, [("grasp", 20.0, 1.0), ("open",)])
        self.assertEqual([e["command"] for e in command.events], ["close", "open"])

    def test_no_new_command_while_one_is_running(self):
        gripper = Gripper()
        gripper.future = Future(done=False)
        command = ev.GripperCommand(gripper, eval_args())
        command.update(0.9, 0.0)
        command.update(0.1, 0.1)                      # open requested while still closing: deferred
        self.assertEqual(gripper.calls, [("grasp", 20.0, 1.0)])
        gripper.future.done = True
        command.update(0.1, 0.2)
        self.assertEqual(gripper.calls, [("grasp", 20.0, 1.0), ("open",)])


class ServoTests(unittest.TestCase):
    def servo(self):
        return ev.Servo(None, None, None, None, eval_args(), None)

    def test_chunk_is_interpolated_at_100_hz(self):
        servo = self.servo()
        arm = np.stack([np.full(7, k, float) for k in range(5)])
        servo.publish(10.0, arm)
        np.testing.assert_allclose(servo.target(10.0), 0)
        np.testing.assert_allclose(servo.target(10.0 + 1.5 / ev.POLICY_HZ), 1.5)
        np.testing.assert_allclose(servo.target(10.0 + 9 / ev.POLICY_HZ), 4)      # holds the last target

    def test_stalled_policy_aborts(self):
        servo = self.servo()
        servo.publish(0.0, np.zeros((3, 7)))
        with self.assertRaisesRegex(RuntimeError, "stalled"):
            servo.target(5.0)

    def test_hold_keeps_the_current_reference(self):
        servo = self.servo()
        servo.q_ref = np.full(7, 0.3)
        servo.publish(0.0, np.zeros((3, 7)))
        servo.hold()
        np.testing.assert_allclose(servo.target(100.0), 0.3)


class EnsembleTests(unittest.TestCase):
    def test_oldest_prediction_weighs_most(self):
        a = np.zeros((5, 8))
        b = np.ones((5, 8))
        value = ev.ensemble_action([(0, a), (1, b)], 2, 0.01)
        self.assertLess(value[0], 0.5)


if __name__ == "__main__":
    unittest.main()
