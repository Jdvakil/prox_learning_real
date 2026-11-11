import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import h5py
import numpy as np

from prox_real import pick_block as pick

GRASP_Q = np.array([0.449, 0.660, -0.076, -2.105, -0.196, 2.840, 1.352])    # the lab's saved grasp pose
F_T_EE = np.array([[0.7071, 0.7071, 0, 0], [-0.7071, 0.7071, 0, 0], [0, 0, 1, 0.1034], [0, 0, 0, 1]])
SAVED = dict(grasp_q=GRASP_Q.tolist(), grasp_pose=(pick.flange(GRASP_Q) @ F_T_EE).tolist())


def ee(q):
    return (pick.flange(q) @ F_T_EE)[:3, 3]


class Done:
    def __init__(self, value):
        self.value = value

    def wait(self, timeout):
        return True

    def get(self):
        return self.value


class Tracker:
    def __init__(self, robot, **kwargs):
        self.robot = robot

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def tick(self):
        pick.time.sleep(0.002)          # the real tracker paces at 100 Hz
        return 0.0

    def set_target(self, q, dq):
        self.robot.state.q = np.array(q)
        self.robot.history.append(np.array(q))

    def stop(self, stop_motion=None):
        self.robot.stops.append(stop_motion)


class StopMotion:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def fast_args(*extra):
    return pick.parse_args(["--no-cameras", "--lead", "0", "--hold", "0.1", "--speed", "0.5",
                            "--max-velocity", "4", *extra])


class PlanTests(unittest.TestCase):
    def test_plan_is_smooth_vertical_and_returns_home(self):
        plan = pick.Plan(SAVED, pick.parse_args(["--plan-only"]), np.random.default_rng(0))
        paths = dict((name, path) for name, path in plan.pick + plan.put_back if path is not None)
        np.testing.assert_allclose(paths["approach"][0], pick.HOME)
        np.testing.assert_allclose(paths["home"][-1], pick.HOME)
        grasp = np.asarray(SAVED["grasp_pose"])[:3, 3]
        np.testing.assert_allclose(ee(paths["descend"][-1]), grasp, atol=1e-4)
        np.testing.assert_allclose(ee(paths["lift"][-1]), grasp + [0, 0, 0.1], atol=1e-4)
        for name in ("descend", "lift", "lower", "rise"):
            xy = np.array([ee(q)[:2] for q in paths[name]])
            self.assertLess(np.abs(xy - grasp[:2]).max(), 1e-4, name)        # straight up / down
        full = np.concatenate(list(paths.values()))
        speed = np.abs(np.diff(full, axis=0)).max() * pick.CONTROL_HZ
        self.assertLess(speed, 0.45)
        for path in paths.values():                                     # every segment starts and ends at rest
            np.testing.assert_allclose(path[1] - path[0], 0, atol=1e-4)
            np.testing.assert_allclose(path[-1] - path[-2], 0, atol=1e-4)
        self.assertLess(np.abs(paths["descend"][-1] - GRASP_Q).max(), 0.01)  # elbow as posed

    def test_every_sequence_is_continuous_and_put_back_releases_above_the_table(self):
        args = pick.parse_args(["--plan-only", "--lift", "0.06"])
        plan = pick.Plan(SAVED, args, np.random.default_rng(0))
        grasp_z = np.asarray(SAVED["grasp_pose"])[2, 3]
        for sequence in (plan.pick + plan.put_back, plan.pick + plan.give_up):
            paths = [path for _, path in sequence if path is not None]
            for a, b in zip(paths, paths[1:]):
                np.testing.assert_allclose(a[-1], b[0], atol=1e-9)
        lower = dict(plan.put_back)["lower"]
        self.assertAlmostEqual(ee(lower[-1])[2], grasp_z + args.place_gap, places=4)
        rise_empty = dict(plan.give_up)["rise"]
        self.assertGreater(min(ee(q)[2] for q in rise_empty), grasp_z + 0.06 - 1e-4)

    def test_unreachable_pose_is_rejected_before_moving(self):
        with self.assertRaises(ValueError):          # 1.5 m above the block is out of reach
            pick.Plan(SAVED, pick.parse_args(["--plan-only", "--approach", "1.5"]), np.random.default_rng(0))


class MockRobotTests(unittest.TestCase):
    def run_episode(self, grasp_ok, width):
        robot = SimpleNamespace(state=SimpleNamespace(q=np.array(pick.HOME)), history=[], stops=[])
        grip = SimpleNamespace(grasp_async=lambda *a, **kw: Done(grasp_ok), open_async=lambda speed: Done(True),
                               width=width)
        reader = SimpleNamespace(gripper=grip, width=width)

        def state(robot):
            pose = pick.flange(robot.state.q) @ F_T_EE
            return dict(q=robot.state.q.copy(), dq=np.zeros(7), O_T_EE=pose, translation=pose[:3, 3],
                        quaternion_xyzw=np.array([0, 0, 0, 1.0]), elbow=np.zeros(2))

        args = fast_args()
        plan = pick.Plan(SAVED, args, np.random.default_rng(0))
        with tempfile.TemporaryDirectory() as temp, \
                patch.dict(sys.modules, {"franky": SimpleNamespace(JointImpedanceTracker=Tracker,
                                                                   TorqueStopMotion=StopMotion)}), \
                patch.object(pick.base, "read_robot_state", side_effect=state):
            row = Path(temp) / "episode_0000"
            runner, holding = pick.collect_episode(robot, reader, SimpleNamespace(snapshot=lambda: {}), [],
                                                   plan, args, row, None)
            pick.put_back(runner)
            result = json.loads((row / "result.json").read_text())
            with h5py.File(row / "trajectory.h5") as handle:
                traj = handle["traj_0"]
                actions = [json.loads(bytes(r).rstrip(b"\x00")) for r in traj["actions/joint_pos"][:]]
                q = traj["real/q"][:]
                phases = traj["obs/extra/policy_phase"][:]
                success = bool(traj["success"][-1])
        return robot, holding, result, actions, q, phases, success

    def test_successful_pick_is_labelled_and_block_is_put_back(self):
        robot, holding, result, actions, q, phases, success = self.run_episode(True, 0.03)
        self.assertTrue(holding and success and result["clean_success"])
        np.testing.assert_allclose(robot.history[-1], pick.HOME, atol=1e-9)
        lowest = min(ee(p)[2] for p in robot.history)
        self.assertAlmostEqual(lowest, np.asarray(SAVED["grasp_pose"])[2, 3], places=4)
        # The fake tracker follows exactly, so the action is the state the robot was put in.
        np.testing.assert_allclose([a["arm"] for a in actions], q, atol=1e-9)
        for p, a in zip(phases, actions):
            self.assertEqual(a["gripper"][0], pick.GRIPPER_COMMAND[pick.PHASES[p]])
        self.assertTrue(np.all(np.diff(phases) >= 0))
        self.assertEqual(phases[-1], pick.PHASES.index("hold"))

    def test_tracker_ends_with_a_short_stop(self):
        robot = self.run_episode(True, 0.03)[0]
        self.assertEqual(len(robot.stops), 1)                   # franky's default would sink the arm for 2 s
        self.assertLessEqual(robot.stops[0].kwargs["max_duration"], 0.3)

    def test_arm_away_from_home_is_refused_without_engaging_the_tracker(self):
        robot = SimpleNamespace(state=SimpleNamespace(q=np.array(pick.HOME) + [0, 0.204, 0, 0, 0, 0, 0]),
                                history=[], stops=[])
        plan = pick.Plan(SAVED, fast_args(), np.random.default_rng(0))
        engaged = []
        franky = SimpleNamespace(JointImpedanceTracker=lambda *a, **kw: engaged.append(1), TorqueStopMotion=StopMotion)
        with patch.dict(sys.modules, {"franky": franky}):
            runner = pick.PickRunner(robot, None, plan, fast_args())
            runner.start()
            runner.join(2)
        self.assertIn("not at HOME", str(runner.error))
        self.assertEqual(engaged, [])

    def test_empty_gripper_is_a_failed_episode_and_arm_still_goes_home(self):
        # libfranka's grasp passes with nothing in the hand (width 0 is inside the epsilons).
        robot, holding, result, _, _, _, success = self.run_episode(True, 0.0005)
        self.assertFalse(holding or success or result["clean_success"] or result["gripper_closed_on_block"])
        np.testing.assert_allclose(robot.history[-1], pick.HOME, atol=1e-9)
        low = [ee(q)[2] < np.asarray(SAVED["grasp_pose"])[2, 3] + 0.01 for q in robot.history]
        self.assertEqual(np.count_nonzero(np.diff(np.array(low, dtype=int)) == 1), 1)  # one visit low: no re-descent
        steps = np.abs(np.diff(np.array(robot.history), axis=0)).max()
        self.assertLess(steps, 0.4 / pick.CONTROL_HZ * 1.5 * 10)    # no jump in the streamed reference


class SimTests(unittest.TestCase):
    def test_sim_hand_grasps_a_block_at_the_grasp_pose(self):
        try:
            import mujoco  # noqa: F401
            from prox_real import sim_franky
        except ImportError:
            self.skipTest("mujoco not installed")
        if not sim_franky.MODEL.exists():
            self.skipTest("parent repo model missing")
        with patch.dict(sys.modules):
            world = sim_franky.install(SAVED, GRASP_Q, block=0.05)
            try:
                franky = sys.modules["franky"]
                robot = franky.Robot("sim")
                pose = robot.current_cartesian_state.pose.end_effector_pose
                np.testing.assert_allclose(pose.matrix, SAVED["grasp_pose"], atol=1e-3)
                ok = franky.Gripper("sim").grasp_async(0.0, 0.1, 20.0, epsilon_inner=0.005, epsilon_outer=1.0)
                self.assertTrue(ok.wait(8) and ok.get())
                self.assertAlmostEqual(franky.Gripper("sim").width, 0.05, delta=0.003)
            finally:
                world.stop_event.set()
                world.join(2)


if __name__ == "__main__":
    unittest.main()
