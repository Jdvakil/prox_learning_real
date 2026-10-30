import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import h5py
import numpy as np

import record_act_waypoints as act


class ReferenceTests(unittest.TestCase):
    def test_endpoints_and_analytic_velocity(self):
        plan = act.ReferencePlan(act.START, act.GOAL, seed=12)
        q, dq, _, noise = plan.evaluate([0, plan.duration])
        np.testing.assert_allclose(q, [act.START, act.GOAL], atol=1e-14)
        np.testing.assert_allclose(dq, 0, atol=1e-14)
        np.testing.assert_allclose(noise, 0, atol=1e-14)
        t = np.linspace(0.1, 7.9, 100)
        eps = 1e-5
        numerical = (plan.evaluate(t + eps)[0] - plan.evaluate(t - eps)[0]) / (2 * eps)
        np.testing.assert_allclose(plan.evaluate(t)[1], numerical, atol=1e-8)

    def test_seeded_variation_and_noise_free_baseline(self):
        a = act.ReferencePlan(act.START, act.GOAL, seed=1)
        b = act.ReferencePlan(act.START, act.GOAL, seed=2)
        np.testing.assert_array_equal(a.evaluate(3)[0],
                                      act.ReferencePlan(act.START, act.GOAL, seed=1).evaluate(3)[0])
        self.assertGreater(np.abs(a.evaluate(3)[0] - b.evaluate(3)[0]).max(), 0.001)
        zero = act.ReferencePlan(act.START, act.GOAL, noise_std=0)
        np.testing.assert_array_equal(zero.evaluate(3)[3], np.zeros(7))

    def test_preflight_all_twenty_default_seeds(self):
        for seed in range(20):
            act.ReferencePlan(act.START, act.GOAL, seed=seed).validate()

    def test_preflight_rejects_bad_references(self):
        for kwargs in ({"duration": -1}, {"noise_std": float("nan")},
                       {"start": [0] * 6}):
            values = dict(start=act.START, goal=act.GOAL)
            values.update(kwargs)
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                act.ReferencePlan(**values)
        for plan in (act.ReferencePlan(act.START, [4] * 7),
                     act.ReferencePlan(act.START, act.GOAL, duration=0.1),
                     act.ReferencePlan(act.START, act.GOAL, noise_std=1)):
            with self.subTest(plan=plan), self.assertRaises(ValueError):
                plan.validate()

    def test_dry_run_writes_two_plans_without_franky(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "batch"
            with patch.dict(sys.modules, {"franky": None}):
                act.main(["--dry-run", "--output", str(output)])
            files = sorted(output.glob("*_plan.npz"))
            self.assertEqual(len(files), 2)
            with np.load(files[1]) as plan:
                self.assertEqual(int(plan["seed"]), 1)
                np.testing.assert_allclose(plan["q_ref"][-1], act.GOAL)


class MockRobotTests(unittest.TestCase):
    def test_recorded_actions_match_streamed_references_and_reset_is_excluded(self):
        robot = SimpleNamespace(state=SimpleNamespace(q=np.array(act.START)), moves=[])

        def move(q):
            robot.moves.append(q)
            robot.state.q = np.array(q)

        robot.move = move

        class Tracker:
            def __init__(self, robot, **kwargs):
                self.robot = robot

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def tick(self):
                # Production tracker paces at 100 Hz; use a shorter test interval.
                act.time.sleep(0.001)
                return 0.001

            def set_target(self, q, dq):
                self.robot.state.q = q.copy()

        def state(robot):
            return dict(q=robot.state.q.copy(), dq=np.zeros(7), O_T_EE=np.eye(4),
                        translation=np.zeros(3), quaternion_xyzw=np.array([0, 0, 0, 1]),
                        elbow=np.zeros(2))

        args = act.parse_args(["--no-cameras", "--no-gripper", "--lead", "0",
                               "--reset-settle", "0", "--tail", "0.01"])
        listener = SimpleNamespace(snapshot=lambda: {})
        # Shorten wall time without changing the path mathematics under test.
        plan = act.ReferencePlan(act.START, act.GOAL, duration=0.15, seed=9)
        with tempfile.TemporaryDirectory() as temp, patch.dict(sys.modules, {
            "franky": SimpleNamespace(JointMotion=lambda q: q, JointImpedanceTracker=Tracker)
        }), patch.object(act.base, "read_robot_state", side_effect=state):
            row = Path(temp) / "episode"
            act.collect_episode(robot, listener, None, [], plan, args, row, None)
            self.assertEqual(len(robot.moves), 1)
            self.assertTrue(json.loads((row / "result.json").read_text())["clean_success"])
            with h5py.File(row / "trajectory.h5") as handle:
                traj = handle["traj_0"]
                actions = [json.loads(bytes(row).rstrip(b"\x00"))["arm"]
                           for row in traj["actions/joint_pos"][:]]
                q_ref = traj["real/target_q"][:]
                np.testing.assert_allclose(actions, q_ref)
                self.assertGreater(np.abs(q_ref[1] - plan.goal).max(), 0.001)
                np.testing.assert_allclose(q_ref[0], plan.start)
                np.testing.assert_allclose(q_ref[-1], plan.goal)
                control = traj["real/control/q_ref"][:]
                # Every post-lead action was actually sent, including perturbed positions.
                for action in q_ref[1:]:
                    self.assertTrue(np.any(np.all(control == action, axis=1)))
                self.assertTrue(traj["success"][-1])
                self.assertFalse(np.any(traj["fail"][:]))

    def test_tracker_failure_is_reported(self):
        args = act.parse_args(["--dry-run"])
        runner = act.ReferenceRunner(None, act.ReferencePlan(act.START, act.GOAL), args, None)
        with patch.dict(sys.modules, {"franky": SimpleNamespace(
            JointImpedanceTracker=lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("reflex"))
        )}):
            runner.start()
            runner.join(1)
        self.assertTrue(runner.done.is_set())
        self.assertIsInstance(runner.error, RuntimeError)


if __name__ == "__main__":
    unittest.main()
