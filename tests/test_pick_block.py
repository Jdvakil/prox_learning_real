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


class FakeRobot:
    def __init__(self):
        self.pose = np.eye(4)
        self.moves = []

    def move(self, target):
        self.moves.append(target)
        if isinstance(target, np.ndarray):
            self.pose = target.copy()
        pick.time.sleep(0.05)

    def recover_from_errors(self):
        pass


def fake_franky():
    return SimpleNamespace(Affine=lambda m: np.asarray(m), CartesianMotion=lambda target: target,
                           JointMotion=lambda q: list(q))


class PickTests(unittest.TestCase):
    def run_episode(self, grasp_ok, width):
        robot = FakeRobot()
        grip = SimpleNamespace(grasp=lambda *a, **kw: grasp_ok, open=lambda speed: True)
        reader = SimpleNamespace(gripper=grip, width=width)
        grasp_pose = np.eye(4)
        grasp_pose[:3, 3] = [0.5, 0.0, 0.05]

        def state(robot):
            q = np.full(7, robot.pose[2, 3])
            return dict(q=q, dq=np.zeros(7), O_T_EE=robot.pose.copy(), translation=robot.pose[:3, 3].copy(),
                        quaternion_xyzw=np.array([0, 0, 0, 1.0]), elbow=np.zeros(2))

        args = pick.parse_args(["--no-cameras", "--lead", "0", "--hold", "0.1"])
        with tempfile.TemporaryDirectory() as temp, patch.dict(sys.modules, {"franky": fake_franky()}), \
                patch.object(pick.base, "read_robot_state", side_effect=state):
            row = Path(temp) / "episode_0000"
            runner, holding = pick.collect_episode(robot, reader, SimpleNamespace(snapshot=lambda: {}), [],
                                                   grasp_pose, args, row, np.random.default_rng(0), None)
            result = json.loads((row / "result.json").read_text())
            with h5py.File(row / "trajectory.h5") as handle:
                traj = handle["traj_0"]
                actions = [json.loads(bytes(r).rstrip(b"\x00")) for r in traj["actions/joint_pos"][:]]
                q = traj["real/q"][:]
                phases = traj["obs/extra/policy_phase"][:]
                success = bool(traj["success"][-1])
            pick.put_back(robot, grip, runner, args)
        return robot, runner, holding, result, actions, q, phases, success

    def test_successful_pick_is_labelled_and_block_is_put_back(self):
        robot, runner, holding, result, actions, q, phases, success = self.run_episode(True, 0.03)
        self.assertTrue(holding and success and result["clean_success"])
        # approach, descend, lift (recorded); then put back: down, up, home.
        np.testing.assert_allclose(robot.moves[0][:3, 3], [0.5, 0, 0.15])
        np.testing.assert_allclose(robot.moves[1][:3, 3], [0.5, 0, 0.05])
        np.testing.assert_allclose(robot.moves[2][:3, 3], [0.5, 0, 0.15])
        np.testing.assert_allclose(robot.moves[3][:3, 3], [0.5, 0, 0.05])
        self.assertEqual(robot.moves[-1], pick.HOME)
        np.testing.assert_allclose([a["arm"] for a in actions[:-1]], q[1:])
        grip = [a["gripper"][0] for a in actions]
        for p, g in zip(phases, grip):
            self.assertEqual(g, pick.GRIPPER_COMMAND[pick.PHASES[p]])
        self.assertIn(pick.PHASES.index("hold"), phases)

    def test_empty_gripper_is_a_failed_episode(self):
        _, _, holding, result, _, _, _, success = self.run_episode(True, 0.0005)
        self.assertFalse(holding or success or result["clean_success"])


if __name__ == "__main__":
    unittest.main()
