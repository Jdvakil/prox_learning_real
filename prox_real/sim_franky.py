"""A MuJoCo stand-in for the slice of franky that pick_block.py uses, for dry runs on a desktop.

install() puts a fake `franky` module in sys.modules. Behind it a thread steps MuJoCo physics in
real time at 1 kHz: the FR3 + Franka Hand model of the parent repo, a table and a block centred
at the saved grasp pose. The arm is torque controlled like the real robot: joint impedance
(stiffness * position error + damping * velocity error) plus gravity and Coriolis compensation;
outside a tracker it holds its last position. Gripper commands drive the hand's finger actuator
with the grasp force as its force limit and report success with libfranka's epsilon rule.

Poses come from the same FR3 DH kinematics as pick_block.py, so logged O_T_EE uses the real
robot's convention. Needs mujoco and the parent repo's assets/ (not on the robot PC).
"""

import sys
import threading
import time
import types
from pathlib import Path

import numpy as np

MODEL = Path(__file__).resolve().parents[3] / "assets/robots/franka_skin/model_hybrid_franka_hand.xml"
TIMESTEP = 0.001
HOLD_STIFFNESS = np.array([3000, 3000, 3000, 2500, 1500, 1000, 500.0])
HOLD_DAMPING = np.array([100, 100, 100, 80, 50, 40, 20.0])
FR3_VMAX = np.array([2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26])
MAX_WIDTH = 0.08
# Mass at the hand that the robot's gravity model leaves out (skin, sensors, camera). The real arm
# shows ~5 Nm on joint 2: 0.0086 rad lag at stiffness 600 while holding, and a 0.1 rad/s sink under
# franky's damping-only stop (2026-11-11 logs). 1.2 kg at the hand reproduces both.
UNMODELED_LOAD_KG = 1.2
STOP_DAMPING = np.array([50, 50, 50, 50, 30, 25, 15.0])     # franky TorqueStopMotion: damping only


class World(threading.Thread):
    """Physics in real time. All access to model/data goes through self.lock."""

    def __init__(self, grasp_pose, grasp_q, f_t_ee, block, home, viewer=False):
        super().__init__(daemon=True)
        import mujoco

        self.mujoco = mujoco
        if not MODEL.exists():
            raise SystemExit(f"--sim needs the parent repo's MuJoCo model: {MODEL} missing")
        spec = mujoco.MjSpec.from_file(str(MODEL))
        spec.option.timestep = TIMESTEP
        spec.option.noslip_iterations = 10         # rigid-finger grip; soft contacts creep otherwise
        grasp = np.asarray(grasp_pose)[:3, 3]
        # The block is square to the hand at the grasp joints, centred on the tool point.
        probe = spec.compile()
        data = mujoco.MjData(probe)
        data.qpos[:7] = grasp_q
        mujoco.mj_forward(probe, data)
        hand = data.xmat[probe.body("fr3_hand").id].reshape(3, 3)
        yaw = np.arctan2(hand[1, 0], hand[0, 0])
        self.block_size = block
        table_top = grasp[2] - (block / 2 if block > 0 else 0.025)
        table = spec.worldbody.add_body(name="table", pos=[grasp[0], grasp[1], table_top - 0.02])
        table.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.15, 0.15, 0.02], rgba=[0.6, 0.5, 0.4, 1],
                       friction=[1.0, 0.01, 0.001])
        if block > 0:
            body = spec.worldbody.add_body(name="block", pos=[grasp[0], grasp[1], table_top + block / 2],
                                           quat=[np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
            body.add_freejoint()
            body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[block / 2] * 3, mass=0.1, rgba=[0.9, 0.2, 0.2, 1],
                          friction=[1.5, 0.02, 0.001], condim=4)
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.arm_dofs = np.array([m.joint(f"fr3_joint{i}").dofadr[0] for i in range(1, 8)])
        self.arm_qpos = np.array([m.joint(f"fr3_joint{i}").qposadr[0] for i in range(1, 8)])
        self.fingers = [m.joint(f"fr3_finger_joint{i}").qposadr[0] for i in (1, 2)]
        self.finger_dofs = [m.joint(f"fr3_finger_joint{i}").dofadr[0] for i in (1, 2)]
        self.hand_actuator = m.actuator("gripper/fingers_actuator").id
        for i in range(m.nu):                    # arm actuators off: torques come from step()
            if i != self.hand_actuator:
                m.actuator_gainprm[i, :] = 0
                m.actuator_biasprm[i, :] = 0
        # The model's finger servo is soft (100 N/m); make it stiff so the force limit sets the
        # squeeze, as on the Franka Hand. Same zero point, so set_width() is unchanged.
        h = self.hand_actuator
        m.actuator_gainprm[h, 0] *= 100
        m.actuator_biasprm[h, :2] *= 100
        m.actuator_biasprm[h, 2] = -50
        self.block_qpos = m.joint(m.body("block").jntadr[0]).qposadr[0] if block > 0 else None
        self.data.qpos[self.arm_qpos] = home
        self.set_width(MAX_WIDTH)
        self.data.qpos[self.fingers] = MAX_WIDTH / 2
        mujoco.mj_forward(m, self.data)
        self.f_t_ee = np.asarray(f_t_ee)
        self.lock = threading.Lock()
        self.stiffness, self.damping = HOLD_STIFFNESS, HOLD_DAMPING
        self.q_ref, self.dq_ref = np.array(home, dtype=np.float64), np.zeros(7)
        self.max_torque = np.zeros(7)
        self.block_top = -np.inf          # highest block centre since reset_block_stats()
        self.stop_event = threading.Event()
        self.viewer = viewer
        self.hand_body = m.body("fr3_hand").id

    def set_width(self, width):
        """Finger actuator target: force = 4 - ctrl / 63.75 - 100 * opening per finger."""
        self.data.ctrl[self.hand_actuator] = (4 - 100 * width / 2) * 63.75

    def width(self):
        return float(self.data.qpos[self.fingers].sum())

    def finger_speed(self):
        return float(np.abs(self.data.qvel[self.finger_dofs]).sum())

    def reset_block_stats(self):
        with self.lock:
            self.block_top = -np.inf
            self.max_torque = np.zeros(7)

    def block_position(self):
        with self.lock:
            return None if self.block_qpos is None else self.data.qpos[self.block_qpos:self.block_qpos + 3].copy()

    def step(self):
        d = self.data
        q, dq = d.qpos[self.arm_qpos], d.qvel[self.arm_dofs]
        tau = self.stiffness * (self.q_ref - q) + self.damping * (self.dq_ref - dq)
        self.max_torque = np.maximum(self.max_torque, np.abs(tau))
        d.qfrc_applied[:] = 0
        d.qfrc_applied[self.arm_dofs] = tau + d.qfrc_bias[self.arm_dofs]
        d.xfrc_applied[self.hand_body, 2] = -UNMODELED_LOAD_KG * 9.81      # not compensated, as on the robot
        self.mujoco.mj_step(self.model, d)
        if self.block_qpos is not None:
            self.block_top = max(self.block_top, d.qpos[self.block_qpos + 2])

    def run(self):
        viewer = None
        if self.viewer:
            import mujoco.viewer
            viewer = mujoco.viewer.launch_passive(self.model, self.data)
        start = time.monotonic()
        while not self.stop_event.is_set():
            behind = time.monotonic() - start - self.data.time
            if behind <= 0:
                time.sleep(min(-behind, 0.001))
                continue
            with self.lock:
                for _ in range(min(50, int(behind / TIMESTEP) + 1)):
                    self.step()
            if viewer is not None and self.data.time % 0.03 < 0.002:
                viewer.sync()
        if viewer is not None:
            viewer.close()


def fake_franky(world, f_t_ee):
    from .pick_block import flange

    mod = types.ModuleType("franky")

    def read():
        with world.lock:
            return world.data.qpos[world.arm_qpos].copy(), world.data.qvel[world.arm_dofs].copy()

    def pose_of(q):
        matrix = flange(q) @ f_t_ee
        w, x, y, z = _quat_wxyz(matrix[:3, :3])
        return types.SimpleNamespace(matrix=matrix, translation=matrix[:3, 3].copy(), quaternion=np.array([x, y, z, w]))

    class JointMotion:
        def __init__(self, target, *args, **kwargs):
            self.target = np.asarray(target, dtype=np.float64)

    class Robot:
        def __init__(self, host, *args, **kwargs):
            self.relative_dynamics_factor = 1.0

        def recover_from_errors(self):
            return True

        def stop(self):
            pass

        def move(self, motion, asynchronous=False):
            q0, _ = read()
            span = np.abs(motion.target - q0)
            duration = max(0.5, (1.875 * span / (FR3_VMAX * self.relative_dynamics_factor)).max())
            t0 = time.monotonic()
            while (elapsed := time.monotonic() - t0) < duration:
                s = elapsed / duration
                with world.lock:
                    world.q_ref = q0 + (10 * s**3 - 15 * s**4 + 6 * s**5) * (motion.target - q0)
                    world.dq_ref = np.zeros(7)
                time.sleep(0.01)
            with world.lock:
                world.q_ref = motion.target.copy()
            time.sleep(0.5)

        @property
        def current_joint_state(self):
            q, dq = read()
            return types.SimpleNamespace(position=q, velocity=dq)

        @property
        def current_cartesian_state(self):
            q, _ = read()
            return types.SimpleNamespace(pose=types.SimpleNamespace(end_effector_pose=pose_of(q)))

        @property
        def state(self):
            q, dq = read()
            return types.SimpleNamespace(q=q, dq=dq, elbow=types.SimpleNamespace(joint_3_pos=q[2], joint_4_flip=None))

    class TorqueStopMotion:
        def __init__(self, damping=None, ramp_duration=0.2, velocity_epsilon=0.02, max_duration=2.0, **kwargs):
            self.damping = STOP_DAMPING if damping is None else np.asarray(damping, dtype=np.float64)
            self.ramp_duration, self.velocity_epsilon, self.max_duration = ramp_duration, velocity_epsilon, max_duration

    class JointImpedanceTracker:
        """Like franky: control starts at construction; stop()/__exit__ run a damping-only stop."""

        def __init__(self, robot, stiffness, damping, lower_joint_limits, upper_joint_limits, period, **kwargs):
            self.lower, self.upper = np.asarray(lower_joint_limits), np.asarray(upper_joint_limits)
            self.period, self.stopped = period, False
            q, _ = read()
            with world.lock:
                world.stiffness, world.damping = np.asarray(stiffness, float), np.asarray(damping, float)
                world.q_ref, world.dq_ref = q, np.zeros(7)
            self.next = time.monotonic()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.stop()

        def stop(self, stop_motion=None):
            """Zero stiffness, damping only, until at rest or max_duration; then the robot idles (holds)."""
            if self.stopped:
                return
            self.stopped = True
            motion = stop_motion or TorqueStopMotion()
            with world.lock:
                world.stiffness, world.damping = np.zeros(7), motion.damping
                world.dq_ref = np.zeros(7)
            t0 = time.monotonic()
            while (elapsed := time.monotonic() - t0) < motion.max_duration:
                if elapsed > motion.ramp_duration and np.abs(read()[1]).max() < motion.velocity_epsilon:
                    break
                time.sleep(0.005)
            q, _ = read()
            with world.lock:
                world.stiffness, world.damping = HOLD_STIFFNESS, HOLD_DAMPING
                world.q_ref, world.dq_ref = q, np.zeros(7)

        def tick(self):
            if self.stopped:
                return None
            self.next += self.period
            time.sleep(max(0.0, self.next - time.monotonic()))
            return self.period

        def set_target(self, q, dq):
            q = np.asarray(q, dtype=np.float64)
            if np.any(q < self.lower) or np.any(q > self.upper):
                raise RuntimeError("target outside joint limits")
            with world.lock:
                world.q_ref, world.dq_ref = q.copy(), np.asarray(dq, dtype=np.float64).copy()

    class Future:
        def __init__(self, fn):
            self.result, self.event = None, threading.Event()

            def run():
                try:
                    self.result = fn()
                finally:
                    self.event.set()

            threading.Thread(target=run, daemon=True).start()

        def wait(self, timeout):
            return self.event.wait(timeout)

        def get(self):
            self.event.wait()
            return self.result

    class Gripper:
        def __init__(self, host, *args, **kwargs):
            pass

        @property
        def width(self):
            time.sleep(0.01)                 # libfranka gripper state arrives at ~100 Hz
            with world.lock:
                return world.width()

        def _move(self, width, speed, force):
            """Ramp the finger target at speed; done when the fingers stop moving."""
            with world.lock:
                start = world.width()
                world.model.actuator_forcerange[world.hand_actuator] = [-2 * force, 2 * force]
            t0, still = time.monotonic(), 0
            while True:
                time.sleep(0.01)
                elapsed = time.monotonic() - t0
                with world.lock:
                    goal = start + np.clip(width - start, -speed * elapsed, speed * elapsed)
                    world.set_width(goal)
                    still = still + 1 if goal == width and world.finger_speed() < 1e-3 else 0
                    current = world.width()
                if still >= 20 or elapsed > 8:
                    return current

        def grasp_async(self, width, speed, force, epsilon_inner=0.005, epsilon_outer=0.005):
            def grasp():
                final = self._move(width, speed, force)
                return width - epsilon_inner <= final <= width + epsilon_outer
            return Future(grasp)

        def open_async(self, speed):
            return Future(lambda: self._move(MAX_WIDTH, speed, 100.0) > MAX_WIDTH - 0.005)

        def open(self, speed):
            return self.open_async(speed).get()

    mod.JointMotion, mod.Robot, mod.Gripper, mod.JointImpedanceTracker = JointMotion, Robot, Gripper, JointImpedanceTracker
    mod.TorqueStopMotion = TorqueStopMotion
    return mod


def _quat_wxyz(r):
    w = np.sqrt(max(0.0, 1 + r[0, 0] + r[1, 1] + r[2, 2])) / 2
    x = np.copysign(np.sqrt(max(0.0, 1 + r[0, 0] - r[1, 1] - r[2, 2])) / 2, r[2, 1] - r[1, 2])
    y = np.copysign(np.sqrt(max(0.0, 1 - r[0, 0] + r[1, 1] - r[2, 2])) / 2, r[0, 2] - r[2, 0])
    z = np.copysign(np.sqrt(max(0.0, 1 - r[0, 0] - r[1, 1] + r[2, 2])) / 2, r[1, 0] - r[0, 1])
    return w, x, y, z


def install(saved, home, block, viewer=False):
    """Start the physics thread and make `import franky` return the simulated robot."""
    from .pick_block import flange

    grasp_pose, grasp_q = np.asarray(saved["grasp_pose"]), np.asarray(saved["grasp_q"])
    f_t_ee = np.linalg.inv(flange(grasp_q)) @ grasp_pose
    world = World(grasp_pose, grasp_q, f_t_ee, block, home, viewer)
    world.start()
    sys.modules["franky"] = fake_franky(world, f_t_ee)
    return world
