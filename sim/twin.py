"""Rehearse kinesthetic.py (teach and replay) on a MuJoCo copy of the lab bench.

kinesthetic.py runs unchanged. This script stands in for everything it talks to:

  franky        Robot, Gripper, JointMotion, JointImpedanceTracker, TorqueStopMotion and
                FrictionCompensationParams, backed by a thread that steps the lab scene
                (sim/lab_scene.py) in real time under gravity compensation, as the Franka
                controller does. Impedance motions apply K (q_ref - q) + D (dq_ref - dq), the
                soft joint limits and the friction feedforward; JointMotion follows a quintic
                profile timed from the FR3 limits x relative_dynamics_factor under stiff joint
                servos. grasp closes at the commanded speed, then squeezes with the commanded force.
  sensors       the 41 proximity sensors, ray-cast into the scene (the arm's own visual meshes
                included, skin shells excluded) and sent as real packets on the real multicast
                ports, each at its own ~15 Hz clock.
  pyrealsense2  the two D435s: pipelines that render wrist_camera / exo_camera_1 at 30 fps,
                colour + z depth in mm (0 = no depth), D435-like intrinsics.

--auto also stands in for you: in teach a scripted demonstrator holds and pushes the floating arm
through a pick-and-place (block on the table -> pulled-out top drawer) and presses the keys; in
replay the prompts are answered and "task done?" is scored from where the block ended up.
Without --auto the keys and prompts are yours, in this terminal; --viewer opens the MuJoCo viewer,
where you can drag the floating arm (double-click a link, then Ctrl + right-drag). The block and
drawer are reset at every "reset the scene" prompt.

  python sim/twin.py rehearse                            # teach --auto, then replay --auto
  python sim/twin.py teach --auto                        # -> sim_out/demos/<stamp>.h5
  python sim/twin.py replay --auto sim_out/demos/<stamp>.h5
                                                         # -> sim_out/dataset/rows/<stamp>_<demo>_r0/
  python sim/twin.py replay --auto <demo> -- --speed 0.5 # flags after -- go to kinesthetic.py

Needs mujoco >= 3.2 and an EGL (or other offscreen) OpenGL; MUJOCO_GL defaults to egl.
"""
from __future__ import annotations

import argparse
import os
import socket
import sys
import threading
import time
import types
from pathlib import Path

import numpy as np

SIM = Path(__file__).resolve().parent
REAL = SIM.parent
sys.path.insert(0, str(REAL))
sys.path.insert(0, str(SIM))
os.environ.setdefault("MUJOCO_GL", "egl")

import h5py  # noqa: E402
import mujoco  # noqa: E402

import lab_scene  # noqa: E402
from sensor_layout import LINKS, NO_TARGET_MM, SENSORS_PER_LINK, sim_name  # noqa: E402

GROUP = "239.0.0.1"
MAX_RANGE_M = 4.0
ZONES, SENSOR_FOVY = 8, 45.0
RAY_GROUPS = np.array([1, 1, 1, 0, 0, 0], np.uint8)        # scene + arm visual meshes; skin is in group 5
RENDER_GROUPS = np.array([1, 1, 1, 0, 0, 1], np.uint8)     # what a D435 sees: scene, arm and skin
FR3_VMAX = np.array([2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26])
FR3_AMAX = np.full(7, 10.0)
FR3_JMAX = np.full(7, 5000.0)
EE_YAW = np.deg2rad(-45.0)    # franky O_T_EE vs the model's fr3_hand_tcp site, measured on the arm
SERVO_KP = np.array([4500.0, 4500.0, 3500.0, 3500.0, 2000.0, 2000.0, 2000.0])   # the model's servos
SERVO_KV = SERVO_KP / 10.0
TAU_MAX = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])                  # FR3 joint torque limits
STOP_DAMPING = np.array([10.0, 10.0, 10.0, 10.0, 5.0, 5.0, 2.0])
ZERO7 = np.zeros(7)
GRIP_GAIN, GRIP_BIAS = -0.0156863, np.array([4.0, -100.0, -10.0])   # finger actuator: ctrl 0 open .. 255 closed
MAX_HALF_WIDTH = 0.04
SIM_SERIALS = {"SIM-WRIST": "wrist_camera", "SIM-EXO": "exo_camera_1"}


class Physics(threading.Thread):
    """Steps the scene in real time; the fake franky classes set what the arm and gripper do."""

    def __init__(self, model, layout):
        super().__init__(daemon=True)
        self.m, self.lay = model, layout
        self.d = mujoco.MjData(model)
        self.lock = threading.Lock()
        self.d.qpos[layout.arm_q] = lab_scene.HOME
        self.d.qpos[layout.finger_q] = MAX_HALF_WIDTH
        layout.reset_scene(self.d)
        mujoco.mj_forward(model, self.d)
        self.mode = "hold"                   # hold | trajectory | impedance | torque_stop
        self.q_hold = lab_scene.HOME.copy()
        self.traj = None                     # (start, q0, delta, duration)
        self.imp = None                      # JointImpedanceTrackingMotion being run
        self.stop_deadline = 0.0
        self.in_control = False
        self.motion_done = threading.Event()
        self.motion_done.set()
        self.error = None
        self.grip = {"mode": "position", "goal": MAX_HALF_WIDTH, "target": MAX_HALF_WIDTH, "speed": 0.05,
                     "force": 20.0, "since": 0.0, "future": None}
        self.human = None                    # callable(now, q, dq) -> joint torques, e.g. the demonstrator
        self.viewer = None
        self.tau_d = np.zeros(7)
        self.started = time.monotonic()
        self.lag_drops = []                  # (when, seconds dropped) each time the physics fell behind
        self.stop_event = threading.Event()
        self._publish(time.monotonic())

    # ---- commands (any thread) ----

    def start_trajectory(self, target, factor):
        q0 = self.snap.qpos[self.lay.arm_q].copy()
        delta = np.asarray(target, float) - q0
        d = np.abs(delta) + 1e-12
        f = max(float(factor), 1e-3)
        duration = float(np.max(np.maximum.reduce([
            1.875 * d / (FR3_VMAX * f),                 # quintic peak velocity
            np.sqrt(5.7735 * d / (FR3_AMAX * f)),       # quintic peak acceleration
            np.cbrt(60.0 * d / (FR3_JMAX * f)),         # quintic peak jerk
        ])))
        with self.lock:
            self.traj = (time.monotonic(), q0, delta, duration)
            self.mode, self.in_control = "trajectory", True
            self.motion_done.clear()

    def start_impedance(self, motion):
        with self.lock:
            self.imp, self.mode, self.in_control = motion, "impedance", True
            self.motion_done.clear()

    def start_torque_stop(self, max_duration):
        with self.lock:
            if self.mode != "impedance":
                return
            self.mode, self.stop_deadline = "torque_stop", time.monotonic() + max_duration

    def gripper_command(self, kind, half_width, speed, force, future):
        with self.lock:
            old = self.grip["future"]
            if old is not None:
                old.set(False)
            target = float(np.mean(self.d.qpos[self.lay.finger_q]))
            self.grip.update(mode="closing" if kind == "grasp" else "position", goal=float(half_width),
                             target=target, speed=float(speed) / 2.0, force=float(force),
                             since=time.monotonic(), future=future)

    def reset_scene(self):
        with self.lock:
            self.lay.reset_scene(self.d)
            mujoco.mj_forward(self.m, self.d)

    # ---- physics thread ----

    def run(self):
        dt = self.m.opt.timestep
        wall0, sim0 = time.monotonic(), self.d.time
        next_view = 0.0
        while not self.stop_event.is_set():
            now = time.monotonic()
            steps = int((sim0 + (now - wall0) - self.d.time) / dt)
            if steps > 100:                  # > 0.2 s behind: drop the time instead of racing
                self.lag_drops.append((now - self.started, steps * dt))
                wall0, sim0, steps = now, self.d.time, 1
            with self.lock:
                for _ in range(max(steps, 0)):
                    self._control(wall0 + (self.d.time - sim0))
                    mujoco.mj_step(self.m, self.d)
                self._publish(wall0 + (self.d.time - sim0))
                if self.viewer is not None and now >= next_view:
                    if self.viewer.is_running():
                        self.viewer.sync()
                    next_view = now + 1 / 60
            time.sleep(0.001)

    def _servo(self, target, kp=SERVO_KP, kv=SERVO_KV):
        acts = self.lay.arm_act
        self.m.actuator_gainprm[acts, 0] = kp
        self.m.actuator_biasprm[acts, 1] = -kp
        self.m.actuator_biasprm[acts, 2] = -kv
        self.d.ctrl[acts] = target

    def _control(self, now):
        lay = self.lay
        q, dq = self.d.qpos[lay.arm_q], self.d.qvel[lay.arm_v]
        extra = np.zeros(7)
        if self.mode == "trajectory":
            start, q0, delta, duration = self.traj
            s = min((now - start) / duration, 1.0) if duration > 0 else 1.0
            target = q0 + delta * (s ** 3 * (10 - 15 * s + 6 * s ** 2))
            self._servo(target)
            self.tau_d = SERVO_KP * (target - q) - SERVO_KV * dq
            if s >= 1.0:
                self.mode, self.q_hold, self.in_control = "hold", q0 + delta, False
                self.motion_done.set()
        elif self.mode == "hold":
            self._servo(self.q_hold)
            self.tau_d = SERVO_KP * (self.q_hold - q) - SERVO_KV * dq
        elif self.mode == "impedance":
            # Servos off; the whole law is one torque, clipped to the joint limits like on the arm.
            p = self.imp
            self._servo(q, ZERO7, ZERO7)
            tau = (p.K * (p.q_ref - q) + p.D * (p.dq_ref - dq) + p.tau_ff + p.friction_torque(dq)
                   + p.limit_torque(q, dq))
            extra = self.tau_d = np.clip(tau, -TAU_MAX, TAU_MAX)
        elif self.mode == "torque_stop":
            self._servo(q, ZERO7, ZERO7)
            extra = self.tau_d = np.clip(-STOP_DAMPING * dq, -TAU_MAX, TAU_MAX)
            if np.abs(dq).max() < 0.01 or now > self.stop_deadline:
                self.mode, self.q_hold, self.in_control = "hold", q.copy(), False
                self.motion_done.set()
        human = self.human(now, q, dq) if self.human is not None else 0.0
        self.d.qfrc_applied[lay.arm_v] = extra + human
        self._gripper(now)

    def _gripper(self, now):
        g, lay, m = self.grip, self.lay, self.m
        half = float(np.mean(self.d.qpos[lay.finger_q]))
        a = lay.grip_act
        if g["mode"] in ("position", "closing"):
            step = g["speed"] * m.opt.timestep
            g["target"] += float(np.clip(g["goal"] - g["target"], -step, step))
            m.actuator_gainprm[a, 0] = GRIP_GAIN
            m.actuator_biasprm[a, :3] = GRIP_BIAS
            self.d.ctrl[a] = np.clip((GRIP_BIAS[0] + GRIP_BIAS[1] * g["target"]) / -GRIP_GAIN, 0, 255)
            if g["mode"] == "closing" and half - g["target"] > 0.003:      # fingers stopped on the object
                g["mode"], g["since"] = "force", now
            elif abs(g["target"] - g["goal"]) < 1e-9 and abs(half - g["goal"]) < 0.002:
                self._grip_done(True)
            elif now - g["since"] > 5.0:
                self._grip_done(False)
        elif g["mode"] == "force":
            if half < 0.0005:                # the object slipped out: hold closed, as the hand does
                g.update(mode="position", goal=0.0, target=0.0)
                self._grip_done(True)
                return
            m.actuator_gainprm[a, 0] = 0.0
            m.actuator_biasprm[a, :3] = [-2.0 * g["force"], 0.0, GRIP_BIAS[2]]
            if now - g["since"] > 0.15:
                self._grip_done(True)

    def _grip_done(self, ok):
        future, self.grip["future"] = self.grip["future"], None
        if future is not None:
            future.set(ok)

    def _publish(self, now):
        lay = self.lay
        self.snap = types.SimpleNamespace(
            t=now, qpos=self.d.qpos.copy(), qvel=self.d.qvel.copy(), q_d=np.array(self.d.ctrl[lay.arm_act]),
            tau_J=self.d.qfrc_actuator[lay.arm_v] + self.d.qfrc_applied[lay.arm_v], tau_J_d=self.tau_d.copy(),
            grasped=self.grip["mode"] == "force")


def ee_pose(model, data, layout):
    """franky-style O_T_EE (4x4) of the kinematics in `data`."""
    c, s = np.cos(EE_YAW), np.sin(EE_YAW)
    R = data.site_xmat[layout.tcp_site].reshape(3, 3) @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, data.site_xpos[layout.tcp_site]
    return T


def fake_franky(phys: Physics) -> types.ModuleType:
    """The slice of the franky 2.0 API kinesthetic.py uses."""
    mod = types.ModuleType("franky")
    ns = types.SimpleNamespace
    fk_data, fk_lock = mujoco.MjData(phys.m), threading.Lock()

    class ControlException(Exception):
        pass

    class InvalidMotionTypeException(Exception):
        pass

    class BoolFuture:
        def __init__(self):
            self._event, self._value = threading.Event(), None

        def set(self, value):
            self._value = bool(value)
            self._event.set()

        def wait(self, timeout=None):
            return self._event.wait(timeout)

        def get(self):
            self._event.wait()
            return self._value

    class FrictionCompensationParams:
        def __init__(self, coulomb=None, viscous=None, max_torque=None, velocity_epsilon=0.03):
            self.coulomb = np.zeros(7) if coulomb is None else np.asarray(coulomb, float)
            self.viscous = np.zeros(7) if viscous is None else np.asarray(viscous, float)
            self.max_torque = np.full(7, np.inf) if max_torque is None else np.asarray(max_torque, float)
            self.velocity_epsilon = velocity_epsilon

    class JointReference:
        def __init__(self, q, dq=None, tau_ff=None):
            self.q = np.asarray(q, float)
            self.dq = np.zeros(7) if dq is None else np.asarray(dq, float)
            self.tau_ff = np.zeros(7) if tau_ff is None else np.asarray(tau_ff, float)

    class JointImpedanceGains:
        def __init__(self, stiffness, damping=None):
            self.stiffness = np.asarray(stiffness, float)
            self.damping = None if damping is None else np.asarray(damping, float)

    class JointImpedanceTrackingMotion:
        def __init__(self, stiffness, damping=None, friction=None, lower_joint_limits=None,
                     upper_joint_limits=None, joint_limit_activation_distance=0.1, joint_limit_stiffness=4.0,
                     joint_limit_damping=1.0, joint_limit_max_torque=5.0, **unused):
            self.set_gains(JointImpedanceGains(stiffness, damping))
            self.friction = friction
            self.lower = None if lower_joint_limits is None else np.asarray(lower_joint_limits, float)
            self.upper = None if upper_joint_limits is None else np.asarray(upper_joint_limits, float)
            self.limit = (joint_limit_activation_distance, joint_limit_stiffness, joint_limit_damping,
                          joint_limit_max_torque)
            self.q_ref, self.dq_ref, self.tau_ff = np.zeros(7), np.zeros(7), np.zeros(7)

        def set_reference(self, ref):
            with phys.lock:
                self.q_ref, self.dq_ref, self.tau_ff = ref.q, ref.dq, ref.tau_ff

        def set_gains(self, gains):
            self.K = np.asarray(gains.stiffness, float)
            self.D = 2.0 * np.sqrt(self.K) if gains.damping is None else np.asarray(gains.damping, float)

        def get_gains(self):
            return JointImpedanceGains(self.K.copy(), self.D.copy())

        def friction_torque(self, dq):
            f = self.friction
            if f is None:
                return 0.0
            tau = f.coulomb * np.tanh(dq / f.velocity_epsilon) + f.viscous * dq
            return np.clip(tau, -f.max_torque, f.max_torque)

        def limit_torque(self, q, dq):
            if self.lower is None or self.upper is None:
                return 0.0
            reach, k, c, cap = self.limit
            push = np.clip(reach - (q - self.lower), 0, None) - np.clip(reach - (self.upper - q), 0, None)
            tau = k * push - c * dq * (push != 0)
            return np.clip(tau, -cap, cap)

    class TorqueStopMotion:
        def __init__(self, damping=None, max_duration=2.0, **unused):
            self.max_duration = max_duration

    class JointMotion:
        def __init__(self, target, *args, **kwargs):
            self.target = np.asarray(target, float)

    class Robot:
        def __init__(self, host, *args, **kwargs):
            self.relative_dynamics_factor = 1.0
            self.collision_behavior = None

        def recover_from_errors(self):
            phys.error = None
            return True

        def set_collision_behavior(self, *thresholds):
            self.collision_behavior = thresholds

        def move(self, motion, asynchronous=False):
            if phys.error is not None:
                raise ControlException(phys.error)
            if isinstance(motion, JointMotion):
                if phys.mode in ("impedance", "torque_stop"):
                    raise InvalidMotionTypeException("the type of motion cannot change during runtime")
                factor = self.relative_dynamics_factor
                phys.start_trajectory(motion.target, getattr(factor, "value", factor))
            elif isinstance(motion, JointImpedanceTrackingMotion):
                if phys.mode == "trajectory":
                    raise InvalidMotionTypeException("the type of motion cannot change during runtime")
                phys.start_impedance(motion)
            elif isinstance(motion, TorqueStopMotion):
                phys.start_torque_stop(motion.max_duration)
            else:
                raise TypeError(f"sim franky does not run {type(motion).__name__}")
            if not asynchronous:
                self.join_motion()

        def join_motion(self, timeout=None):
            finished = phys.motion_done.wait(timeout)
            if phys.error is not None:
                raise ControlException(phys.error)
            return finished

        def poll_motion(self):
            return phys.motion_done.is_set()

        @property
        def is_in_control(self):
            return phys.in_control

        @property
        def current_joint_positions(self):
            return phys.snap.qpos[phys.lay.arm_q].copy()

        @property
        def current_joint_state(self):
            s = phys.snap
            return ns(position=s.qpos[phys.lay.arm_q].copy(), velocity=s.qvel[phys.lay.arm_v].copy())

        @property
        def state(self):
            s = phys.snap
            with fk_lock:
                fk_data.qpos[:] = s.qpos
                mujoco.mj_kinematics(phys.m, fk_data)
                T = ee_pose(phys.m, fk_data, phys.lay)
            q = s.qpos[phys.lay.arm_q].copy()
            return ns(q=q, dq=s.qvel[phys.lay.arm_v].copy(), q_d=s.q_d, tau_J=s.tau_J, tau_J_d=s.tau_J_d,
                      O_T_EE=ns(matrix=T, translation=T[:3, 3].copy()),
                      elbow=ns(joint_3_pos=float(q[2]), joint_4_flip=None), time=ns(to_sec=lambda: s.t))

    class JointImpedanceTracker:
        """Same contract as franky.JointImpedanceTracker (franky/tracker.py)."""

        def __init__(self, robot, *, stiffness=None, damping=None, friction=None, lower_joint_limits=None,
                     upper_joint_limits=None, period=None, **kwargs):
            self._robot, self._period = robot, period
            self._tick_count, self._t_next, self._t_last, self._dt = 0, 0.0, 0.0, 0.0
            self._t_start = time.perf_counter()
            self._motion = JointImpedanceTrackingMotion(
                np.full(7, 50.0) if stiffness is None else stiffness, damping, friction,
                lower_joint_limits, upper_joint_limits, **kwargs)
            self._motion.set_reference(JointReference(q=robot.current_joint_positions))
            robot.move(self._motion, asynchronous=True)

        def tick(self):
            first = self._tick_count == 0
            if self._period is not None:
                if first:
                    self._t_next = time.perf_counter() + self._period
                else:
                    remaining = self._t_next - time.perf_counter()
                    if remaining > 0:
                        time.sleep(remaining)
                    self._t_next += self._period
            if not self._robot.is_in_control:
                return None
            now = time.perf_counter()
            self._dt = (self._period or 1e-9) if first else max(now - self._t_last, 1e-9)
            self._t_last = now
            self._tick_count += 1
            return self._dt

        def set_target(self, q, dq=None, tau_ff=None):
            self._motion.set_reference(JointReference(q, dq, tau_ff))

        def set_gains(self, *, stiffness=None, damping=None):
            g = self._motion.get_gains()
            self._motion.set_gains(JointImpedanceGains(g.stiffness if stiffness is None else stiffness, damping))

        @property
        def state(self):
            return self._robot.state

        @property
        def current_joint_state(self):
            return self._robot.current_joint_state

        @property
        def is_running(self):
            return self._robot.is_in_control

        @property
        def motion(self):
            return self._motion

        def stop(self, stop_motion=None):
            if self._robot.is_in_control:
                self._robot.move(stop_motion or TorqueStopMotion(), asynchronous=True)
            self._robot.join_motion()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            try:
                self.stop()
            except ControlException:
                if exc_type is not None:
                    return False
                raise
            return False

    class Gripper:
        def __init__(self, host, *args, **kwargs):
            self.max_width = 2 * MAX_HALF_WIDTH

        def _command(self, kind, width, speed, force=0.0):
            future = BoolFuture()
            phys.gripper_command(kind, np.clip(width / 2.0, 0.0, MAX_HALF_WIDTH), speed, force, future)
            return future

        @property
        def width(self):
            time.sleep(0.02)                 # the real call waits for the next gripper state
            return 2.0 * float(np.mean(phys.snap.qpos[phys.lay.finger_q]))

        @property
        def is_grasped(self):
            return bool(phys.snap.grasped)

        def grasp_async(self, width, speed, force, epsilon_inner=0.005, epsilon_outer=0.005):
            return self._command("grasp", width, speed, force)

        def grasp(self, *args, **kwargs):
            return self.grasp_async(*args, **kwargs).get()

        def move_async(self, width, speed):
            return self._command("move", width, speed)

        def move(self, width, speed):
            return self.move_async(width, speed).get()

        def open_async(self, speed):
            return self._command("move", self.max_width, speed)

        def open(self, speed):
            return self.open_async(speed).get()

        def homing(self):
            return self.open(0.05)

    for cls in (ControlException, InvalidMotionTypeException, BoolFuture, FrictionCompensationParams, JointReference,
                JointImpedanceGains, JointImpedanceTrackingMotion, TorqueStopMotion, JointMotion, Robot,
                JointImpedanceTracker, Gripper):
        setattr(mod, cls.__name__, cls)
    mod.CRITICAL = None
    return mod


class CameraRig(threading.Thread):
    """Renders every started camera from one thread: one GL context, ~3 ms per camera for colour +
    depth (a context per thread is ~30x slower)."""

    def __init__(self, phys):
        super().__init__(daemon=True)
        self.phys = phys
        self.cond = threading.Condition()
        self.cams = {}               # scene camera -> {"size", "fps", "frame", "number"}
        self.error = None
        self.stop_event = threading.Event()

    def add(self, camera, width, height, fps):
        with self.cond:
            self.cams[camera] = {"size": (width, height), "fps": fps, "frame": None, "number": 0}
        if not self.is_alive():
            self.start()

    def run(self):
        try:
            m = self.phys.m
            data, option, renderers = mujoco.MjData(m), mujoco.MjvOption(), {}
            option.geomgroup[:] = RENDER_GROUPS
            due = time.monotonic()
            while not self.stop_event.is_set():
                with self.cond:
                    cams = list(self.cams.items())
                delay = due - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                due = max(due + 1.0 / max(c["fps"] for _, c in cams), time.monotonic())
                data.qpos[:] = self.phys.snap.qpos
                mujoco.mj_kinematics(m, data)
                mujoco.mj_comPos(m, data)
                mujoco.mj_camlight(m, data)
                stamp = time.time() * 1000.0
                for camera, cam in cams:
                    width, height = cam["size"]
                    if cam["size"] not in renderers:
                        renderers[cam["size"]] = mujoco.Renderer(m, height, width)
                    r = renderers[cam["size"]]
                    r.update_scene(data, camera=camera, scene_option=option)
                    rgb = r.render().copy()
                    r.enable_depth_rendering()
                    r.update_scene(data, camera=camera, scene_option=option)
                    z = r.render()
                    r.disable_depth_rendering()
                    depth = np.where((z > 0.1) & (z < 6.0), np.round(z * 1000.0), 0).astype(np.uint16)
                    with self.cond:
                        cam["frame"], cam["number"] = (rgb, depth, stamp), cam["number"] + 1
                        self.cond.notify_all()
        except BaseException as exc:      # surface it to the waiting pipelines
            with self.cond:
                self.error = exc
                self.cond.notify_all()
            raise

    def wait(self, camera, after, timeout):
        with self.cond:
            ok = self.cond.wait_for(lambda: self.error is not None or self.cams[camera]["number"] > after, timeout)
            if self.error is not None:
                raise RuntimeError(f"sim camera rendering failed: {self.error!r}")
            if not ok:
                raise RuntimeError("Frame didn't arrive within timeout")
            return self.cams[camera]["frame"], self.cams[camera]["number"]


def fake_realsense(phys: Physics, rig: CameraRig, serials: dict) -> types.ModuleType:
    """The slice of pyrealsense2 cameras.py uses: one pipeline per serial, fed by the camera rig."""
    rs = types.ModuleType("pyrealsense2")
    ns = types.SimpleNamespace
    rs.stream = ns(color="color", depth="depth")
    rs.format = ns(rgb8="rgb8", bgr8="bgr8", z16="z16")
    rs.camera_info = ns(serial_number="serial_number", name="name")

    class device:
        def __init__(self, serial):
            self.serial = serial

        def get_info(self, key):
            return self.serial if key == "serial_number" else "Intel RealSense D435 (sim)"

        def first_depth_sensor(self):
            return ns(get_depth_scale=lambda: 0.001)

    class context:
        def query_devices(self):
            return [device(s) for s in serials]

    class config:
        def __init__(self):
            self.serial, self.streams = None, {}

        def enable_device(self, serial):
            self.serial = serial

        def enable_stream(self, stream, width, height, fmt, fps):
            self.streams[stream] = (width, height, fmt, fps)

    class frame:
        def __init__(self, data, number, stamp):
            self.data, self.number, self.stamp = data, number, stamp

        def get_data(self):
            return self.data

        def get_frame_number(self):
            return self.number

        def get_timestamp(self):
            return self.stamp

        def __bool__(self):
            return True

    class align:
        def __init__(self, stream):
            pass

        def process(self, frames):
            return frames            # colour and depth come from the same scene camera

    class pipeline:
        def start(self, cfg):
            if cfg.serial not in serials:
                raise RuntimeError(f"No device connected with serial {cfg.serial}")
            self.camera = serials[cfg.serial]
            self.width, self.height, _, self.fps = cfg.streams["color"]
            fovy = phys.m.cam_fovy[phys.m.camera(self.camera).id]
            fy = (self.height / 2) / np.tan(np.deg2rad(fovy) / 2)
            intr = ns(width=self.width, height=self.height, fx=fy, fy=fy, ppx=self.width / 2, ppy=self.height / 2,
                      model="distortion.none (sim)", coeffs=[0.0] * 5)
            stream = ns(as_video_stream_profile=lambda: ns(get_intrinsics=lambda: intr))
            self.number = 0
            rig.add(self.camera, self.width, self.height, self.fps)
            return ns(get_device=lambda: device(cfg.serial), get_stream=lambda s: stream)

        def wait_for_frames(self, timeout_ms=5000):
            (rgb, depth, stamp), self.number = rig.wait(self.camera, self.number, timeout_ms / 1000.0)
            number = self.number
            return ns(get_color_frame=lambda: frame(rgb, number, stamp),
                      get_depth_frame=lambda: frame(depth, number, stamp))

        def stop(self):
            pass

    rs.device, rs.context, rs.config, rs.pipeline, rs.align = device, context, config, pipeline, align
    return rs


def zone_dirs():
    """Unit-depth 8x8 zone directions in the camera frame (x right, y up, looks along -z), row-major;
    the same convention as prox_learning scripts/real_skin_viz.py (no flip, planar range)."""
    f = (ZONES / 2) / np.tan(np.deg2rad(SENSOR_FOVY / 2))
    c = (ZONES - 1) / 2
    r, col = np.meshgrid(np.arange(ZONES), np.arange(ZONES), indexing="ij")
    return np.stack([(col - c) / f, -(r - c) / f, -np.ones_like(r, float)], -1).reshape(64, 3)


class SensorSim(threading.Thread):
    """Ray-casts the 41 sensors into the scene and multicasts their packets like the real boards."""

    def __init__(self, phys, rate_hz, noise_mm, drop, conf_range, seed):
        super().__init__(daemon=True)
        self.phys, self.m = phys, phys.m
        self.d = mujoco.MjData(self.m)
        self.rng = np.random.default_rng(seed)
        self.rate, self.noise, self.drop, self.conf_mm = rate_hz, noise_mm, drop, conf_range * 1000
        self.sensors = [(link, k) for link, n in SENSORS_PER_LINK.items() for k in range(n)]
        self.cams = [self.m.camera(sim_name(*s)).id for s in self.sensors]
        dirs = zone_dirs()
        self.norm = np.linalg.norm(dirs, axis=1)
        self.unit = dirs / self.norm[:, None]
        self.sent = np.zeros(len(self.sensors), int)
        self.stop_event = threading.Event()
        self.error = None
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 0)   # stay on this host
        self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_LOOP, 1)

    def packet(self, i):
        cid = self.cams[i]
        o, R = self.d.cam_xpos[cid].copy(), self.d.cam_xmat[cid].reshape(3, 3)
        vec = (self.unit @ R.T).ravel()
        geomid = np.zeros(64, np.int32)
        dist = np.zeros(64)
        mujoco.mj_multiRay(self.m, self.d, o, vec, RAY_GROUPS, 1, int(self.m.cam_bodyid[cid]),
                           geomid, dist, None, 64, MAX_RANGE_M)
        hit = (dist >= 0) & (dist <= MAX_RANGE_M)
        planar = np.where(hit, dist / self.norm, 0.0) * 1000
        if self.noise:
            planar = planar + self.rng.normal(0, self.noise, 64)
        mm = np.clip(np.round(planar), 1, NO_TARGET_MM - 1)
        b2 = np.where(hit, mm, NO_TARGET_MM).astype("<u2")
        b1 = np.where(hit & (mm < self.conf_mm), mm, NO_TARGET_MM).astype("<u2")
        return np.concatenate([b1, b2]).tobytes()

    def run(self):
        try:
            period = 1.0 / self.rate
            due = time.monotonic() + self.rng.uniform(0, period, len(self.sensors))
            while not self.stop_event.is_set():
                ready = np.flatnonzero(due <= time.monotonic())
                if len(ready):
                    self.d.qpos[:] = self.phys.snap.qpos
                    mujoco.mj_kinematics(self.m, self.d)
                    mujoco.mj_comPos(self.m, self.d)
                    mujoco.mj_camlight(self.m, self.d)
                    for i in ready:
                        due[i] += period * (1 + self.rng.normal(0, 0.03))
                        depth = self.packet(i)
                        if self.rng.random() < self.drop:
                            continue
                        link, k = self.sensors[i]
                        self.sock.sendto(bytes([k]) + depth, (GROUP, LINKS[link]))
                        self.sent[i] += 1
                time.sleep(0.002)
        except BaseException as exc:      # surface it instead of going quiet
            self.error = exc
            raise


class Demonstrator:
    """Scripted hands for teach --auto: holds the floating arm, pushes it along a pick-and-place
    path with joint springs (as a person guiding it would), and presses the keys."""

    K = np.array([300.0, 300.0, 300.0, 300.0, 120.0, 120.0, 60.0])
    D = np.array([40.0, 40.0, 40.0, 40.0, 15.0, 15.0, 6.0])

    def __init__(self, model, layout):
        lab = layout.lab
        z0, bs = lab["table_top_z"], lab["block_size"]
        bx, by = lab["block_pos"]
        cx, cy = lab["cabinet_center"]
        D, W, H = lab["cabinet_size"]
        t = lab["cabinet_wall"]
        block_z = z0 + bs / 2
        floor_z = z0 + H / 2 + t / 2 + 0.003 + 0.008             # top drawer floor, top face
        place = np.array([cx - D / 2 - lab["drawer_open"] * 0.4, cy, floor_z + bs / 2 + 0.012])
        R = np.diag([1.0, -1.0, -1.0])                           # fingers down, closing along world y (as at home)
        ik = IK(model, layout)
        home = lab_scene.HOME.copy()
        pre = ik.solve([bx, by, block_z + 0.14], R, home)
        grasp = ik.solve([bx, by, block_z + 0.002], R, pre)
        lift = ik.solve([bx, by, block_z + 0.22], R, grasp)
        # Swing over on the robot side of the cabinet: a joint-space move from the lift straight
        # to the drawer bows outwards and drags the block across the cabinet's top corner.
        via = ik.solve([cx - D / 2 - 0.15, (by + cy) / 2, place[2] + 0.2], R, lift)
        above = ik.solve(place + [0, 0, 0.15], R, via)
        down = ik.solve(place, R, above)
        # (pose, hold after reaching it [s], key pressed after the first 0.4 s of the hold)
        path = [(pre, 0.2, None), (grasp, 2.0, "g"), (lift, 0.2, None), (via, 0.0, None), (above, 0.3, None),
                (down, 1.6, "g"), (above, 0.2, None), (via, 0.0, None), (home, 0.6, None)]
        self.segments, self.presses = [], [(0.5, " ")]
        clock, q = 1.0, home
        for target, hold, key in path:
            duration = float(np.clip(np.abs(target - q).max() / 0.45, 1.2, 4.0))
            self.segments.append((clock, clock + duration, q, target))
            clock += duration
            if key:
                self.presses.append((clock + 0.4, key))
            clock += hold
            q = target
        self.presses += [(clock + 0.4, " "), (clock + 1.0, "q")]
        self.home, self.t0, self.sent = home, None, 0

    def plan(self, t):
        for start, end, q0, q1 in self.segments:
            if t < start:
                return q0, np.zeros(7)
            if t <= end:
                s = (t - start) / (end - start)
                ds = 30 * s ** 2 * (1 - s) ** 2 / (end - start)
                return q0 + (q1 - q0) * s ** 3 * (10 - 15 * s + 6 * s ** 2), (q1 - q0) * ds
        return self.segments[-1][3], np.zeros(7)

    def keys(self, now):
        if self.t0 is None:
            self.t0 = now
        due = [key for t, key in self.presses[self.sent:] if t <= now - self.t0]
        self.sent += len(due)
        return due

    def torque(self, now, q, dq):
        if self.t0 is None:
            return self.K * (self.home - q) - self.D * dq
        q_plan, dq_plan = self.plan(now - self.t0)
        return self.K * (q_plan - q) + self.D * (dq_plan - dq)


class IK:
    """Damped least squares on the fr3_hand_tcp position and the fr3_hand orientation."""

    def __init__(self, model, layout):
        self.m, self.lay = model, layout
        self.d = mujoco.MjData(model)
        self.lo = model.jnt_range[[model.joint(f"fr3_joint{i}").id for i in range(1, 8)], 0]
        self.hi = model.jnt_range[[model.joint(f"fr3_joint{i}").id for i in range(1, 8)], 1]

    def solve(self, pos, R_hand, q_start, iters=300):
        m, d, lay = self.m, self.d, self.lay
        q = np.array(q_start, float)
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for _ in range(iters):
            d.qpos[lay.arm_q] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            e_p = np.asarray(pos) - d.site_xpos[lay.tcp_site]
            Rc = d.xmat[lay.hand_body].reshape(3, 3)
            e_r = 0.5 * sum(np.cross(Rc[:, k], R_hand[:, k]) for k in range(3))
            if np.linalg.norm(e_p) < 1e-4 and np.linalg.norm(e_r) < 1e-3:
                return q
            mujoco.mj_jacSite(m, d, jp, None, lay.tcp_site)
            mujoco.mj_jacBody(m, d, None, jr, lay.hand_body)
            J = np.vstack([jp[:, lay.arm_v], jr[:, lay.arm_v]])
            JJt = J @ J.T + 1e-4 * np.eye(6)
            step = J.T @ np.linalg.solve(JJt, np.concatenate([e_p, e_r]))
            null = np.eye(7) - J.T @ np.linalg.solve(JJt, J)
            q = np.clip(q + step + null @ (0.05 * (lab_scene.HOME - q)), self.lo, self.hi)
        raise SystemExit(f"IK did not converge for {np.round(pos, 3)} (edit LAB in sim/lab_scene.py)")


class ScriptedConsole:
    """Stands in for the terminal in --auto: the demonstrator's keys, scripted prompt answers."""

    def __init__(self, twin, demonstrator=None):
        self.twin, self.demonstrator = twin, demonstrator

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def keys(self):
        if self.demonstrator is None:
            return []
        keys = self.demonstrator.keys(time.monotonic())
        for key in keys:
            print(f"[sim demonstrator] key {key!r}")
        return keys

    def ask(self, kind, question):
        answer = ""
        if kind == "reset":
            self.twin.reset_scene()
        elif kind == "result":
            answer = "y" if self.twin.block_in_drawer() else "n"
        elif kind == "save":
            answer = "y"
        print(f"{question}[sim: {answer or 'Enter'}]")
        return answer


class Twin:
    def __init__(self, args):
        self.args = args
        self.model = lab_scene.build(args.model)
        self.layout = lab_scene.Layout(self.model)
        self.phys = Physics(self.model, self.layout)
        self.sensors = SensorSim(self.phys, args.sensor_hz, args.noise_mm, args.drop, args.conf_range, args.seed)
        self.rig = CameraRig(self.phys)
        sys.modules["franky"] = fake_franky(self.phys)
        sys.modules["pyrealsense2"] = fake_realsense(self.phys, self.rig, SIM_SERIALS)

    def start(self):
        self.phys.start()
        self.sensors.start()
        if self.args.viewer:
            if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
                raise SystemExit("--viewer needs a display (DISPLAY is not set); run without it, or ssh -X")
            import mujoco.viewer

            self.phys.viewer = mujoco.viewer.launch_passive(self.model, self.phys.d)
        time.sleep(0.3)
        if self.sensors.error is not None:
            raise SystemExit(f"sensor simulation failed: {self.sensors.error!r}")

    def stop(self):
        self.sensors.stop_event.set()
        self.rig.stop_event.set()
        self.phys.stop_event.set()
        self.sensors.join(timeout=1.0)
        if self.rig.is_alive():
            self.rig.join(timeout=1.0)
        self.phys.join(timeout=1.0)
        if self.phys.viewer is not None:
            self.phys.viewer.close()
        if self.phys.lag_drops:
            print("[sim] physics fell > 0.2 s behind real time: "
                  + ", ".join(f"{lag:.2f} s at {when:.1f} s" for when, lag in self.phys.lag_drops))

    def reset_scene(self):
        self.phys.reset_scene()
        print("[sim] block back on the table, top drawer pulled out")

    def block_in_drawer(self):
        with self.phys.lock:
            ok = self.layout.block_in_drawer(self.model, self.phys.d)
        print(f"[sim] block in the drawer: {ok}")
        return ok

    def console(self, demonstrator=None):
        if self.args.auto:
            return ScriptedConsole(self, demonstrator)
        import kinesthetic

        twin = self

        class SimConsole(kinesthetic.Console):
            def ask(self, kind, question):
                if kind == "reset":
                    twin.reset_scene()
                elif kind == "result":
                    twin.block_in_drawer()
                return super().ask(kind, question)

        return SimConsole()

    def teach(self, extra):
        import kinesthetic

        demonstrator = None
        if self.args.auto:
            demonstrator = Demonstrator(self.model, self.layout)
            self.phys.human = demonstrator.torque
        out = self.args.out / "demos"
        before = set(out.glob("*.h5"))
        try:
            kinesthetic.main(["teach", "--host", "sim", "--out", str(out), *extra], self.console(demonstrator))
        finally:
            self.phys.human = None
        if demonstrator is not None:
            self.block_in_drawer()
        return sorted(set(out.glob("*.h5")) - before)

    def replay(self, demos, extra):
        import kinesthetic

        rows = self.args.out / "dataset" / "rows"
        before = set(rows.glob("*")) if rows.is_dir() else set()
        kinesthetic.main(["replay", *map(str, demos), "--host", "sim", "--out", str(self.args.out / "dataset"),
                          "--wrist-serial", "SIM-WRIST", "--exo-serial", "SIM-EXO", *extra], self.console())
        new = sorted(set(rows.glob("*")) - before) if rows.is_dir() else []
        for row in new:
            with h5py.File(row / "trajectory.h5", "a") as f:
                f.attrs["sim"] = True
                f["traj_0"].attrs["simulated"] = True
                f["traj_0"].attrs["sim_args"] = str(vars(self.args))
        return new


def main():
    argv = sys.argv[1:]
    extra = []
    if "--" in argv:
        cut = argv.index("--")
        argv, extra = argv[:cut], argv[cut + 1:]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["teach", "replay", "rehearse"])
    ap.add_argument("demos", nargs="*", type=Path, help="replay: demo files")
    ap.add_argument("--auto", action="store_true", help="scripted demonstrator and prompt answers")
    ap.add_argument("--viewer", action="store_true", help="open the MuJoCo viewer (needs a display)")
    ap.add_argument("--model", type=Path, default=lab_scene.DEFAULT_MODEL, help="FR3 + Franka Hand skin MJCF")
    ap.add_argument("--out", type=Path, default=REAL / "sim_out", help="demos/ and dataset/ go here")
    ap.add_argument("--sensor-hz", type=float, default=15.0, help="packet rate of each simulated sensor")
    ap.add_argument("--noise-mm", type=float, default=0.0, help="gaussian range noise per zone")
    ap.add_argument("--drop", type=float, default=0.0, help="fraction of sensor packets lost")
    ap.add_argument("--conf-range", type=float, default=2.0, help="m; farther hits go to block 2 only")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    if args.command == "rehearse":
        args.auto = True
    if args.command == "replay" and not args.demos:
        ap.error("replay needs demo files")

    twin = Twin(args)
    twin.start()
    try:
        if args.command in ("teach", "rehearse"):
            demos = twin.teach(extra if args.command == "teach" else [])
            print("[sim] demos: " + (", ".join(map(str, demos)) or "none saved"))
            if args.command == "rehearse":
                if not demos:
                    raise SystemExit("teach saved no demo")
                time.sleep(0.5)
                rows = twin.replay(demos, extra)
                print("[sim] rows: " + (", ".join(map(str, rows)) or "none saved"))
        else:
            rows = twin.replay(args.demos, extra)
            print("[sim] rows: " + (", ".join(map(str, rows)) or "none saved"))
    finally:
        twin.stop()


if __name__ == "__main__":
    main()
