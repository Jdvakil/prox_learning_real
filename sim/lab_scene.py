"""MuJoCo copy of the lab bench, for rehearsing kinesthetic.py without the robot.

The FR3 + Franka Hand skin model (prox_learning assets/robots/franka_skin/
model_hybrid_franka_hand.xml, skin colours unchanged) stands on its rails on the wooden table,
next to a two-drawer cabinet with the top drawer pulled out and a block to pick. Two D435s:
one on the hand (wrist_camera) and one on a tripod (exo_camera_1).

Every measurement is in LAB, in the robot base frame (x forward, z up, metres). The defaults
are read off the bench photos and the 2026-10-27 recording (the hand sensor sees the work
surface at z ~ 0); tape-measure the cabinet, block and tripod and edit LAB to match.
"""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

REAL = Path(__file__).resolve().parents[1]
# prox_learning_real is checked out at prox_learning/submodules/prox_learning_real.
DEFAULT_MODEL = REAL.parents[1] / "assets/robots/franka_skin/model_hybrid_franka_hand.xml"

LAB = {
    "table_top_z": -0.045,                       # wood top; the base stands on 45 mm rails
    "table_x": (-0.40, 1.10),
    "table_y": (-0.60, 0.70),
    "table_height": 0.90,                        # floor to wood top
    "rail_x": (-0.30, 0.30),
    "rail_y": (-0.0675, -0.0225, 0.0225, 0.0675),
    "rail_size": 0.045,                          # square extrusion
    "cabinet_center": (0.66, 0.34),              # x, y of the box centre on the table
    "cabinet_size": (0.30, 0.32, 0.26),          # depth (x), width (y), height (z); front faces the robot
    "cabinet_wall": 0.012,
    "drawer_open": 0.16,                         # top drawer pulled out towards the robot
    "drawer_travel": 0.22,
    "block_pos": (0.45, -0.18),                  # x, y on the table
    "block_size": 0.05,                          # cube edge
    "block_mass": 0.06,
    # D435 colour stream at 640x480: fx = fy ~ 615 px -> 42.5 deg vertical field of view.
    "d435_fovy": 42.5,
    "d435_resolution": (640, 480),
    # Wrist D435 in the fr3_hand frame (z along the fingers, fingers close along y): on the -x
    # side of the hand, opposite the hand ToF sensor, tilted towards the fingertips.
    "wrist_camera_pos": (-0.075, 0.0, 0.03),
    "wrist_camera_tilt_deg": 35.0,
    "exo_camera_pos": (1.15, -0.85, 0.85),
    "exo_camera_target": (0.4, 0.05, 0.18),
}

HOME = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
ROBOT_ROOT = "fr3_link0"

WOOD = [0.76, 0.6, 0.42, 1]
ALUMINIUM = [0.74, 0.75, 0.77, 1]
CABINET = [0.07, 0.07, 0.08, 1]
DRAWER = [0.12, 0.12, 0.13, 1]
BLOCK = [0.15, 0.4, 0.9, 1]
D435 = [0.18, 0.18, 0.2, 1]


def look_quat(forward, up):
    """wxyz quat of a MuJoCo camera (looks along -z, image up = +y) facing `forward`."""
    f = np.asarray(forward, float)
    f /= np.linalg.norm(f)
    x = np.cross(f, up)
    x /= np.linalg.norm(x)
    y = np.cross(x, f)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.column_stack([x, y, -f]).ravel())
    return quat


def _box(body, name, center, half, rgba, collide=True, **kwargs):
    g = body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, pos=list(center), size=list(half),
                      rgba=list(rgba), **kwargs)
    g.group = 0
    g.contype = g.conaffinity = 1 if collide else 0
    return g


def _robot_bodies(spec):
    root = spec.body(ROBOT_ROOT)
    out, stack = [], [root]
    while stack:
        body = stack.pop()
        out.append(body)
        stack.extend(body.bodies)
    return out


def build(model_path: Path = DEFAULT_MODEL, lab: dict = LAB) -> mujoco.MjModel:
    if not Path(model_path).is_file():
        raise SystemExit(f"skin model not found: {model_path}\n"
                         "build it in prox_learning with python scripts/build_franka_hand_ee_skin.py, "
                         "or pass --model")
    spec = mujoco.MjSpec.from_file(str(model_path))
    spec.option.timestep = 0.002
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10

    # Gravity compensation on every robot body, as the Franka controller does internally.
    # Skin geoms go to group 5: the sensor rays skip them (a sensor sits in its own shell),
    # the D435 renders show them.
    for body in _robot_bodies(spec):
        body.gravcomp = 1
        if body.name.endswith("_skin"):
            for geom in body.geoms:
                geom.group = 5
    # Closed on nothing, the two finger hulls interlock and never let go; the real fingers just touch.
    spec.add_exclude(bodyname1="fr3_leftfinger", bodyname2="fr3_rightfinger")

    world = spec.worldbody
    z0 = lab["table_top_z"]
    world.add_light(name="lab_key", pos=[0.4, -0.3, 2.4], dir=[0, 0.1, -1], diffuse=[0.75, 0.75, 0.72],
                    specular=[0.2, 0.2, 0.2], castshadow=True)
    world.add_light(name="lab_fill", pos=[1.6, 1.0, 1.8], dir=[-0.6, -0.4, -0.7], diffuse=[0.35, 0.35, 0.38],
                    specular=[0, 0, 0], castshadow=False)
    floor = world.add_geom(name="lab_floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[4, 4, 0.1],
                           pos=[0, 0, z0 - lab["table_height"]], rgba=[0.35, 0.33, 0.32, 1])
    floor.group = 0

    (x0, x1), (y0, y1) = lab["table_x"], lab["table_y"]
    _box(world, "lab_table", [(x0 + x1) / 2, (y0 + y1) / 2, z0 - 0.02], [(x1 - x0) / 2, (y1 - y0) / 2, 0.02], WOOD)
    leg_h = (lab["table_height"] - 0.04) / 2
    for i, (lx, ly) in enumerate([(x0 + 0.05, y0 + 0.05), (x0 + 0.05, y1 - 0.05), (x1 - 0.05, y0 + 0.05), (x1 - 0.05, y1 - 0.05)]):
        _box(world, f"lab_leg{i}", [lx, ly, z0 - 0.04 - leg_h], [0.03, 0.03, leg_h], ALUMINIUM, collide=False)
    s = lab["rail_size"]
    (rx0, rx1) = lab["rail_x"]
    for i, ry in enumerate(lab["rail_y"]):
        _box(world, f"lab_rail{i}", [(rx0 + rx1) / 2, ry, z0 + s / 2], [(rx1 - rx0) / 2, s / 2 - 0.0005, s / 2], ALUMINIUM)

    # Cabinet: open front facing the robot (-x), a divider between the two drawers.
    cx, cy = lab["cabinet_center"]
    D, W, H = lab["cabinet_size"]
    t = lab["cabinet_wall"]
    cab = world.add_body(name="cabinet", pos=[cx, cy, z0])
    _box(cab, "cabinet_bottom", [0, 0, t / 2], [D / 2, W / 2, t / 2], CABINET)
    _box(cab, "cabinet_top", [0, 0, H - t / 2], [D / 2, W / 2, t / 2], CABINET)
    _box(cab, "cabinet_back", [D / 2 - t / 2, 0, H / 2], [t / 2, W / 2, H / 2], CABINET)
    _box(cab, "cabinet_left", [0, W / 2 - t / 2, H / 2], [D / 2, t / 2, H / 2], CABINET)
    _box(cab, "cabinet_right", [0, -W / 2 + t / 2, H / 2], [D / 2, t / 2, H / 2], CABINET)
    _box(cab, "cabinet_divider", [0, 0, H / 2], [D / 2 - t, W / 2 - t, t / 2], CABINET)

    # Drawers. Each body origin sits at the front plane, centred on its opening.
    gap = 0.003
    dw = W / 2 - t - gap                         # half width of a drawer box
    dd = D - t - 2 * gap                         # drawer box depth
    dh = H / 2 - t - t / 2 - 2 * gap             # drawer box height
    wall = 0.008
    for name, zb in (("drawer", H / 2 + t / 2 + gap), ("drawer_bottom", t + gap)):
        drawer = cab.add_body(name=name, pos=[-D / 2, 0, zb])
        if name == "drawer":
            joint = drawer.add_joint(name="drawer_slide", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[-1, 0, 0],
                                     range=[0, lab["drawer_travel"]], damping=4.0, frictionloss=0.3)
            joint.limited = mujoco.mjtLimited.mjLIMITED_TRUE
        _box(drawer, f"{name}_floor", [dd / 2, 0, wall / 2], [dd / 2, dw, wall / 2], DRAWER, mass=0.25)
        _box(drawer, f"{name}_rear", [dd - wall / 2, 0, dh / 2], [wall / 2, dw, dh / 2], DRAWER, mass=0.05)
        _box(drawer, f"{name}_left", [dd / 2, dw - wall / 2, dh / 2], [dd / 2, wall / 2, dh / 2], DRAWER, mass=0.05)
        _box(drawer, f"{name}_right", [dd / 2, -dw + wall / 2, dh / 2], [dd / 2, wall / 2, dh / 2], DRAWER, mass=0.05)
        _box(drawer, f"{name}_front", [-0.0075, 0, dh / 2 + gap], [0.0075, W / 2 - 0.002, dh / 2 + gap], CABINET, mass=0.1)
        _box(drawer, f"{name}_handle", [-0.035, 0, dh / 2 + gap], [0.008, 0.06, 0.008], ALUMINIUM, mass=0.02)
        for k, hy in enumerate((-0.05, 0.05)):
            _box(drawer, f"{name}_post{k}", [-0.022, hy, dh / 2 + gap], [0.0125, 0.006, 0.006], ALUMINIUM, mass=0.005)

    bs = lab["block_size"]
    bx, by = lab["block_pos"]
    block = world.add_body(name="block", pos=[bx, by, z0 + bs / 2])
    block.add_freejoint(name="block_free")
    g = _box(block, "block_geom", [0, 0, 0], [bs / 2] * 3, BLOCK, mass=lab["block_mass"])
    g.condim = 4
    g.friction = [1.0, 0.01, 0.0001]

    # Wrist D435 (visual only; the arm sensors can see it, as they will on the bench).
    hand = spec.body("fr3_hand")
    tilt = np.deg2rad(lab["wrist_camera_tilt_deg"])
    forward = np.array([np.sin(tilt), 0.0, np.cos(tilt)])          # towards the fingertips
    wpos = np.asarray(lab["wrist_camera_pos"], float)
    wcam = hand.add_body(name="wrist_d435", pos=list(wpos), quat=list(look_quat(forward, [-1.0, 0.0, 0.0])))
    wcam.gravcomp = 1
    shell = wcam.add_geom(name="wrist_d435_shell", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.045, 0.0125, 0.0125],
                          pos=[0, 0, 0.0125], rgba=D435, mass=0.072)
    shell.group, shell.contype, shell.conaffinity = 2, 0, 0
    wcam.add_camera(name="wrist_camera", fovy=lab["d435_fovy"], resolution=list(lab["d435_resolution"]))

    epos = np.asarray(lab["exo_camera_pos"], float)
    eq = look_quat(np.asarray(lab["exo_camera_target"]) - epos, [0.0, 0.0, 1.0])
    exo = world.add_body(name="exo_d435", pos=list(epos), quat=list(eq))
    shell = exo.add_geom(name="exo_d435_shell", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.045, 0.0125, 0.0125],
                         pos=[0, 0, 0.0125], rgba=D435)
    shell.group, shell.contype, shell.conaffinity = 0, 0, 0
    exo.add_camera(name="exo_camera_1", fovy=lab["d435_fovy"], resolution=list(lab["d435_resolution"]))
    stand_h = epos[2] - (z0 - lab["table_height"])
    stand = world.add_geom(name="exo_tripod", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.012, stand_h / 2 - 0.03, 0],
                           pos=[epos[0], epos[1], epos[2] - 0.03 - (stand_h / 2 - 0.03)], rgba=[0.1, 0.1, 0.1, 1])
    stand.group, stand.contype, stand.conaffinity = 0, 0, 0

    model = spec.compile()
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, lab["d435_resolution"][0])
    model.vis.global_.offheight = max(model.vis.global_.offheight, lab["d435_resolution"][1])
    return model


class Layout:
    """qpos / qvel addresses of the parts the twin moves or checks."""

    def __init__(self, model: mujoco.MjModel, lab: dict = LAB):
        self.lab = lab
        joint = lambda name: model.joint(name)
        self.arm_q = np.array([model.jnt_qposadr[joint(f"fr3_joint{i}").id] for i in range(1, 8)])
        self.arm_v = np.array([model.jnt_dofadr[joint(f"fr3_joint{i}").id] for i in range(1, 8)])
        self.finger_q = np.array([model.jnt_qposadr[joint(f"fr3_finger_joint{i}").id] for i in (1, 2)])
        self.finger_v = np.array([model.jnt_dofadr[joint(f"fr3_finger_joint{i}").id] for i in (1, 2)])
        self.block_q = model.jnt_qposadr[joint("block_free").id]
        self.block_v = model.jnt_dofadr[joint("block_free").id]
        self.drawer_q = model.jnt_qposadr[joint("drawer_slide").id]
        self.drawer_v = model.jnt_dofadr[joint("drawer_slide").id]
        self.arm_act = np.array([model.actuator(f"fr3_joint{i}").id for i in range(1, 8)])
        self.grip_act = model.actuator("gripper/fingers_actuator").id
        self.tcp_site = model.site("fr3_hand_tcp").id
        self.hand_body = model.body("fr3_hand").id
        self.block_body = model.body("block").id
        self.drawer_body = model.body("drawer").id

    def reset_scene(self, data: mujoco.MjData):
        """Block back on the table, top drawer pulled out to LAB['drawer_open']."""
        bs = self.lab["block_size"]
        bx, by = self.lab["block_pos"]
        data.qpos[self.block_q:self.block_q + 7] = [bx, by, self.lab["table_top_z"] + bs / 2, 1, 0, 0, 0]
        data.qvel[self.block_v:self.block_v + 6] = 0
        data.qpos[self.drawer_q] = self.lab["drawer_open"]
        data.qvel[self.drawer_v] = 0

    def block_in_drawer(self, model: mujoco.MjModel, data: mujoco.MjData) -> bool:
        """Block centre inside the top drawer box and resting low in it."""
        local = data.xmat[self.drawer_body].reshape(3, 3).T @ (data.xpos[self.block_body] - data.xpos[self.drawer_body])
        D, W, H = self.lab["cabinet_size"]
        t = self.lab["cabinet_wall"]
        return bool(0 < local[0] < D - t and abs(local[1]) < W / 2 - t and local[2] < self.lab["block_size"])
