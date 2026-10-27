"""Play proximity readings from an HDF5 log as a point cloud on the moving arm.

Each VL53L5CX sensor is an 8x8 depth image. Rays leave the sensor frame along +Z,
which is the boresight in fr3_full_skin.urdf, across a 45 degree square field of
view (65 degree diagonal). Distances of 0 or 4000 mm are misses and are dropped.

The URDF places sensors on links 2, 3, 5, and 6 only. Streams in the log that
have no matching frame are skipped.
"""

import argparse
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent
DEFAULT_H5 = ROOT / "recordings" / "20261027T223217Z.h5"
DEFAULT_URDF = ROOT / "fr3_full_skin.urdf"

# UDP stream name -> URDF sensor prefix (link2_sensor_0, ...).
STREAM_TO_LINK = {
    "link2": "link2",
    "link3": "link3",
    "link5_large": "link5",
    "link6": "link6",
}

ZONES = 8
FOV_RAD = math.radians(45.0)
NO_TARGET_MM = 4000
ARM_LINKS = [f"fr3_link{i}" for i in range(8)]


def rpy_matrix(roll, pitch, yaw):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rotation_z = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=float)
    rotation_y = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=float)
    rotation_x = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
    return rotation_z @ rotation_y @ rotation_x


def origin_matrix(element):
    matrix = np.eye(4)
    if element is None:
        return matrix
    matrix[:3, :3] = rpy_matrix(*[float(value) for value in element.get("rpy", "0 0 0").split()])
    matrix[:3, 3] = [float(value) for value in element.get("xyz", "0 0 0").split()]
    return matrix


def rotation_about_z(angle):
    cosine, sine = math.cos(angle), math.sin(angle)
    matrix = np.eye(4)
    matrix[:3, :3] = [[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]]
    return matrix


def load_urdf(path):
    root = ET.parse(path).getroot()
    joints = []
    for joint in root.findall("joint"):
        joints.append({
            "name": joint.get("name"),
            "type": joint.get("type"),
            "parent": joint.find("parent").get("link"),
            "child": joint.find("child").get("link"),
            "origin": origin_matrix(joint.find("origin")),
        })
    return joints


def link_poses(joints, joint_position):
    poses = {"base": np.eye(4)}
    angles = {f"fr3_joint{index + 1}": float(joint_position[index]) for index in range(7)}
    pending = list(joints)
    for _ in range(len(joints) + 1):
        remaining = []
        for joint in pending:
            if joint["parent"] not in poses:
                remaining.append(joint)
                continue
            pose = poses[joint["parent"]] @ joint["origin"]
            if joint["type"] == "revolute":
                pose = pose @ rotation_about_z(angles[joint["name"]])
            poses[joint["child"]] = pose
        pending = remaining
        if not pending:
            break
    return poses


def zone_directions():
    """Unit rays for an 8x8 image. Row 0 is the top, column 0 is the left."""
    centers = (np.arange(ZONES) + 0.5) / ZONES * FOV_RAD - FOV_RAD / 2
    rays = np.zeros((ZONES, ZONES, 3))
    for row, y_angle in enumerate(centers):
        for column, x_angle in enumerate(centers):
            direction = np.array([math.tan(x_angle), math.tan(y_angle), 1.0])
            rays[row, column] = direction / np.linalg.norm(direction)
    return rays


RAYS = zone_directions()


def eight_by_eight(depth_mm):
    """Closer valid target for each zone, in millimeters. None is a miss."""
    depth = np.asarray(depth_mm, dtype=float).reshape(-1)
    image = np.full((ZONES, ZONES), np.nan)
    for index in range(ZONES * ZONES):
        candidates = [depth[index]]
        second = ZONES * ZONES + index
        if second < depth.size:
            candidates.append(depth[second])
        hits = [value for value in candidates if 0 < value < NO_TARGET_MM]
        if hits:
            row, column = divmod(index, ZONES)
            image[row, column] = min(hits)
    return image


def load_recording(path):
    steps = []
    with h5py.File(path, "r") as handle:
        joint_position = np.asarray(handle["robot/q"])
        for _ in range(joint_position.shape[0]):
            steps.append({})
        for stream, link_name in STREAM_TO_LINK.items():
            group = handle["sensors"][stream]
            depth = np.asarray(group["depth_mm"])
            sensor_id = np.asarray(group["sensor_id"])
            if depth.ndim == 2:
                packets = [(step, int(sensor_id[step]), depth[step]) for step in range(depth.shape[0])]
            else:
                ids = sensor_id if sensor_id.ndim == 1 else None
                packets = []
                for step in range(depth.shape[0]):
                    for sensor_index in range(depth.shape[1]):
                        sid = int(ids[sensor_index] if ids is not None else sensor_id[step, sensor_index])
                        packets.append((step, sid, depth[step, sensor_index]))
            for step, sid, zones in packets:
                steps[step].setdefault(link_name, []).append((sid, zones))
        skipped = [name for name in handle["sensors"] if name not in STREAM_TO_LINK]
    return joint_position, steps, skipped


def cloud_for_step(poses, packets, max_range_m):
    points = []
    distances = []
    for link_name, readings in packets.items():
        for sensor_id, depth in readings:
            frame = f"{link_name}_sensor_{sensor_id}"
            if frame not in poses:
                continue
            pose = poses[frame]
            image = eight_by_eight(depth)
            for row in range(ZONES):
                for column in range(ZONES):
                    distance_mm = image[row, column]
                    if not np.isfinite(distance_mm) or distance_mm / 1000.0 > max_range_m:
                        continue
                    local = RAYS[row, column] * (distance_mm / 1000.0)
                    world = pose[:3, :3] @ local + pose[:3, 3]
                    points.append(world)
                    distances.append(distance_mm / 1000.0)
    if not points:
        return np.zeros((0, 3)), np.zeros(0)
    return np.vstack(points), np.asarray(distances)


def arm_polyline(poses):
    return np.vstack([poses[name][:3, 3] for name in ARM_LINKS if name in poses])


def show(joint_position, steps, joints, save_path=None, step_index=None, max_range_m=1.5):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    frames = []
    for index, packets in enumerate(steps):
        poses = link_poses(joints, joint_position[index])
        points, distances = cloud_for_step(poses, packets, max_range_m)
        frames.append((poses, points, distances))
        print(f"step {index}: {len(points)} points")

    cloud = np.vstack([frame[1] for frame in frames if len(frame[1])])
    arm = np.vstack([arm_polyline(frame[0]) for frame in frames])
    both = np.vstack([cloud, arm])
    center = both.mean(axis=0)
    span = np.max(np.linalg.norm(both - center, axis=1)) * 1.15

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    def draw(index):
        ax.cla()
        poses, points, distances = frames[index]
        arm_points = arm_polyline(poses)
        ax.plot(arm_points[:, 0], arm_points[:, 1], arm_points[:, 2], color="black", linewidth=2.5, marker="o", markersize=4)
        if len(points):
            ax.scatter(
                points[:, 0], points[:, 1], points[:, 2],
                c=distances, cmap="turbo", s=12, vmin=0.0, vmax=max_range_m,
            )
        ax.set_xlim(center[0] - span, center[0] + span)
        ax.set_ylim(center[1] - span, center[1] + span)
        ax.set_zlim(center[2] - span, center[2] + span)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
        ax.set_title(f"step {index}  q={np.round(joint_position[index], 2).tolist()}")
        ax.set_box_aspect((1, 1, 1))

    if save_path is not None or step_index is not None:
        draw(0 if step_index is None else step_index)
        if save_path is not None:
            fig.savefig(save_path, dpi=120)
            print(f"saved {save_path}")
        if step_index is not None and save_path is None:
            plt.show()
        return

    draw(0)
    animation = FuncAnimation(fig, draw, frames=len(frames), interval=1200, repeat=True)
    plt.show()
    return animation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", type=Path, default=DEFAULT_H5)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--step", type=int, default=None, help="show one step instead of playing all of them")
    parser.add_argument("--save", type=Path, default=None, help="write a PNG of one step and exit")
    parser.add_argument("--max-range", type=float, default=1.5, help="drop returns farther than this, in meters")
    args = parser.parse_args()

    joints = load_urdf(args.urdf)
    joint_position, steps, skipped = load_recording(args.h5)
    if skipped:
        print("no URDF frame for: " + ", ".join(skipped))
    show(
        joint_position,
        steps,
        joints,
        save_path=args.save,
        step_index=args.step,
        max_range_m=args.max_range,
    )


if __name__ == "__main__":
    main()
