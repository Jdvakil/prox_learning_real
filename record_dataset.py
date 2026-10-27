"""Move the Franka through joint positions and save each stopped pose with all proximity sensors.

Each row in the HDF5 file is one step: the joint target, the measured robot state after
the motion finishes, and one packet from each of the 41 proximity sensors.

Requires franky (built from submodules/franky) and h5py.
"""

import argparse
import select
import socket
import struct
import time
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

from sensor_layout import LINKS, SENSORS_PER_LINK

GROUP = "239.0.0.1"

# Joint targets [rad] from the franky waypoint example. Edit these for the workspace.
POSITIONS = np.array(
    [
        [0.0, 0.0, 0.0, -2.2, 0.0, 2.2, 0.7],
        [0.2, -0.1, 0.3, -2.0, 0.1, 1.9, 0.6],
        [0.4, -0.3, 0.1, -2.3, 0.0, 2.2, 0.5],
        [0.0, 0.0, 0.0, -2.2, 0.0, 2.2, 0.7],
    ],
    dtype=np.float64,
)


def open_sensor_sockets():
    membership = struct.pack("4sL", socket.inet_aton(GROUP), socket.INADDR_ANY)
    sockets = {}
    for name, port in LINKS.items():
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("0.0.0.0", port))
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, membership)
        sockets[sock] = name
    return sockets


def close_sensor_sockets(sockets):
    for sock in sockets:
        sock.close()


def parse_packet(data, addr):
    if len(data) < 1:
        raise ValueError("empty sensor packet")
    depths = np.frombuffer(data[1:], dtype=np.uint16).copy()
    return {
        "address": addr[0],
        "sensor": int(data[0]),
        "len": len(data),
        "depth_mm": depths,
    }


def drain_sensors(sockets):
    for sock in sockets:
        sock.setblocking(False)
        try:
            while True:
                sock.recvfrom(65535)
        except BlockingIOError:
            pass
        finally:
            sock.setblocking(True)


def read_sensor_snapshot(sockets, timeout):
    readings = {name: {} for name in LINKS}
    deadline = time.monotonic() + timeout

    def missing_sensors():
        missing = []
        for name, count in SENSORS_PER_LINK.items():
            for sensor_id in range(count):
                if sensor_id not in readings[name]:
                    missing.append(f"{name} sensor {sensor_id}")
        return missing

    while missing_sensors():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("timed out waiting for sensors: " + ", ".join(missing_sensors()))
        readable, _, _ = select.select(list(sockets), [], [], remaining)
        for sock in readable:
            data, addr = sock.recvfrom(65535)
            packet = parse_packet(data, addr)
            readings[sockets[sock]][packet["sensor"]] = packet
    return {
        name: [readings[name][sensor_id] for sensor_id in sorted(readings[name])]
        for name in LINKS
    }


def elbow_vector(elbow):
    flip = elbow.joint_4_flip
    flip_value = np.nan if flip is None else float(int(flip))
    return np.array([float(elbow.joint_3_pos), flip_value], dtype=np.float64)


def read_robot_state(robot):
    joints = robot.current_joint_state
    pose = robot.current_cartesian_state.pose.end_effector_pose
    return {
        "q": np.asarray(joints.position, dtype=np.float64).reshape(7),
        "dq": np.asarray(joints.velocity, dtype=np.float64).reshape(7),
        "O_T_EE": np.asarray(pose.matrix, dtype=np.float64).reshape(4, 4),
        "translation": np.asarray(pose.translation, dtype=np.float64).reshape(3),
        "quaternion_xyzw": np.asarray(pose.quaternion, dtype=np.float64).reshape(4),
        "elbow": elbow_vector(robot.state.elbow),
    }


def save_h5(path, records, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        for key, value in metadata.items():
            handle.attrs[key] = value
        handle.attrs["links"] = np.array(list(LINKS), dtype=h5py.string_dtype(encoding="utf-8"))
        handle.create_dataset("step", data=np.array([record["step"] for record in records], dtype=np.int32))
        handle.create_dataset("timestamp", data=np.array([record["timestamp"] for record in records], dtype=np.float64))
        handle.create_dataset("target_q", data=np.stack([record["target_q"] for record in records]))
        handle["target_q"].attrs["unit"] = "rad"

        robot = handle.create_group("robot")
        for key, unit in (
            ("q", "rad"),
            ("dq", "rad/s"),
            ("translation", "m"),
            ("quaternion_xyzw", "xyzw"),
            ("elbow", "joint 3 rad, joint 4 flip (-1, 0, 1)"),
        ):
            robot.create_dataset(key, data=np.stack([record[key] for record in records]))
            robot[key].attrs["unit"] = unit
        robot.create_dataset("O_T_EE", data=np.stack([record["O_T_EE"] for record in records]))
        robot["O_T_EE"].attrs["layout"] = "4x4 homogeneous, translation in the last column"

        sensors = handle.create_group("sensors")
        for name, port in LINKS.items():
            group = sensors.create_group(name)
            group.attrs["port"] = port
            per_step = [record["sensors"][name] for record in records]
            sensor_ids = [packet["sensor"] for packet in per_step[0]]
            depth = np.stack([
                np.stack([np.asarray(packet["depth_mm"], dtype=np.uint16) for packet in step_packets])
                for step_packets in per_step
            ])
            group.create_dataset("sensor_id", data=np.array(sensor_ids, dtype=np.uint8))
            group.create_dataset("depth_mm", data=depth, compression="gzip")
            group["depth_mm"].attrs["unit"] = "mm"
            group["depth_mm"].attrs["layout"] = "(step, sensor, zone)"
            group.create_dataset(
                "packet_len",
                data=np.array([[packet["len"] for packet in step_packets] for step_packets in per_step], dtype=np.int32),
            )
            group.create_dataset(
                "address",
                data=np.array(
                    [[packet["address"] for packet in step_packets] for step_packets in per_step],
                    dtype=h5py.string_dtype(encoding="utf-8"),
                ),
            )
    return path


def collect(host, dynamics, settle, sensor_timeout, output):
    from franky import JointMotion, Robot

    sockets = open_sensor_sockets()
    records = []
    try:
        robot = Robot(host)
        robot.recover_from_errors()
        robot.relative_dynamics_factor = dynamics

        for step, target in enumerate(POSITIONS):
            print(f"step {step}: moving to {target.tolist()}")
            robot.move(JointMotion(target.tolist()))
            time.sleep(settle)
            drain_sensors(sockets)
            sensors = read_sensor_snapshot(sockets, sensor_timeout)
            record = read_robot_state(robot)
            record["step"] = step
            record["timestamp"] = time.time()
            record["target_q"] = np.asarray(target, dtype=np.float64)
            record["sensors"] = sensors
            records.append(record)
            counted = sum(len(sensors[name]) for name in LINKS)
            print(f"step {step}: recorded q={record['q'].round(4).tolist()} sensors {counted}")
    finally:
        close_sensor_sockets(sockets)
        if records:
            metadata = {
                "host": host,
                "relative_dynamics_factor": dynamics,
                "multicast_group": GROUP,
                "created_utc": datetime.now(timezone.utc).isoformat(),
            }
            saved = save_h5(output, records, metadata)
            print(f"saved {len(records)} steps to {saved}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="172.16.0.2", help="FCI IP of the robot")
    parser.add_argument("--dynamics", type=float, default=0.05, help="relative velocity, acceleration, and jerk")
    parser.add_argument("--settle", type=float, default=0.5, help="seconds to wait after each motion before sensing")
    parser.add_argument("--sensor-timeout", type=float, default=5.0, help="seconds to wait for every link")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="HDF5 output path (default: recordings/<utc timestamp>.h5)",
    )
    args = parser.parse_args()
    output = args.output
    if output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = Path("recordings") / f"{stamp}.h5"
    collect(args.host, args.dynamics, args.settle, args.sensor_timeout, output)


if __name__ == "__main__":
    main()
