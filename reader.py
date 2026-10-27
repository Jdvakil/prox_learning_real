import socket
import struct
import select
from collections import deque

import numpy as np
import json

from sensor_layout import LINKS, SENSORS_PER_LINK

GROUP = "239.0.0.1"
OUT_PATH = "steps.json"

membership = struct.pack("4sL", socket.inet_aton(GROUP), socket.INADDR_ANY)
sockets = {}
for name, port in LINKS.items():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", port))
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, membership)
    sockets[sock] = name

buffers = {name: {sensor_id: deque() for sensor_id in range(count)} for name, count in SENSORS_PER_LINK.items()}
step = 0


def full_step():
    return all(buffers[name][sensor_id] for name, count in SENSORS_PER_LINK.items() for sensor_id in range(count))


with open(OUT_PATH, "a", encoding="utf-8") as out:
    while True:
        readable, _, _ = select.select(list(sockets), [], [])
        for sock in readable:
            data, addr = sock.recvfrom(4096)
            name = sockets[sock]
            sensor_id = int(data[0])
            depths = np.frombuffer(data[1:], dtype=np.uint16)
            buffers[name].setdefault(sensor_id, deque()).append({
                "address": addr[0],
                "sensor": sensor_id,
                "len": len(data),
                "depth_mm": depths.tolist(),
            })

        if full_step():
            sensors = {}
            for name in LINKS:
                ids = list(range(SENSORS_PER_LINK[name]))
                ids.extend(sensor_id for sensor_id in sorted(buffers[name]) if sensor_id not in ids)
                sensors[name] = [buffers[name][sensor_id].popleft() for sensor_id in ids if buffers[name][sensor_id]]
            record = {"step": step, "sensors": sensors}
            json.dump(record, out)
            out.write("\n")
            out.flush()
            step += 1
