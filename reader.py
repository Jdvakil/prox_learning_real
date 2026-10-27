import socket
import struct
import select
from collections import deque

import numpy as np
import json

GROUP = '239.0.0.1'
OUT_PATH = 'steps.json'
LINKS = {
    "link1": 5004,
    "link2": 5005,
    "link3": 5006,
    "link5_small": 5007,
    "link5_large": 5008,
    "link6": 5009,
    "link4": 5010,
    "end_effector": 5011,
}

membership = struct.pack('4sL', socket.inet_aton(GROUP), socket.INADDR_ANY)
sockets = {}
for name, port in LINKS.items():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('0.0.0.0', port))
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, membership)
    sockets[sock] = name

buffers = {name: deque() for name in LINKS}
step = 0

with open(OUT_PATH, 'a', encoding='utf-8') as out:
    while True:
        readable, _, _ = select.select(list(sockets), [], [])
        for sock in readable:
            data, addr = sock.recvfrom(4096)
            name = sockets[sock]
            depths = np.frombuffer(data[1:], dtype=np.uint16)
            buffers[name].append({
                "address": addr[0],
                "sensor": data[0],
                "len": len(data),
                "depth_mm": depths.tolist(),
            })

        if all(buffers[name] for name in LINKS):
            record = {
                "step": step,
                "sensors": {name: buffers[name].popleft() for name in LINKS},
            }
            json.dump(record, out)
            out.write('\n')
            out.flush()
            step += 1
