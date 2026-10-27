
import socket
import struct
import numpy as np

PORT = 5006
GROUP = '239.0.0.1'

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind(('0.0.0.0', PORT))

sock.setsockopt(
    socket.IPPROTO_IP,
    socket.IP_ADD_MEMBERSHIP,
    struct.pack('4sL', socket.inet_aton(GROUP), socket.INADDR_ANY)
)

while True:
    data, addr = sock.recvfrom(4096)
    depths = np.frombuffer(data[1:], dtype=np.uint16)

    print(
        f"{addr[0]} "
        f"sensor={data[0]} "
        f"len={len(data)} "
        f"depth_mm={depths}"
    )
