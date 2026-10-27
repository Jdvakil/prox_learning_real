import socket
import struct
import numpy as np
import json

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

count = 0
datum = {}
while True:
    data, addr = sock.recvfrom(4096)

    depths = np.frombuffer(data[1:], dtype=np.uint16)
    datum[count]= {
        "address"   :  addr[0],
        "sensor"    :  data[0],
        "len"       :  len(data),
        "depth_mm"  :  depths.tolist(),
    }
    count += 1
    json.dump(datum, "link3.json", indent=4)
