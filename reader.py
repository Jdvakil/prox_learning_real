import socket
import time
import json
def read_data(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect(('localhost', port))
        data = s.recv(1024)
        return json.loads(data)

if __name__ == "__main__":
    while True:
        for port in [5004, 5005, 5006, 5007, 5008, 5009, 5010, 5011]:
            data = read_data(port)
            print(f"Port {port}: {data}")
            time.sleep(1)