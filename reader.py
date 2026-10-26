import socket
import time
def read_data(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect(('localhost', port))
        data = s.recv(1024)
        return data

if __name__ == "__main__":
    while True:
        data = read_data(5004)
        print(data)
        time.sleep(1)