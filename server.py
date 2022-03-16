# echo-server.py

import socket
import json
import subprocess

HOST = "127.0.0.1"  # Standard loopback interface address (localhost)
PORT = 65432  # Port to listen on (non-privileged ports are > 1023)


def check_data(data):
    for field in ["dataset", "config", "split", "feature", "email"]:
        if field not in data:
            return False

        if not isinstance(data[field], str):
            return False

    if "label_field" in data:
        if not isinstance(data["label_field"], str):
            return False
    return True

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    while True:
        conn, addr = s.accept()
        print(f"Connected by {addr}")
        data = json.loads(conn.recv(1024).decode("utf-8"))
        if check_data(data):
            command = "python3 run_data_measurements.py --dataset=" + data["dataset"] + " --config=" + data["config"] + " --split=" + data["split"] + " --feature=" + data["feature"] + " --email=" + data["email"]
            if "label_field" in data:
                command += " --label_field=" + data["label_field"]
            subprocess.run(command, shell=True, check=True)
            conn.sendall(bytes(json.dumps({"success": True}), encoding="utf-8"))
        else:
            conn.sendall(bytes(json.dumps({"success": False}), encoding="utf-8"))