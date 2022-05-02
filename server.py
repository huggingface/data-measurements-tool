# echo-server.py

import socket
import json
import subprocess

HOST = "127.0.0.1"  # Standard loopback interface address (localhost)
PORT = 65432  # Port to listen on (non-privileged ports are > 1023)

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    while True:
        conn, addr = s.accept()
        print(f"Connected by {addr}")
        data = json.loads(conn.recv(1024).decode("utf-8"))
        try:
            command = "python3 run_data_measurements.py --dataset=" + data["dset_name"] + " --config=" + data["dset_config_name"] + " --split=" + data["split_name"] + " --feature=" + data["text_field"][0] + " --email=" + data["email"] + " --label_field=" + data["label_field"][0]
            subprocess.run(command, shell=True, check=True)
            conn.sendall(bytes(json.dumps({"success": True}), encoding="utf-8"))
        except Exception as e:
            print(e)
            conn.sendall(bytes(json.dumps({"success": False}), encoding="utf-8"))