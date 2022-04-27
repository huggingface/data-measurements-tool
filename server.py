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
            for config_obj in data["dset_configs"].values():
                config_name = config_obj["config_name"]
                split_names = config_obj["splits"].keys()
                feature_names = config_obj["features"].keys()
                for split_name in split_names:
                    for feature_name in feature_names:
                        command = "python3 run_data_measurements.py --dataset=" + data["dset_name"] + " --config=" + config_name + " --split=" + split_name + " --feature=" + feature_name + " --email=" + data["email"]
                        if "label" in feature_names:
                            command += " --label_field=label"
                        subprocess.run(command, shell=True, check=True)
            conn.sendall(bytes(json.dumps({"success": True}), encoding="utf-8"))
        except Exception as e:
            print(e)
            conn.sendall(bytes(json.dumps({"success": False}), encoding="utf-8"))