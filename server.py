from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from urllib.parse import parse_qs
import subprocess

class handler(BaseHTTPRequestHandler):

    def do_POST(self):

        content_len = int(self.headers.get('Content-Length'))
        query_string = self.rfile.read(content_len).decode("utf-8")
        dataset_args = parse_qs(query_string)

        self.send_response(200)
        self.send_header('Content-type','text/html')
        self.end_headers()

        message = "success"
        self.wfile.write(bytes(message, "utf8"))

        print('recieved message:', dataset_args)

        command = "python3 run_data_measurements.py --dataset=" + dataset_args["dset_name"][0] + " --config=" + dataset_args["dset_config"][0] + " --split=" + dataset_args["split_name"][0] + " --feature=" + dataset_args["text_field"][0] + " --email=" + dataset_args["email"][0]
        if "label_field" in dataset_args and len(dataset_args["label_field"]) == 1:
            command += " --label_field=" + dataset_args["label_field"][0]
        print('running command:', command)
        subprocess.Popen(command, shell=True)

with HTTPServer(('', 80), handler) as server:
    server.serve_forever()