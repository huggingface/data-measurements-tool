from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from urllib.parse import parse_qs
import os

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

        command = "&& python3 run_data_measurements.py --dataset=" + dataset_args["dset_name"][0] + " --config=" + dataset_args["dset_config"][0] + " --split=" + dataset_args["split_name"][0] + " --feature=" + dataset_args["text_field"][0] + " --email=" + dataset_args["email"][0]
        if "label_field" in dataset_args:
            command += " --label_field=label"
        print('running command:', command)
        os.system(command)

with HTTPServer(('', 80), handler) as server:
    server.serve_forever()