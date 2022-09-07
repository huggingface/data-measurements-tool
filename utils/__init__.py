import logging
import os
from pathlib import Path

def prepare_logging(fid):
    ds_utils.make_path("./log_files/")
    log_fid = Path(fid).stem
    logs = logging.getLogger(log_fid)
    logs.setLevel(logging.INFO)
    logs.propagate = False

    if not logs.handlers:
        # Logging info to log file
        print("Logging output in ./log_files/")
        os.makedirs("./log_files/", exist_ok=True)
        file_path = ("./log_files/%s.log" % log_fid)
        file = logging.FileHandler(file_path)
        fileformat = logging.Formatter("%(asctime)s:%(message)s")
        file.setLevel(logging.INFO)
        file.setFormatter(fileformat)

    # Logging debug messages to stream
    stream = logging.StreamHandler()
    streamformat = logging.Formatter("[data_measurements_tool] %(message)s")
    stream.setLevel(logging.DEBUG)
    stream.setFormatter(streamformat)

    logs.addHandler(file)
    logs.addHandler(stream)

    return logs