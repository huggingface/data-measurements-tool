import logging
import os
from pathlib import Path


def prepare_logging(fid):
    # Create the directory for log files (if it doesn't exist)
    Path('./log_files').mkdir(exist_ok=True)
    log_fid = Path(fid).stem
    logs = logging.getLogger(log_fid)
    logs.setLevel(logging.ERROR)

    logs.propagate = False
    log_fid = Path(fid).stem
    if not logs.handlers:
        # Logging info to log file
        file_path = ("./log_files/%s.log" % log_fid)
        print("Logging output in %s " % file_path)
        file = logging.FileHandler(file_path)
        fileformat = logging.Formatter("%(asctime)s:%(pathname)s,  %(module)s:%(lineno)s\n%(message)s")
        file.setLevel(logging.ERROR)
        file.setFormatter(fileformat)
        # Logging debug messages to stream
        stream = logging.StreamHandler()
        streamformat = logging.Formatter("[data_measurements_tool] {%(pathname)s:%(lineno)d} %(module)s %(levelname)s - %(message)s")
        stream.setLevel(logging.ERROR)
        stream.setFormatter(streamformat)
        logs.addHandler(file)
        logs.addHandler(stream)
    return logs