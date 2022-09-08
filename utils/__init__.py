import logging
from pathlib import Path
import utils.dataset_utils as ds_utils

def prepare_logging(fid):
    logs = logging.getLogger(__name__)
    logs.setLevel(logging.DEBUG)
    logs.propagate = False
    log_fid = Path(fid).stem

    if not logs.handlers:
        # Logging info to log file
        print("Logging output in ../log_files/")
        file_path = ("./log_files/%s.log" % log_fid)
        ds_utils.make_path("./log_files/")
        file = logging.FileHandler(file_path)
        fileformat = logging.Formatter("%(asctime)s:%(pathname)s,  %(module)s:%(lineno)s\n%(message)s")
        file.setLevel(logging.INFO)
        file.setFormatter(fileformat)

        # Logging debug messages to stream
        stream = logging.StreamHandler()
        streamformat = logging.Formatter("[data_measurements_tool] %(pathname)s,  %(module)s:%(lineno)s\n%(message)s")
        stream.setLevel(logging.DEBUG)
        stream.setFormatter(streamformat)

        logs.addHandler(file)
        logs.addHandler(stream)

    return logs