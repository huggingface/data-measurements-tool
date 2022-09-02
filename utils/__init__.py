import logging
from pathlib import Path
import utils.dataset_utils as ds_utils


def prepare_logging(fid):
    ds_utils.make_path("./log_files/")
    logs = logging.getLogger(__name__)
    logs.setLevel(logging.INFO)
    logs.propagate = False
    log_fid = Path(fid).stem

    # Logging info to log file
    file_path = ("./log_files/%s.log" % log_fid)
    print("Logging output in %s" % file_path)
    ds_utils.make_path("./log_files/")
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