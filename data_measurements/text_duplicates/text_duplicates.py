import evaluate
import logging
import os
import pandas as pd
import plotly.express as px
import utils.dataset_utils as utils
from collections import Counter
from os.path import exists, isdir
from os.path import join as pjoin

TEXT = "text"
# These are string constants defined in the evaluate library.
# They may need to be updated if the evaluate library changes these strings
DUPS_FRAC = "duplicate_fraction"
# Evaluate calls the dictionary a "list"
DUPS_DICT = "duplicates_list"
# This isn't in the evaluate measurement, but TODO to add that...
# DUPS_SUM = "duplicate_sum"

logs = logging.getLogger(__name__)
logs.setLevel(logging.WARNING)
logs.propagate = False

if not logs.handlers:
    # Logging info to log file
    file = logging.FileHandler("./log_files/text_duplicates.log")
    fileformat = logging.Formatter("%(asctime)s:%(message)s")
    file.setLevel(logging.INFO)
    file.setFormatter(fileformat)

    # Logging debug messages to stream
    stream = logging.StreamHandler()
    streamformat = logging.Formatter("[data_measurements_tool] %(message)s")
    stream.setLevel(logging.WARNING)
    stream.setFormatter(streamformat)

    logs.addHandler(file)
    logs.addHandler(stream)

class DMTHelper:
    """Helper class for the Data Measurements Tool.
    This allows us to keep all variables and functions related to labels
    in one file.
    Does caching and using the evaluate library for computation.
    """

    def __init__(self, dstats, save):
        # Input HuggingFace Dataset.
        self.dset = dstats.text_dset[TEXT]
        if self.dset is None:
            dstats.load_or_prepare_text_dset()
            self.dset = dstats.text_dset
        self.use_cache = dstats.use_cache
        self.duplicates_results = dstats.duplicates_results
        self.cache_path = dstats.cache_path
        #self.label_field = dstats.label_field
        # TODO: Should this just be an attribute of dstats instead?
        self.save = save
        # Filenames
        self.dups_dir = "text_duplicates"
        dups_json = "text_duplicates.json"
        dups_html = "text_duplicates.html"
        self.dups_result_json_fid = pjoin(self.cache_path, self.dups_dir, dups_json)
        self.dups_result_html_fid = pjoin(self.cache_path, self.dups_dir, dups_html)

    def run_DMT_processing(self, list_duplicates=True):
        """Calls functions to do the main work.
        DMT uses the full duplicates list in a widget,
        so it is set to default True.
        """

        # First look to see what we can load from cache.
        if self.use_cache:
            self.duplicates_results = self._load_duplicates_cache()
            if self.duplicates_results:
                logs.info("Loaded cached text duplicate results.")
        if not self.duplicates_results:
            self.duplicates_results = self._prepare_duplicates(list_duplicates=list_duplicates)
            logs.info("Prepared duplicates.")
        if self.save:
            self._write_duplicates_cache()

    def _prepare_duplicates(self, list_duplicates=True):
        """Wraps the evaluate library."""
        duplicates = evaluate.load("text_duplicates")
        results = duplicates.compute(data=self.dset, list_duplicates=list_duplicates)
        return results

    def _load_duplicates_cache(self):
        """Loads previously computed results from cache."""
        results = {}
        if exists(self.dups_result_json_fid):
            results = utils.read_json(self.dups_result_json_fid)
        return results
    
    def _write_duplicates_cache(self):
        """Writes newly computer results to cache."""
        utils.make_cache_path(pjoin(self.cache_path, self.dups_dir))
        if self.duplicates_results:
            utils.write_json(self.duplicates_results, self.dups_result_json_fid)
            # TODO: Use df_to_html rather than write_json_as_html;
            # this will make it possible to order the results.
            # But they must first be turned into a dataframe.
            utils.write_json_as_html(self.duplicates_results, self.dups_result_html_fid)

    def get_duplicates_filenames(self):
        dups_fid_dict = {"statistics": self.dups_result_json_fid, "html":self.dups_result_html_fid}
        return dups_fid_dict