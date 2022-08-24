import evaluate
import logging
import os
import pandas as pd
import plotly.express as px
import utils
import utils.dataset_utils as ds_utils
from collections import Counter
from os.path import exists, isdir
from os.path import join as pjoin
from pathlib import Path

# String constants used to index data and results.
TEXT = "text"
# String constants defined in the evaluate library.
DUPS_FRAC = "duplicate_fraction"
DUPS_DICT = "duplicates_dict"
# This isn't in the evaluate measurement, but TODO to add that...
# DUPS_SUM = "duplicate_sum"

logs = utils.prepare_logging(__file__)

class DMTHelper:
    """Helper class for the Data Measurements Tool.
    This allows us to keep all variables and functions related to labels
    in one file.
    Does caching and using the evaluate library for computation.
    """

    def __init__(self, dstats, save):
        # Input HuggingFace Dataset.
        self.dset = dstats.text_dset[TEXT]
        self.duplicates_results = dstats.duplicates_results
        self.use_cache = dstats.use_cache
        self.cache_path = dstats.cache_path
        self.save = save
        # Filenames
        module = Path(fid).stem
        self.file_info = ds_utils.FileHandler(self, module)

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
            logs.info("Preparing duplicates.")
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
        if exists(self.file_info.module_result_json_fid):
            results = utils.read_json(self.file_info.module_result_json_fid)
        return results
    
    def _write_duplicates_cache(self):
        """Writes newly computer results to cache."""
        if self.duplicates_results:
            utils.write_json(self.duplicates_results, self.file_info.module_result_json_fid)
            # TODO: Use df_to_html rather than write_json_as_html;
            # this will make it possible to order the results.
            # But they must first be turned into a dataframe.
            utils.write_json_as_html(self.duplicates_results, self.file_info.module_result_html_fid)

    def get_duplicates_filenames(self):
        dups_fid_dict = self.file_info.get_filenames(has_html=True)
        return dups_fid_dict