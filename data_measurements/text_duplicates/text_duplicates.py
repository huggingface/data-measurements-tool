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
EVAL_DUP_FRAC = "duplicate_fraction"
EVAL_DUP_LIST = "duplicates_list"

class DMTHelper:
    """Helper class for the Data Measurements Tool.
    This allows us to keep all variables and functions related to labels
    in one file.
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
        self.duplicates_dir = "text_duplicates"
        duplicates_json = "text_duplicates.json"
        duplicates_html = "text_duplicates.html"
        #label_fig_json = "labels_fig.json"
        #label_fig_html = "labels_fig.html"
        self.duplicates_result_json_fid = pjoin(self.cache_path, self.duplicates_dir,
                                         duplicates_json)
        self.duplicates_result_html_fid = pjoin(self.cache_path, self.duplicates_dir, duplicates_html)

    def run_DMT_processing(self):
        # First look to see what we can load from cache.
        if self.use_cache:
            self.duplicates_results = self._load_duplicates_cache()
            if self.duplicates_results:
                logs.info("Loaded cached text duplicate results.")
        if not self.duplicates_results:
            self.duplicates_results = self._prepare_duplicates()
        if self.save:
            self._write_duplicates_cache()

    def _prepare_duplicates(self):
        duplicates = evaluate.load("text_duplicates")
        results = duplicates.compute(data=self.dset, list_duplicates=True)
        return results

    def _load_duplicates_cache(self):
        results = {}
        if exists(self.duplicates_result_json_fid):
            results = utils.read_json(self.duplicates_result_json_fid)
        return results
    
    def _write_duplicates_cache(self):
        utils.make_cache_path(pjoin(self.cache_path, self.duplicates_dir))
        if self.duplicates_results:
            utils.write_json(self.duplicates_results, self.duplicates_result_json_fid)
            utils.write_html(self.duplicates_results, self.duplicates_result_html_fid)

    def get_duplicates_filenames(self):
        duplicates_fid_dict = {"statistics": self.duplicates_result_json_fid}
        return duplicates_fid_dict