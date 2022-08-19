import evaluate
import logging
import os
import pandas as pd
import plotly.express as px
import utils.dataset_utils as utils
from collections import Counter
from os.path import exists, isdir
from os.path import join as pjoin

class DMTHelper:
    """Helper class for the Data Measurements Tool.
    This allows us to keep all variables and functions related to labels
    in one file.
    """

    def __init__(self, dstats, save):
        self.use_cache = dstats.use_cache
        self.fig_duplicates = dstats.fig_duplicates
        self.duplicate_results = dstats.duplicate_results
        self.cache_path = dstats.cache_path
        #self.label_field = dstats.label_field
        # TODO: Should this just be an attribute of dstats instead?
        self.save = save
        # Filenames
        self.duplicates_dir = "text_duplicates"
        duplicates_json = "text_duplicates.json"
        #label_fig_json = "labels_fig.json"
        #label_fig_html = "labels_fig.html"
        self.duplicates_result_json_fid = pjoin(self.cache_path, self.duplicates_dir,
                                         duplicates_json)
        #self.labels_fig_json_fid = pjoin(self.cache_path, self.label_dir,
        #                                 label_fig_json)
        #self.labels_fig_html_fid = pjoin(self.cache_path, self.label_dir,
        #                                 label_fig_html)

    def run_DMT_processing(self):
        # First look to see what we can load from cache.
        if self.use_cache:
            self.duplicate_results = self._load_duplicates_cache()
            if self.duplicate_results:
                logs.info("Loaded cached text duplicate results.")
        if not self.duplicate_results:
            self.duplicate_results = self._prepare_duplicates()
        if self.save:
            self._write_duplicates_cache()


    def _prepare_duplicates(self):
        duplicates = evaluate.load("text_duplicates")
        results = duplicates.compute(data=self.dstats.text_dset)
        return results

    def _load_duplicates_cache(self):
        results = {}
        if exists(self.duplicates_result_json_fid):
            results = utils.read_json(self.duplicates_result_json_fid)
        return results
    
    def _write_duplicates_cache(self):
        utils.make_cache_path(pjoin(self.cache_path, self.duplicates_dir))
        if self.duplicate_results_results:
            utils.write_json(self.duplicate_results, self.duplicates_result_json_fid)