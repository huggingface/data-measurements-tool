#import evaluate
from evaluate import load
import logging
import os
import pandas as pd
import plotly.express as px
import utils
import utils.dataset_utils as ds_utils
from collections import Counter
from os.path import exists, isdir
from os.path import join as pjoin

LABEL_FIELD = "labels"
LABEL_NAMES = "label_names"
LABEL_LIST = "label_list"
LABEL_MEASUREMENT = "label_measurement"
# Specific to the evaluate library
EVAL_LABEL_MEASURE = "label_distribution"
EVAL_LABEL_ID = "labels"
EVAL_LABEL_FRAC = "fractions"
# TODO: This should ideally be in what's returned from the evaluate library
EVAL_LABEL_SUM = "sums"

logs = utils.prepare_logging(__file__)


def map_labels(label_field, ds_name_to_dict, ds_name, config_name):
    label_field, label_names = (
        ds_name_to_dict[ds_name][config_name]["features"][label_field][0]
        if len(
            ds_name_to_dict[ds_name][config_name]["features"][label_field]) > 0
        else ((), [])
    )
    return label_names


def make_label_results_dict(label_measurement, label_names):
    label_dict = {LABEL_MEASUREMENT: label_measurement,
                  LABEL_NAMES: label_names}
    return label_dict


def make_label_fig(label_results, chart_type="pie"):
    try:
        label_names = label_results[LABEL_NAMES]
        label_measurement = label_results[LABEL_MEASUREMENT]
        label_sums = label_measurement[EVAL_LABEL_SUM]
        if chart_type == "bar":
            fig_labels = plt.bar(
                label_measurement[EVAL_LABEL_MEASURE][EVAL_LABEL_ID],
                label_measurement[EVAL_LABEL_MEASURE][EVAL_LABEL_FRAC])
        else:
            if chart_type != "pie":
                logs.info("Oops! Don't have that chart-type implemented.")
                logs.info("Making the default pie chart")
            fig_labels = px.pie(names=label_names, values=label_sums)
    except KeyError:
        logs.info("Input label data missing required key(s).")
        logs.info("We require %s, %s" % (LABEL_NAMES, LABEL_MEASUREMENT))
        logs.info("We found: %s" % ",".join(label_results.keys()))
        return False
    return fig_labels


def extract_label_names(label_field, ds_name, config_name):
    ds_name_to_dict = ds_utils.get_dataset_info_dicts(ds_name)
    label_names = map_labels(label_field, ds_name_to_dict, ds_name, config_name)
    return label_names


class DMTHelper:
    """Helper class for the Data Measurements Tool.
    This allows us to keep all variables and functions related to labels
    in one file.
    """

    def __init__(self, dstats, load_only, save):
        logs.info("Initializing labels.")
        # -- Data Measurements Tool variables
        self.label_results = dstats.label_results
        self.fig_labels = dstats.fig_labels
        self.use_cache = dstats.use_cache
        self.cache_dir = dstats.dataset_cache_dir
        self.load_only = load_only
        self.save = save
        # -- Hugging Face Dataset variables
        self.label_field = dstats.label_field
        # Input HuggingFace dataset
        self.dset = dstats.dset
        self.dset_name = dstats.dset_name
        self.dset_config = dstats.dset_config
        self.label_names = dstats.label_names
        # -- Filenames
        self.label_dir = "labels"
        label_json = "labels.json"
        label_fig_json = "labels_fig.json"
        label_fig_html = "labels_fig.html"
        self.labels_json_fid = pjoin(self.cache_dir, self.label_dir,
                                     label_json)
        self.labels_fig_json_fid = pjoin(self.cache_dir, self.label_dir,
                                         label_fig_json)
        self.labels_fig_html_fid = pjoin(self.cache_dir, self.label_dir,
                                         label_fig_html)

    def run_DMT_processing(self):
        """
        Loads or prepares the Labels measurements and figure as specified by
        the DMT options.
        """
        # First look to see what we can load from cache.
        if self.use_cache:
            logs.info("Trying to load labels.")
            self.fig_labels, self.label_results = self._load_label_cache()
            if self.fig_labels:
                logs.info("Loaded cached label figure.")
            if self.label_results:
                logs.info("Loaded cached label results.")
        # If we can prepare the results afresh...
        if not self.load_only:
            # If we didn't load them already, compute label statistics.
            if not self.label_results:
                logs.info("Preparing labels.")
                self.label_results = self._prepare_labels()
            # If we didn't load it already, create figure.
            if not self.fig_labels:
                logs.info("Creating label figure.")
                self.fig_labels = \
                    make_label_fig(self.label_results)
            # Finish
            if self.save:
                self._write_label_cache()

    def _load_label_cache(self):
        fig_labels = {}
        label_results = {}
        # Measurements exist. Load them.
        if exists(self.labels_json_fid):
            # Loads the label list, names, and results
            label_results = ds_utils.read_json(self.labels_json_fid)
        # Image exists. Load it.
        if exists(self.labels_fig_json_fid):
            fig_labels = ds_utils.read_plotly(self.labels_fig_json_fid)
        return fig_labels, label_results

    def _prepare_labels(self):
        """Loads a Labels object and computes label statistics"""
        # Label object for the dataset
        label_obj = Labels(dataset=self.dset,
                           dataset_name=self.dset_name,
                           config_name=self.dset_config)
        # TODO: Handle the case where there are multiple label columns.
        # The logic throughout the code assumes only one.
        if type(self.label_field) == tuple:
            label_field = self.label_field[0]
        elif type(self.label_field) == str:
            label_field = self.label_field
        else:
            logs.warning("Unexpected format %s for label column name(s). "
                         "Not computing label statistics." %
                         type(self.label_field))
            return {}
        label_results = label_obj.prepare_labels(label_field, self.label_names)
        return label_results

    def _write_label_cache(self):
        ds_utils.make_path(pjoin(self.cache_dir, self.label_dir))
        if self.label_results:
            ds_utils.write_json(self.label_results, self.labels_json_fid)
        if self.fig_labels:
            ds_utils.write_plotly(self.fig_labels, self.labels_fig_json_fid)
            self.fig_labels.write_html(self.labels_fig_html_fid)

    def get_label_filenames(self):
        label_fid_dict = {"statistics": self.labels_json_fid,
                          "figure json": self.labels_fig_json_fid,
                          "figure html": self.labels_fig_html_fid}
        return label_fid_dict


class Labels:
    """Generic class for label processing.
    Uses the Dataset to extract the label column and compute label measurements.
    """

    def __init__(self, dataset, dataset_name=None, config_name=None):
        # Input HuggingFace Dataset.
        self.dset = dataset
        # These are used to extract label names, when the label names
        # are stored in the Dataset object but not in the "label" column
        # we are working with, which may instead just be ints corresponding to
        # the names
        self.ds_name = dataset_name
        self.config_name = config_name
        # For measurement data and additional metadata.
        self.label_results_dict = {}

    def prepare_labels(self, label_field, label_names=[]):
        """ Uses the evaluate library to return the label distribution. """
        logs.info("Inside main label calculation function.")
        # The input Dataset object
        # When the label field is not found, an error will be thrown.
        label_list = self.dset[label_field]
        # Get the evaluate library's measurement for label distro.
        label_distribution = load(EVAL_LABEL_MEASURE)
        # Measure the label distro.
        label_measurement = label_distribution.compute(data=label_list)
        # TODO: Incorporate this summation into what the evaluate library returns.
        label_sum_dict = Counter(label_list)
        label_sums = [label_sum_dict[key] for key in sorted(label_sum_dict)]
        label_measurement["sums"] = label_sums
        if not label_names:
            # Have to extract the label names from the Dataset object when the
            # actual dataset columns are just ints representing the label names.
            label_names = extract_label_names(label_field, self.ds_name,
                                              self.config_name)
        label_results = make_label_results_dict(label_measurement, label_names)
        return label_results
