import logging
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns
import statistics
from os.path import join as pjoin
import pandas as pd
import utils
from utils import dataset_utils as ds_utils


from collections import Counter
from os.path import exists, isdir
from os.path import join as pjoin

TOKENIZED_FIELD = "tokenized_text"
LENGTH_FIELD = "length"

logs = utils.prepare_logging(__file__)

def make_fig_lengths(lengths_df):
    fig_tok_lengths, axs = plt.subplots(figsize=(15, 6), dpi=150)
    sns.histplot(data=lengths_df, kde=True, bins=100, ax=axs)
    sns.rugplot(data=lengths_df, ax=axs)
    return fig_tok_lengths


class DMTHelper:
    def __init__(self, dstats, load_only=False, save=True):
        self.tokenized_df = dstats.tokenized_df
        # Whether to only use cache
        self.load_only = load_only
        # Whether to try using cache first.
        # Must be true when self.load_only = True; this function assures that.
        self.use_cache = ds_utils.check_load_and_use_cache(self.load_only, dstats.use_cache)
        self.cache_path = dstats.cache_path
        self.save = save
        self.fig_lengths = None
        # Lengths class object
        self.lengths_obj = None
        # Data structure for lengths list that can be easily manipulated
        self.lengths_df = None
        # Measurements
        self.avg_length = None
        self.std_length = None
        # Dict for the measurements
        self.length_stats_dict = {}
        # Filenames
        self.lengths_dir = "lengths"
        length_meas_json = "length_measurements.json"
        lengths_fig_png = "lengths_fig.png"
        lengths_list_feather = "length_list.feather"
        self.length_stats_json_fid = pjoin(self.cache_path, self.lengths_dir, length_meas_json)
        self.lengths_fig_png_fid = pjoin(self.cache_path, self.lengths_dir, lengths_fig_png)
        self.lengths_list_feather_fid = pjoin(self.cache_path, self.lengths_dir, lengths_list_feather)

    def run_DMT_processing(self):
        """
        Associates the measurements to the dataset.
        """
        # First look to see what we can load from cache.
        if self.use_cache:
            self.lengths_df, self.length_stats_dict, self.fig_lengths = self._load_lengths_cache()
            if isinstance(self.lengths_df, pd.DataFrame):
                logs.info("Loaded cached sentences with lengths.")
            if self.length_stats_dict != {}:
                logs.info("Loaded cached length results.")
            if isinstance(self.fig_lengths, Figure):
                logs.info("Loaded cached length figure.")
        # If we do not have a figure loaded from cache...
        # Compute length statistics.
        if not self.length_stats_dict and not self.load_only:
            logs.info("Preparing length results")
            self.lengths_obj = self._prepare_lengths()
            self.length_stats_dict = self.lengths_obj.length_stats_dict
            self.avg_length = self.lengths_obj.avg_length
            self.std_length = self.lengths_obj.std_length
            self.lengths_df = self.lengths_obj.lengths_df
            logs.info("Creating lengths figure.")
            self.fig_lengths = make_fig_lengths(self.lengths_df)
            # Finish
            if self.save:
                self._write_lengths_cache()

    def _load_lengths_cache(self):
        lengths_df = None
        fig_lengths = None
        length_stats_dict = {}
        # Dataframe with <sentence, length> exists. Load it.
        if exists(self.lengths_list_feather_fid):
            lengths_df = ds_utils.read_df(self.lengths_list_feather_fid)
        # Image exists. Load it.
        if exists(self.lengths_fig_png_fid):
            fig_lengths = mpimg.imread(self.lengths_fig_png_fid)
        # Measurements exist. Load them.
        if exists(self.length_stats_json_fid):
            # Loads the length sufficient statistics
            length_stats_dict = ds_utils.read_json(self.length_stats_json_fid)
        return lengths_df, length_stats_dict, fig_lengths

    def _write_lengths_cache(self):
        # Writes the data structures using the corresponding filetypes.
        ds_utils.make_path(pjoin(self.cache_path, self.lengths_dir))
        if self.length_stats_dict != {}:
            ds_utils.write_json(self.length_stats_dict, self.length_stats_json_fid)
        if isinstance(self.fig_lengths, Figure):
            self.fig_lengths.savefig(self.lengths_fig_png_fid)
        if isinstance(self.lengths_df, pd.DataFrame):
            ds_utils.write_df(self.lengths_df, self.lengths_list_feather_fid)


    def _prepare_lengths(self):
        """Loads a Lengths object and computes length statistics"""
        # Length object for the dataset
        lengths_obj = Lengths(dataset=self.tokenized_df)
        lengths_obj.prepare_lengths()
        return lengths_obj


    def get_filenames(self):
        lengths_fid_dict = {"statistics": self.length_stats_json_fid,
                            "figure png": self.lengths_fig_png_fid,
                            "list": self.lengths_list_feather_fid}
        return lengths_fid_dict


class Lengths:
    """Generic class for text length processing.
    Uses DataFrames for faster processing.
    Given a dataframe with tokenized words, computes statistics.
    """

    def __init__(self, dataset):
        # TODO: Implement the option of an input tokenizer
        self.dset_df = dataset
        self.avg_length = None
        self.std_length = None
        self.num_uniq_lengths = None
        self.length_stats_dict = {}
        self.lengths_df = None

    def prepare_lengths(self):
        self.dset_df[LENGTH_FIELD] = self.dset_df[TOKENIZED_FIELD].apply(len)
        self.lengths_df = self.dset_df[[LENGTH_FIELD]].sort_values(by=[LENGTH_FIELD], ascending=False)
        lengths_array = self.lengths_df[LENGTH_FIELD] #.iloc[:, 0]
        self.avg_length = statistics.mean(lengths_array)
        self.std_length = statistics.stdev(lengths_array)
        self.num_uniq_lengths = len(lengths_array.unique())
        self.length_stats_dict = {
            "average_instance_length": self.avg_length,
            "standard_dev_instance_length": self.std_length,
            "num_instance_lengths": self.num_uniq_lengths,
        }