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

TEXT_FIELD = "text"
TOKENIZED_FIELD = "tokenized_text"
LENGTH_FIELD = "length"

UNIQ = "num_instance_lengths"
AVG = "average_instance_length"
STD = "standard_dev_instance_length"

logs = utils.prepare_logging(__file__)

def make_fig_lengths(lengths_df):
    # How the hell is this working? plt transforms to sns  ?!
    logs.info("Creating lengths figure.")
    fig_tok_lengths, axs = plt.subplots(figsize=(15, 6), dpi=150)
    plt.xlabel("Number of tokens")
    plt.title("Binned counts of text lengths, with kernel density estimate and ticks for each instance.")
    sns.histplot(data=lengths_df, kde=True, ax=axs, x=LENGTH_FIELD, legend=False)
    sns.rugplot(data=lengths_df, ax=axs)
    return fig_tok_lengths

class DMTHelper:
    def __init__(self, dstats, load_only=False, save=True):
        self.tokenized_df = dstats.tokenized_df
        # Whether to only use cache
        self.load_only = load_only
        # Whether to try using cache first.
        # Must be true when self.load_only = True; this function assures that.
        self.use_cache = dstats.use_cache
        self.cache_dir = dstats.dataset_cache_dir
        self.save = save
        # Lengths class object
        self.lengths_obj = None
        # Content shared in the DMT:
        # The figure, the table, and the sufficient statistics (measurements)
        self.fig_lengths = None
        self.lengths_df = None
        self.avg_length = None
        self.std_length = None
        self.uniq_counts = None
        # Dict for the measurements, used in caching
        self.length_stats_dict = {}
        # Filenames, used in caching
        self.lengths_dir = "lengths"
        length_meas_json = "length_measurements.json"
        lengths_fig_png = "lengths_fig.png"
        lengths_df_json = "lengths_table.json"
        self.length_stats_json_fid = pjoin(self.cache_dir, self.lengths_dir, length_meas_json)
        self.lengths_fig_png_fid = pjoin(self.cache_dir, self.lengths_dir, lengths_fig_png)
        self.lengths_df_json_fid = pjoin(self.cache_dir, self.lengths_dir, lengths_df_json)

    def run_DMT_processing(self):
        """
        Gets data structures for the figure, table, and measurements.
        """
        # First look to see what we can load from cache.
        if self.use_cache:
            logs.info("Trying to load from cache...")
            # Defines self.lengths_df, self.length_stats_dict, self.fig_lengths
            # This is the table, the dict of measurements, and the figure
            self.load_lengths_cache()
            # Sets the measurements as attributes of the DMT object
            self.set_attributes()
        # If we do not have measurements loaded from cache...
        if not self.length_stats_dict and not self.load_only:
            logs.info("Preparing length results")
            # Compute length statistics. Uses the Lengths class.
            self.lengths_obj = self._prepare_lengths()
            # Dict of measurements
            self.length_stats_dict = self.lengths_obj.length_stats_dict
            # Table of text and lengths
            self.lengths_df = self.lengths_obj.lengths_df
            # Sets the measurements in the length_stats_dict
            self.set_attributes()
            # Makes the figure
            self.fig_lengths = make_fig_lengths(self.lengths_df)
            # Finish
            if self.save:
                logs.info("Saving results.")
                self._write_lengths_cache()

    def set_attributes(self):
        if self.length_stats_dict:
            self.avg_length = self.length_stats_dict[AVG]
            self.std_length = self.length_stats_dict[STD]
            self.uniq_counts = self.length_stats_dict[UNIQ]
        else:
            logs.info("Dictionary of results is empty, couldn't load measurements. =(")

    def load_lengths_cache(self):
        # Dataframe with <sentence, length> exists. Load it.
        if exists(self.lengths_df_json_fid):
            self.lengths_df = ds_utils.read_df(self.lengths_df_json_fid)
        # Image exists. Load it.
        if exists(self.lengths_fig_png_fid):
            self.fig_lengths = mpimg.imread(self.lengths_fig_png_fid)
        # Measurements exist. Load them.
        if exists(self.length_stats_json_fid):
            # Loads the length measurements
            self.length_stats_dict = ds_utils.read_json(self.length_stats_json_fid)

    def _write_lengths_cache(self):
        # Writes the data structures using the corresponding filetypes.
        ds_utils.make_path(pjoin(self.cache_dir, self.lengths_dir))
        if self.length_stats_dict != {}:
            ds_utils.write_json(self.length_stats_dict, self.length_stats_json_fid)
        if isinstance(self.fig_lengths, Figure):
            self.fig_lengths.savefig(self.lengths_fig_png_fid)
        if isinstance(self.lengths_df, pd.DataFrame):
            ds_utils.write_df(self.lengths_df, self.lengths_df_json_fid)

    def _prepare_lengths(self):
        """Loads a Lengths object and computes length statistics"""
        # Length object for the dataset
        lengths_obj = Lengths(dataset=self.tokenized_df)
        lengths_obj.prepare_lengths()
        return lengths_obj

    def get_filenames(self):
        lengths_fid_dict = {"statistics": self.length_stats_json_fid,
                            "figure png": self.lengths_fig_png_fid,
                            "table": self.lengths_df_json_fid}
        return lengths_fid_dict


class Lengths:
    """Generic class for text length processing.
    Uses DataFrames for faster processing.
    Given a dataframe with tokenized words in a column called TOKENIZED_TEXT,
    and the text instances in a column called TEXT, compute statistics.
    """

    def __init__(self, dataset):
        self.dset_df = dataset
        # Dict of measurements
        self.length_stats_dict = {}
        # Measurements
        self.avg_length = None
        self.std_length = None
        self.num_uniq_lengths = None
        # Table of lengths and sentences
        self.lengths_df = None

    def prepare_lengths(self):
        self.lengths_df = pd.DataFrame(self.dset_df[TEXT_FIELD])
        self.lengths_df[LENGTH_FIELD] = self.dset_df[TOKENIZED_FIELD].apply(len)
        lengths_array = self.lengths_df[LENGTH_FIELD]
        self.avg_length = statistics.mean(lengths_array)
        self.std_length = statistics.stdev(lengths_array)
        self.num_uniq_lengths = len(lengths_array.unique())
        self.length_stats_dict = {
            "average_instance_length": self.avg_length,
            "standard_dev_instance_length": self.std_length,
            "num_instance_lengths": self.num_uniq_lengths,
        }
