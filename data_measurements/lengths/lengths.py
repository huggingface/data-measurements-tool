import logging
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
from os.path import join as pjoin
import pandas as pd
from utils import dataset_utils as ds_utils


TOKENIZED_FIELD = "tokenized_text"

logs = logging.getLogger(__name__)
logs.setLevel(logging.WARNING)
logs.propagate = False

if not logs.handlers:
    # Logging info to log file
    file = logging.FileHandler("./log_files/lengths.log")
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


def make_fig_lengths(length_df):
    fig_tok_length, axs = plt.subplots(figsize=(15, 6), dpi=150)
    sns.histplot(data=length_df, kde=True, bins=100, ax=axs)
    sns.rugplot(data=length_df, ax=axs)
    return fig_tok_length


class DMTHelper:
    def __init__(self, dstats, save):
        self.tokenized_df = dstats.tokenized_df
        self.use_cache = dstats.use_cache
        self.fig_lengths = dstats.fig_lengths
        self.length_results = dstats.length_results
        self.cache_path = dstats.cache_path
        # TODO: Should this just be an attribute of dstats instead?
        self.save = save
        # Filenames
        self.length_dir = "lengths"
        length_json = "lengths.json"
        length_fig_png = "lengths_fig.png"
        self.lengths_json_fid = pjoin(self.cache_path, self.length_dir, length_json)
        self.lengths_fig_png_fid = pjoin(self.cache_path, self.length_dir, length_fig_png)

    def run_DMT_processing(self):
        # First look to see what we can load from cache.
        if self.use_cache:
            self.fig_lengths, self.length_results = self._load_length_cache()
            if self.fig_lengths:
                logs.info("Loaded cached length figure.")
            if self.length_results:
                logs.info("Loaded cached length results.")
        # If we do not have a figure loaded from cache...
        # Compute length statistics.
        if not self.length_results:
            logs.info("Preparing length results")
            self.length_results = self._prepare_lengths()
        # Create figure
        if not self.fig_lengths:
            logs.info("Creating length figure.")
            self.fig_lengths = make_fig_lengths(self.length_results)
        # Finish
        if self.save:
            self._write_length_cache()

    def _write_length_cache(self):
        ds_utils.make_cache_path(pjoin(self.cache_path, self.length_dir))
        if self.length_results:
            ds_utils.write_json(self.length_results, self.lengths_json_fid)
        if self.fig_lengths:
            self.fig_lengths.savefig(self.lengths_fig_png_fid)

    def _prepare_lengths(self):
        """Loads a Lengths object and computes length statistics"""
        # Length object for the dataset
        length_obj = Lengths(dataset=self.tokenized_df[TOKENIZED_FIELD])
        # TODO(?): DataFrame is a faster data structure to use (I think),
        # but no one appears to be using it, so move it to be a
        # non-default option?
        length_results = length_obj.prepare_lengths()
        return length_results

    def _load_length_cache(self):
        fig_lengths = {}
        length_results = {}
        # Image exists. Load it.
        if exists(self.lengths_fig_png_fid):
            fig_lengths = mpimg.imread(self.lengths_fig_png_fid)
        # Measurements exist. Load them.
        if exists(self.lengths_json_fid):
            # Loads the length list, names, and results
            length_results = ds_utils.read_json(self.lengths_json_fid)
        return length_results, fig_lengths

    def get_length_filenames(self):
        length_fid_dict = {"statistics": self.lengths_json_fid,
                           "figure png": self.lengths_fig_png_fid}
        return length_fid_dict


class Lengths:
    """Generic class for text length processing.
    Uses DataFrames for faster processing.
    Given a dataframe with tokenized words, computes statistics.
    """

    def __init__(self, dataset):
        # TODO: Implement the option of an input tokenizer
        self.length_df = dataset
        self.avg_length = None
        self.std_length = None
        self.num_uniq_lengths = None
        self.length_stats_dict = {}

    def prepare_lengths(self):
        self.length_df = pd.DataFrame(self.length_df.apply(len))
        length_array = self.length_df.iloc[:, 0]
        self.avg_length = statistics.mean(length_array)
        self.std_length = statistics.stdev(length_array)
        self.num_uniq_lengths = len(length_array.unique())
        self.length_stats_dict = {
            "average_instance_length": self.avg_length,
            "standard_dev_instance_length": self.std_length,
            "num_instance_lengths": self.num_uniq_lengths,
        }
        print(self.length_stats_dict)
        return self.length_stats_dict