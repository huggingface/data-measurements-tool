# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import statistics
from os import mkdir, getenv
from os.path import exists, isdir
from os.path import join as pjoin
from pathlib import Path
# from dotenv import load_dotenv

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import nltk
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from datasets import load_from_disk, load_metric
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from huggingface_hub import Repository, list_datasets

from utils import dataset_utils as utils
from utils.dataset_utils import (CNT, DEDUP_TOT, EMBEDDING_FIELD, LENGTH_FIELD,
                            OUR_LABEL_FIELD, OUR_TEXT_FIELD, PERPLEXITY_FIELD, PROP,
                            TEXT_NAN_CNT, TOKENIZED_FIELD, TOT_OPEN_WORDS,
                            TOT_WORDS, VOCAB, WORD)
from data_measurements.embeddings.embeddings import Embeddings
# TODO(meg): Incorporate this from evaluate library.
# import evaluate
from data_measurements.zipf.zipf import Zipf, make_zipf_fig, get_zipf_fids
from data_measurements.npmi.npmi import nPMI

#if Path(".env").is_file():
#    load_dotenv(".env")

HF_TOKEN = getenv("HF_TOKEN")

pd.options.display.float_format = "{:,.3f}".format

logs = logging.getLogger(__name__)
logs.setLevel(logging.WARNING)
logs.propagate = False

if not logs.handlers:

    # Logging info to log file
    file = logging.FileHandler("./log_files/dataset_statistics.log")
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


# TODO: Read this in depending on chosen language / expand beyond english
nltk.download("stopwords")
_CLOSED_CLASS = (
    stopwords.words("english")
    + [
        "t",
        "n",
        "ll",
        "d",
        "wasn",
        "weren",
        "won",
        "aren",
        "wouldn",
        "shouldn",
        "didn",
        "don",
        "hasn",
        "ain",
        "couldn",
        "doesn",
        "hadn",
        "haven",
        "isn",
        "mightn",
        "mustn",
        "needn",
        "shan",
        "would",
        "could",
        "dont",
        "u",
    ]
    + [str(i) for i in range(0, 21)]
)
_IDENTITY_TERMS = [
    "man",
    "woman",
    "non-binary",
    "gay",
    "lesbian",
    "queer",
    "trans",
    "straight",
    "cis",
    "she",
    "her",
    "hers",
    "he",
    "him",
    "his",
    "they",
    "them",
    "their",
    "theirs",
    "himself",
    "herself",
]
# treating inf values as NaN as well
pd.set_option("use_inf_as_na", True)

_MIN_VOCAB_COUNT = 10
_TREE_DEPTH = 12
_TREE_MIN_NODES = 250
# as long as we're using sklearn - already pushing the resources
_MAX_CLUSTER_EXAMPLES = 5000
_NUM_VOCAB_BATCHES = 2000
_TOP_N = 100
_CVEC = CountVectorizer(token_pattern="(?u)\\b\\w+\\b", lowercase=True)

_PERPLEXITY = load_metric("perplexity")


class DatasetStatisticsCacheClass:

    def __init__(
        self,
        cache_dir,
        dset_name,
        dset_config,
        split_name,
        text_field,
        label_field,
        label_names,
        calculation=None,
        use_cache=False,
    ):
        # This is only used for standalone runs for each kind of measurement.
        self.calculation = calculation
        self.our_text_field = OUR_TEXT_FIELD
        self.our_length_field = LENGTH_FIELD
        self.our_label_field = OUR_LABEL_FIELD
        self.our_tokenized_field = TOKENIZED_FIELD
        self.our_embedding_field = EMBEDDING_FIELD
        self.cache_dir = cache_dir
        # path to the directory used for caching
        if isinstance(text_field, list):
            text_field = "-".join(text_field)
        self.dataset_cache_dir = f"{dset_name}_{dset_config}_{split_name}_{text_field}"
        # TODO: Having "cache_dir" and "cache_path" is confusing.
        self.cache_path = pjoin(
            self.cache_dir,
            self.dataset_cache_dir,
        )
        # Use stored data if there; otherwise calculate afresh
        self.use_cache = use_cache
        ### What are we analyzing?
        # name of the Hugging Face dataset
        self.dset_name = dset_name
        # name of the dataset config
        self.dset_config = dset_config
        # name of the split to analyze
        self.split_name = split_name
        # TODO: Chould this be "feature" ?
        # which text fields are we analysing?
        self.text_field = text_field
        # which label fields are we analysing?
        self.label_field = label_field
        # what are the names of the classes?
        self.label_names = label_names
        ## Hugging Face dataset objects
        self.dset = None  # original dataset
        # HF dataset with all of the self.text_field instances in self.dset
        self.text_dset = None
        self.dset_peek = None
        # HF dataset with text embeddings in the same order as self.text_dset
        self.embeddings_dset = None
        # HF dataset with all of the self.label_field instances in self.dset
        self.label_dset = None
        ## Data frames
        # Tokenized text
        self.tokenized_df = None
        # save sentence length histogram in the class so it doesn't ge re-computed
        self.length_df = None
        self.fig_tok_length = None
        # Data Frame version of self.label_dset
        self.label_df = None
        # save label pie chart in the class so it doesn't ge re-computed
        self.fig_labels = None
        # Save zipf fig so it doesn't need to be recreated.
        self.zipf_fig = None
        # Zipf object
        self.z = None
        # Vocabulary with word counts in the dataset
        self.vocab_counts_df = None
        # Vocabulary filtered to remove stopwords
        self.vocab_counts_filtered_df = None
        self.sorted_top_vocab_df = None
        ## General statistics and duplicates
        self.total_words = 0
        self.total_open_words = 0
        # Number of NaN values (NOT empty strings)
        self.text_nan_count = 0
        # Number of text items that appear more than once in the dataset
        self.dedup_total = 0
        # Duplicated text items along with their number of occurences ("count")
        self.dup_counts_df = None
        self.perplexities_df = None
        self.avg_length = None
        self.std_length = None
        self.general_stats_dict = None
        self.num_uniq_lengths = 0
        # clustering text by embeddings
        # the hierarchical clustering tree is represented as a list of nodes,
        # the first is the root
        self.node_list = []
        # save tree figure in the class so it doesn't ge re-computed
        self.fig_tree = None
        # keep Embeddings object around to explore clusters
        self.embeddings = None
        # nPMI
        # Holds a nPMIStatisticsCacheClass object
        self.npmi_stats = None
        # TODO: Have lowercase be an option for a user to set.
        self.to_lowercase = True
        # The minimum amount of times a word should occur to be included in
        # word-count-based calculations (currently just relevant to nPMI)
        self.min_vocab_count = _MIN_VOCAB_COUNT
        self.cvec = _CVEC
        # File definitions
        # path to the directory used for caching
        if not isinstance(text_field, str):
            text_field = ".".join(text_field)
        # if isinstance(label_field, str):
        #    label_field = label_field
        # else:
        #    label_field = "-".join(label_field)
        self.dataset_cache_dir = f"{dset_name}_{dset_config}_{split_name}_{text_field}"
        self.cache_path = pjoin(
            self.cache_dir,
            self.dataset_cache_dir,  # {label_field},
        )
        # Things that get defined later.
        self.fig_tok_length_png = None
        self.length_stats_dict = None

        # Try to pull from the hub to see if the cache already exists.
        try:
            if not isdir(self.cache_path) and self.dataset_cache_dir in [dataset_info.id.split("/")[-1] for dataset_info in list_datasets(author="datameasurements", use_auth_token=HF_TOKEN)]:
                repo = Repository(local_dir=self.cache_path, clone_from="datameasurements/" + self.dataset_cache_dir, repo_type="dataset", use_auth_token=HF_TOKEN)
            else:
                logs.warning("Cannot find cached repo on the hub.")
        except Exception as e:
            print(e)
            logs.warning("Cannot load cached repo on the hub.")

        # Cache files not needed for UI
        self.dset_fid = pjoin(self.cache_path, "base_dset")
        self.tokenized_df_fid = pjoin(self.cache_path, "tokenized_df.feather")
        self.label_dset_fid = pjoin(self.cache_path, "label_dset")

        # Needed for UI -- embeddings
        self.text_dset_fid = pjoin(self.cache_path, "text_dset")
        # Needed for UI
        self.dset_peek_json_fid = pjoin(self.cache_path, "dset_peek.json")

        ## Label cache files.
        # Needed for UI
        self.fig_labels_json_fid = pjoin(self.cache_path, "fig_labels.json")

        ## Length cache files
        # Needed for UI
        self.length_df_fid = pjoin(self.cache_path, "length_df.feather")
        # Needed for UI
        self.length_stats_json_fid = pjoin(self.cache_path, "length_stats.json")
        self.vocab_counts_df_fid = pjoin(self.cache_path, "vocab_counts.feather")
        # Needed for UI
        self.dup_counts_df_fid = pjoin(self.cache_path, "dup_counts_df.feather")
        # Needed for UI
        self.perplexities_df_fid = pjoin(self.cache_path, "perplexities_df.feather")
        # Needed for UI
        self.fig_tok_length_fid = pjoin(self.cache_path, "fig_tok_length.png")

        ## General text stats
        # Needed for UI
        self.general_stats_json_fid = pjoin(self.cache_path, "general_stats_dict.json")
        # Needed for UI
        self.sorted_top_vocab_df_fid = pjoin(
            self.cache_path, "sorted_top_vocab.feather"
        )


        ## Embeddings cache files
        # Needed for UI
        self.node_list_fid = pjoin(self.cache_path, "node_list.th")
        # Needed for UI
        self.fig_tree_json_fid = pjoin(self.cache_path, "fig_tree.json")

        self.live = False

    def get_cache_dir(self):
        return self.cache_path

    def set_deployment(self, live=True):
        """
        Function that we can hit when we deploy, so that cache files are not
        written out/recalculated, but instead that part of the UI can be punted.
        """
        self.live = live

    def check_cache_dir(self):
        """
        First function to call to create the cache directory.
        If in deployment mode and cache directory does not already exist,
        return False.
        """
        if self.live:
            return isdir(self.cache_path)
        else:
            if not isdir(self.cache_path):
                logs.warning("Creating cache directory %s." % self.cache_path)
                if not isdir(self.cache_dir):
                    mkdir(self.cache_dir)
                mkdir(self.cache_path)
            return isdir(self.cache_path)


    def get_base_dataset(self):
        """Gets a pointer to the truncated base dataset object."""
        if not self.dset:
            self.dset = utils.load_truncated_dataset(
                self.dset_name,
                self.dset_config,
                self.split_name,
                cache_name=self.dset_fid,
                use_cache=True,
                use_streaming=True,
            )

    def load_or_prepare_general_stats(self, save=True):
        """
        Content for expander_general_stats widget.
        Provides statistics for total words, total open words,
        the sorted top vocab, the NaN count, and the duplicate count.
        Args:

        Returns:

        """
        # General statistics
        if (
            self.use_cache
            and exists(self.general_stats_json_fid)
            and exists(self.dup_counts_df_fid)
            and exists(self.sorted_top_vocab_df_fid)
        ):
            logs.info("Loading cached general stats")
            self.load_general_stats()
        else:
            if not self.live:
                logs.info("Preparing general stats")
                self.prepare_general_stats()
                if save:
                    utils.write_df(self.sorted_top_vocab_df, self.sorted_top_vocab_df_fid)
                    utils.write_df(self.dup_counts_df, self.dup_counts_df_fid)
                    utils.write_json(self.general_stats_dict, self.general_stats_json_fid)

    def load_or_prepare_text_lengths(self, save=True):
        """
        The text length widget relies on this function, which provides
        a figure of the text lengths, some text length statistics, and
        a text length dataframe to peruse.
        Args:
            save:
        Returns:

        """
        # Text length figure
        if self.use_cache and exists(self.fig_tok_length_fid):
            self.fig_tok_length_png = mpimg.imread(self.fig_tok_length_fid)
        else:
            if not self.live:
                self.prepare_fig_text_lengths()
                if save:
                    self.fig_tok_length.savefig(self.fig_tok_length_fid)
        # Text length dataframe
        if self.use_cache and exists(self.length_df_fid):
            self.length_df = utils.read_df(self.length_df_fid)
        else:
            if not self.live:
                self.prepare_length_df()
                if save:
                    utils.write_df(self.length_df, self.length_df_fid)

        # Text length stats.
        if self.use_cache and exists(self.length_stats_json_fid):
            with open(self.length_stats_json_fid, "r") as f:
                self.length_stats_dict = json.load(f)
            self.avg_length = self.length_stats_dict["avg length"]
            self.std_length = self.length_stats_dict["std length"]
            self.num_uniq_lengths = self.length_stats_dict["num lengths"]
        else:
            if not self.live:
                self.prepare_text_length_stats()
                if save:
                    utils.write_json(self.length_stats_dict, self.length_stats_json_fid)

    def prepare_length_df(self):
        if not self.live:
            if self.tokenized_df is None:
                self.tokenized_df = self.do_tokenization()
            self.tokenized_df[LENGTH_FIELD] = self.tokenized_df[TOKENIZED_FIELD].apply(
                len
            )
            self.length_df = self.tokenized_df[
                [LENGTH_FIELD, OUR_TEXT_FIELD]
            ].sort_values(by=[LENGTH_FIELD], ascending=True)

    def prepare_text_length_stats(self):
        if not self.live:
            if (
                self.tokenized_df is None
                or LENGTH_FIELD not in self.tokenized_df.columns
                or self.length_df is None
            ):
                self.prepare_length_df()
            avg_length = sum(self.tokenized_df[LENGTH_FIELD]) / len(
                self.tokenized_df[LENGTH_FIELD]
            )
            self.avg_length = round(avg_length, 1)
            std_length = statistics.stdev(self.tokenized_df[LENGTH_FIELD])
            self.std_length = round(std_length, 1)
            self.num_uniq_lengths = len(self.length_df["length"].unique())
            self.length_stats_dict = {
                "avg length": self.avg_length,
                "std length": self.std_length,
                "num lengths": self.num_uniq_lengths,
            }

    def prepare_fig_text_lengths(self):
        if not self.live:
            if (
                self.tokenized_df is None
                or LENGTH_FIELD not in self.tokenized_df.columns
            ):
                self.prepare_length_df()
            self.fig_tok_length = make_fig_lengths(self.tokenized_df, LENGTH_FIELD)

    def load_or_prepare_embeddings(self):
        self.embeddings = Embeddings(self, use_cache=self.use_cache)
        self.embeddings.make_hierarchical_clustering()
        self.node_list = self.embeddings.node_list
        self.fig_tree = self.embeddings.fig_tree

    # get vocab with word counts
    def load_or_prepare_vocab(self, save=True):
        """
        Calculates the vocabulary count from the tokenized text.
        The resulting dataframes may be used in nPMI calculations, zipf, etc.
        :param
        :return:
        """
        if self.use_cache and exists(self.vocab_counts_df_fid):
            logs.info("Reading vocab from cache")
            self.load_vocab()
            self.vocab_counts_filtered_df = filter_vocab(self.vocab_counts_df)
        else:
            logs.info("Calculating vocab afresh")
            if self.tokenized_df is None:
                self.tokenized_df = self.do_tokenization()
                if save:
                    logs.info("Writing out.")
                    utils.write_df(self.tokenized_df, self.tokenized_df_fid)
            word_count_df = count_vocab_frequencies(self.tokenized_df)
            logs.info("Making dfs with proportion.")
            self.vocab_counts_df = calc_p_word(word_count_df)
            self.vocab_counts_filtered_df = filter_vocab(self.vocab_counts_df)
            if save:
                logs.info("Writing out.")
                utils.write_df(self.vocab_counts_df, self.vocab_counts_df_fid)
        logs.info("unfiltered vocab")
        logs.info(self.vocab_counts_df)
        logs.info("filtered vocab")
        logs.info(self.vocab_counts_filtered_df)

    def load_vocab(self):
        with open(self.vocab_counts_df_fid, "rb") as f:
            self.vocab_counts_df = utils.read_df(f)
        # Handling for changes in how the index is saved.
        self.vocab_counts_df = _set_idx_col_names(self.vocab_counts_df)

    def load_or_prepare_text_duplicates(self, save=True):
        if self.use_cache and exists(self.dup_counts_df_fid):
            with open(self.dup_counts_df_fid, "rb") as f:
                self.dup_counts_df = utils.read_df(f)
        elif self.dup_counts_df is None:
            if not self.live:
                self.prepare_text_duplicates()
                if save:
                    utils.write_df(self.dup_counts_df, self.dup_counts_df_fid)
        else:
            if not self.live:
                # This happens when self.dup_counts_df is already defined;
                # This happens when general_statistics were calculated first,
                # since general statistics requires the number of duplicates
                if save:
                    utils.write_df(self.dup_counts_df, self.dup_counts_df_fid)

    def load_or_prepare_text_perplexities(self, save=True):
        if self.use_cache and exists(self.perplexities_df_fid):
            with open(self.perplexities_df_fid, "rb") as f:
                self.perplexities_df = utils.read_df(f)
        elif self.perplexities_df is None:
            if not self.live:
                self.prepare_text_perplexities()
                if save:
                    utils.write_df(self.perplexities_df, self.perplexities_df_fid)
        else:
            if not self.live:
                if save:
                    utils.write_df(self.perplexities_df, self.perplexities_df_fid)

    def load_general_stats(self):
        self.general_stats_dict = json.load(
            open(self.general_stats_json_fid, encoding="utf-8")
        )
        with open(self.sorted_top_vocab_df_fid, "rb") as f:
            self.sorted_top_vocab_df = utils.read_df(f)
        self.text_nan_count = self.general_stats_dict[TEXT_NAN_CNT]
        self.dedup_total = self.general_stats_dict[DEDUP_TOT]
        self.total_words = self.general_stats_dict[TOT_WORDS]
        self.total_open_words = self.general_stats_dict[TOT_OPEN_WORDS]

    def prepare_general_stats(self):
        if not self.live:
            if self.tokenized_df is None:
                logs.warning("Tokenized dataset not yet loaded; doing so.")
                self.load_or_prepare_tokenized_df()
            if self.vocab_counts_df is None:
                logs.warning("Vocab not yet loaded; doing so.")
                self.load_or_prepare_vocab()
            self.sorted_top_vocab_df = self.vocab_counts_filtered_df.sort_values(
                "count", ascending=False
            ).head(_TOP_N)
            self.total_words = len(self.vocab_counts_df)
            self.total_open_words = len(self.vocab_counts_filtered_df)
            self.text_nan_count = int(self.tokenized_df.isnull().sum().sum())
            self.prepare_text_duplicates()
            self.dedup_total = sum(self.dup_counts_df[CNT])
            self.general_stats_dict = {
                TOT_WORDS: self.total_words,
                TOT_OPEN_WORDS: self.total_open_words,
                TEXT_NAN_CNT: self.text_nan_count,
                DEDUP_TOT: self.dedup_total,
            }

    def prepare_text_duplicates(self):
        if not self.live:
            if self.tokenized_df is None:
                self.load_or_prepare_tokenized_df()
            dup_df = self.tokenized_df[self.tokenized_df.duplicated([OUR_TEXT_FIELD])]
            self.dup_counts_df = pd.DataFrame(
                dup_df.pivot_table(
                    columns=[OUR_TEXT_FIELD], aggfunc="size"
                ).sort_values(ascending=False),
                columns=[CNT],
            )
            self.dup_counts_df[OUR_TEXT_FIELD] = self.dup_counts_df.index.copy()

    def prepare_text_perplexities(self):
        if not self.live:
            if self.text_dset is None:
                self.load_or_prepare_text_dset()
            results = _PERPLEXITY.compute(input_texts=self.text_dset[OUR_TEXT_FIELD], model_id='gpt2')
            perplexities = {PERPLEXITY_FIELD: results["perplexities"], OUR_TEXT_FIELD: self.text_dset[OUR_TEXT_FIELD]}
            self.perplexities_df = pd.DataFrame(perplexities).sort_values(by=PERPLEXITY_FIELD, ascending=False)

    def load_or_prepare_dataset(self, save=True):
        """
        Prepares the HF datasets and data frames containing the untokenized and
        tokenized text as well as the label values.
        self.tokenized_df is used further for calculating text lengths,
        word counts, etc.
        Args:
            save: Store the calculated data to disk.

        Returns:

        """
        logs.info("Doing text dset.")
        self.load_or_prepare_text_dset(save)
        #logs.info("Doing tokenized dataframe")
        #self.load_or_prepare_tokenized_df(save)
        logs.info("Doing dataset peek")
        self.load_or_prepare_dset_peek(save)

    def load_or_prepare_dset_peek(self, save=True):
        if self.use_cache and exists(self.dset_peek_json_fid):
            with open(self.dset_peek_json_fid, "r") as f:
                self.dset_peek = json.load(f)["dset peek"]
        else:
            if not self.live:
                if self.dset is None:
                    self.get_base_dataset()
                self.dset_peek = self.dset[:100]
                if save:
                    utils.write_json({"dset peek": self.dset_peek}, self.dset_peek_json_fid)

    def load_or_prepare_tokenized_df(self, save=True):
        if self.use_cache and exists(self.tokenized_df_fid):
            self.tokenized_df = utils.read_df(self.tokenized_df_fid)
        else:
            if not self.live:
                # tokenize all text instances
                self.tokenized_df = self.do_tokenization()
                if save:
                    logs.warning("Saving tokenized dataset to disk")
                    # save tokenized text
                    utils.write_df(self.tokenized_df, self.tokenized_df_fid)

    def load_or_prepare_text_dset(self, save=True):
        if self.use_cache and exists(self.text_dset_fid):
            # load extracted text
            self.text_dset = load_from_disk(self.text_dset_fid)
            logs.warning("Loaded dataset from disk")
            logs.info(self.text_dset)
        # ...Or load it from the server and store it anew
        else:
            if not self.live:
                self.prepare_text_dset()
                if save:
                    # save extracted text instances
                    logs.warning("Saving dataset to disk")
                    self.text_dset.save_to_disk(self.text_dset_fid)

    def prepare_text_dset(self):
        if not self.live:
            self.get_base_dataset()
            # extract all text instances
            self.text_dset = self.dset.map(
                lambda examples: utils.extract_field(
                    examples, self.text_field, OUR_TEXT_FIELD
                ),
                batched=True,
                remove_columns=list(self.dset.features),
            )

    def do_tokenization(self):
        """
        Tokenizes the dataset
        :return:
        """
        if self.text_dset is None:
            self.load_or_prepare_text_dset()
        sent_tokenizer = self.cvec.build_tokenizer()

        def tokenize_batch(examples):
            # TODO: lowercase should be an option
            res = {
                TOKENIZED_FIELD: [
                    tuple(sent_tokenizer(text.lower()))
                    for text in examples[OUR_TEXT_FIELD]
                ]
            }
            res[LENGTH_FIELD] = [len(tok_text) for tok_text in res[TOKENIZED_FIELD]]
            return res

        tokenized_dset = self.text_dset.map(
            tokenize_batch,
            batched=True,
            # remove_columns=[OUR_TEXT_FIELD], keep around to print
        )
        tokenized_df = pd.DataFrame(tokenized_dset)
        return tokenized_df

    def set_label_field(self, label_field="label"):
        """
        Setter for label_field. Used in the CLI when a user asks for information
         about labels, but does not specify the field;
         'label' is assumed as a default.
        """
        self.label_field = label_field

    def load_or_prepare_labels(self, save=True):
        # TODO: This is in a transitory state for creating fig cache.
        # Clean up to be caching and reading everything correctly.
        """
        Extracts labels from the Dataset
        :return:
        """
        # extracted labels
        if len(self.label_field) > 0:
            if self.use_cache and exists(self.fig_labels_json_fid):
                self.fig_labels = utils.read_plotly(self.fig_labels_json_fid)
            elif self.use_cache and exists(self.label_dset_fid):
                # load extracted labels
                self.label_dset = load_from_disk(self.label_dset_fid)
                self.label_df = self.label_dset.to_pandas()
                self.fig_labels = make_fig_labels(
                    self.label_df, self.label_names, OUR_LABEL_FIELD
                )
                if save:
                    utils.write_plotly(self.fig_labels, self.fig_labels_json_fid)
            else:
                if not self.live:
                    self.prepare_labels()
                    if save:
                        # save extracted label instances
                        self.label_dset.save_to_disk(self.label_dset_fid)
                        utils.write_plotly(self.fig_labels, self.fig_labels_json_fid)

    def prepare_labels(self):
        if not self.live:
            self.get_base_dataset()
            self.label_dset = self.dset.map(
                lambda examples: utils.extract_field(
                    examples, self.label_field, OUR_LABEL_FIELD
                ),
                batched=True,
                remove_columns=list(self.dset.features),
            )
            self.label_df = self.label_dset.to_pandas()
            self.fig_labels = make_fig_labels(
                self.label_df, self.label_names, OUR_LABEL_FIELD
            )

    def load_or_prepare_npmi(self):
        self.npmi_stats = nPMIStatisticsCacheClass(self, use_cache=self.use_cache)
        self.npmi_stats.load_or_prepare_npmi_terms()

    def load_or_prepare_zipf(self, save=True):
        zipf_json_fid, zipf_fig_json_fid, zipf_fig_html_fid = get_zipf_fids(
            self.cache_path)
        if self.use_cache:
            # Zipf statistics
            if exists(zipf_json_fid):
                # Read Zipf statistics: Alpha, p-value, etc.
                with open(zipf_json_fid, "r") as f:
                    zipf_dict = json.load(f)
                self.z = Zipf(None)
                self.z.load(zipf_dict)
            # Zipf figure
            if exists(zipf_fig_json_fid):
                self.zipf_fig = utils.read_plotly(zipf_fig_json_fid)
            elif self.z:
                # If the figure doesn't exist, but the object does, make the figure.
                # (Happens if just the figure file got deleted).
                self.zipf_fig = make_zipf_fig(self.vocab_counts_df, self.z)
                # TODO: Save the figure
            else:
                # Cache files do not exist.
                self.prepare_zipf(save)
        else:
            self.prepare_zipf(save)

    def prepare_zipf(self, save=True):
        # Calculate zipf from scratch
        # TODO: Does z even need to be self?
        self.z = Zipf(self.vocab_counts_df)
        self.z.calc_fit()
        self.zipf_fig = make_zipf_fig(self.z)
        if save:
            zipf_dict = self.z.get_zipf_dict()
            zipf_json_fid, zipf_fig_fid, zipf_fig_html_fid = get_zipf_fids(self.cache_path)
            utils.write_json(zipf_dict, zipf_json_fid)
            utils.write_plotly(self.zipf_fig, zipf_fig_fid)
            self.zipf_fig.write_html(zipf_fig_html_fid)

def _set_idx_col_names(input_vocab_df):
    if input_vocab_df.index.name != VOCAB and VOCAB in input_vocab_df.columns:
        input_vocab_df = input_vocab_df.set_index([VOCAB])
        input_vocab_df[VOCAB] = input_vocab_df.index
    return input_vocab_df

def _set_idx_cols_from_cache(csv_df, subgroup=None, calc_str=None):
    """
    Helps make sure all of the read-in files can be accessed within code
    via standardized indices and column names.
    :param csv_df:
    :param subgroup:
    :param calc_str:
    :return:
    """
    # The csv saves with this column instead of the index, so that's weird.
    if "Unnamed: 0" in csv_df.columns:
        csv_df = csv_df.set_index("Unnamed: 0")
        csv_df.index.name = WORD
    elif WORD in csv_df.columns:
        csv_df = csv_df.set_index(WORD)
        csv_df.index.name = WORD
    elif VOCAB in csv_df.columns:
        csv_df = csv_df.set_index(VOCAB)
        csv_df.index.name = WORD
    if subgroup and calc_str:
        csv_df.columns = [subgroup + "-" + calc_str]
    elif subgroup:
        csv_df.columns = [subgroup]
    elif calc_str:
        csv_df.columns = [calc_str]
    return csv_df


class nPMIStatisticsCacheClass:
    """ "Class to interface between the app and the nPMI class
    by calling the nPMI class with the user's selections."""

    def __init__(self, dataset_stats, use_cache=False):
        self.live = dataset_stats.live
        self.dstats = dataset_stats
        self.pmi_cache_path = pjoin(self.dstats.cache_path, "pmi_files")
        if not isdir(self.pmi_cache_path):
            logs.warning("Creating pmi cache directory %s." % self.pmi_cache_path)
            # We need to preprocess everything.
            mkdir(self.pmi_cache_path)
        self.joint_npmi_df_dict = {}
        # TODO: Users ideally can type in whatever words they want.
        self.termlist = _IDENTITY_TERMS
        # termlist terms that are available more than _MIN_VOCAB_COUNT times
        self.available_terms = _IDENTITY_TERMS
        logs.info(self.termlist)
        self.use_cache = use_cache
        # TODO: Let users specify
        self.open_class_only = True
        self.min_vocab_count = self.dstats.min_vocab_count
        self.subgroup_files = {}
        self.npmi_terms_fid = pjoin(self.dstats.cache_path, "npmi_terms.json")

    def load_or_prepare_npmi_terms(self):
        """
        Figures out what identity terms the user can select, based on whether
        they occur more than self.min_vocab_count times
        :return: Identity terms occurring at least self.min_vocab_count times.
        """
        # TODO: Add the user's ability to select subgroups.
        # TODO: Make min_vocab_count here value selectable by the user.
        if (
            self.use_cache
            and exists(self.npmi_terms_fid)
            and json.load(open(self.npmi_terms_fid))["available terms"] != []
        ):
            available_terms = json.load(open(self.npmi_terms_fid))["available terms"]
        else:
            true_false = [
                term in self.dstats.vocab_counts_df.index for term in self.termlist
            ]
            word_list_tmp = [x for x, y in zip(self.termlist, true_false) if y]
            true_false_counts = [
                self.dstats.vocab_counts_df.loc[word, CNT] >= self.min_vocab_count
                for word in word_list_tmp
            ]
            available_terms = [
                word for word, y in zip(word_list_tmp, true_false_counts) if y
            ]
            logs.info(available_terms)
            with open(self.npmi_terms_fid, "w+") as f:
                json.dump({"available terms": available_terms}, f)
        self.available_terms = available_terms
        return available_terms

    def load_or_prepare_joint_npmi(self, subgroup_pair):
        """
        Run on-the fly, while the app is already open,
        as it depends on the subgroup terms that the user chooses
        :param subgroup_pair:
        :return:
        """
        # Canonical ordering for subgroup_list
        subgroup_pair = sorted(subgroup_pair)
        subgroup1 = subgroup_pair[0]
        subgroup2 = subgroup_pair[1]
        subgroups_str = "-".join(subgroup_pair)
        if not isdir(self.pmi_cache_path):
            logs.warning("Creating cache")
            # We need to preprocess everything.
            # This should eventually all go into a prepare_dataset CLI
            mkdir(self.pmi_cache_path)
        joint_npmi_fid = pjoin(self.pmi_cache_path, subgroups_str + "_npmi.csv")
        subgroup_files = define_subgroup_files(subgroup_pair, self.pmi_cache_path)
        # Defines the filenames for the cache files from the selected subgroups.
        # Get as much precomputed data as we can.
        if self.use_cache and exists(joint_npmi_fid):
            # When everything is already computed for the selected subgroups.
            logs.info("Loading cached joint npmi")
            joint_npmi_df = self.load_joint_npmi_df(joint_npmi_fid)
            npmi_display_cols = [
                "npmi-bias",
                subgroup1 + "-npmi",
                subgroup2 + "-npmi",
                subgroup1 + "-count",
                subgroup2 + "-count",
            ]
            joint_npmi_df = joint_npmi_df[npmi_display_cols]
            # When maybe some things have been computed for the selected subgroups.
        else:
            if not self.live:
                logs.info("Preparing new joint npmi")
                joint_npmi_df, subgroup_dict = self.prepare_joint_npmi_df(
                    subgroup_pair, subgroup_files
                )
                # Cache new results
                logs.info("Writing out.")
                for subgroup in subgroup_pair:
                    write_subgroup_npmi_data(subgroup, subgroup_dict, subgroup_files)
                with open(joint_npmi_fid, "w+") as f:
                    joint_npmi_df.to_csv(f)
            else:
                joint_npmi_df = pd.DataFrame()
        logs.info("The joint npmi df is")
        logs.info(joint_npmi_df)
        return joint_npmi_df

    @staticmethod
    def load_joint_npmi_df(joint_npmi_fid):
        """
        Reads in a saved dataframe with all of the paired results.
        :param joint_npmi_fid:
        :return: paired results
        """
        with open(joint_npmi_fid, "rb") as f:
            joint_npmi_df = pd.read_csv(f)
        joint_npmi_df = _set_idx_cols_from_cache(joint_npmi_df)
        return joint_npmi_df.dropna()

    def prepare_joint_npmi_df(self, subgroup_pair, subgroup_files):
        """
        Computs the npmi bias based on the given subgroups.
        Handles cases where some of the selected subgroups have cached nPMI
        computations, but other's don't, computing everything afresh if there
        are not cached files.
        :param subgroup_pair:
        :return: Dataframe with nPMI for the words, nPMI bias between the words.
        """
        subgroup_dict = {}
        # When npmi is computed for some (but not all) of subgroup_list
        for subgroup in subgroup_pair:
            logs.info("Load or failing...")
            # When subgroup npmi has been computed in a prior session.
            cached_results = self.load_or_fail_cached_npmi_scores(
                subgroup, subgroup_files[subgroup]
            )
            # If the function did not return False and we did find it, use.
            if cached_results:
                # FYI: subgroup_cooc_df, subgroup_pmi_df, subgroup_npmi_df = cached_results
                # Holds the previous sessions' data for use in this session.
                subgroup_dict[subgroup] = cached_results
        logs.info("Calculating for subgroup list")
        joint_npmi_df, subgroup_dict = self.do_npmi(subgroup_pair, subgroup_dict)
        return joint_npmi_df.dropna(), subgroup_dict

    # TODO: Update pairwise assumption
    def do_npmi(self, subgroup_pair, subgroup_dict):
        """
        Calculates nPMI for given identity terms and the nPMI bias between.
        :param subgroup_pair: List of identity terms to calculate the bias for
        :return: Subset of data for the UI
        :return: Selected identity term's co-occurrence counts with
                 other words, pmi per word, and nPMI per word.
        """
        logs.info("Initializing npmi class")
        npmi_obj = self.set_npmi_obj()
        # Canonical ordering used
        subgroup_pair = tuple(sorted(subgroup_pair))
        # Calculating nPMI statistics
        for subgroup in subgroup_pair:
            # If the subgroup data is already computed, grab it.
            # TODO: Should we set idx and column names similarly to how we set them for cached files?
            if subgroup not in subgroup_dict:
                logs.info("Calculating statistics for %s" % subgroup)
                vocab_cooc_df, pmi_df, npmi_df = npmi_obj.calc_metrics(subgroup)
                # Store the nPMI information for the current subgroups
                subgroup_dict[subgroup] = (vocab_cooc_df, pmi_df, npmi_df)
        # Pair the subgroups together, indexed by all words that
        # co-occur between them.
        logs.info("Computing pairwise npmi bias")
        paired_results = npmi_obj.calc_paired_metrics(subgroup_pair, subgroup_dict)
        UI_results = make_npmi_fig(paired_results, subgroup_pair)
        return UI_results, subgroup_dict

    def set_npmi_obj(self):
        """
        Initializes the nPMI class with the given words and tokenized sentences.
        :return:
        """
        # TODO(meg): Incorporate this from evaluate library.
        # npmi_obj = evaluate.load('npmi', module_type='measurement').compute(subgroup, vocab_counts_df = self.dstats.vocab_counts_df, tokenized_counts_df=self.dstats.tokenized_df)
        npmi_obj = nPMI(self.dstats.vocab_counts_df, self.dstats.tokenized_df)
        return npmi_obj

    @staticmethod
    def load_or_fail_cached_npmi_scores(subgroup, subgroup_fids):
        """
        Reads cached scores from the specified subgroup files
        :param subgroup: string of the selected identity term
        :return:
        """
        # TODO: Ordering of npmi, pmi, vocab triple should be consistent
        subgroup_npmi_fid, subgroup_pmi_fid, subgroup_cooc_fid = subgroup_fids
        if (
            exists(subgroup_npmi_fid)
            and exists(subgroup_pmi_fid)
            and exists(subgroup_cooc_fid)
        ):
            logs.info("Reading in pmi data....")
            with open(subgroup_cooc_fid, "rb") as f:
                subgroup_cooc_df = pd.read_csv(f)
            logs.info("pmi")
            with open(subgroup_pmi_fid, "rb") as f:
                subgroup_pmi_df = pd.read_csv(f)
            logs.info("npmi")
            with open(subgroup_npmi_fid, "rb") as f:
                subgroup_npmi_df = pd.read_csv(f)
            subgroup_cooc_df = _set_idx_cols_from_cache(
                subgroup_cooc_df, subgroup, "count"
            )
            subgroup_pmi_df = _set_idx_cols_from_cache(
                subgroup_pmi_df, subgroup, "pmi"
            )
            subgroup_npmi_df = _set_idx_cols_from_cache(
                subgroup_npmi_df, subgroup, "npmi"
            )
            return subgroup_cooc_df, subgroup_pmi_df, subgroup_npmi_df
        return False

    def get_available_terms(self):
        return self.load_or_prepare_npmi_terms()


def dummy(doc):
    return doc


def count_vocab_frequencies(tokenized_df):
    """
    Based on an input pandas DataFrame with a 'text' column,
    this function will count the occurrences of all words.
    :return: [num_words x num_sentences] DataFrame with the rows corresponding to the
    different vocabulary words and the column to the presence (0 or 1) of that word.
    """

    cvec = CountVectorizer(
        tokenizer=dummy,
        preprocessor=dummy,
    )
    # We do this to calculate per-word statistics
    # Fast calculation of single word counts
    logs.info(
        "Fitting dummy tokenization to make matrix using the previous tokenization"
    )
    cvec.fit(tokenized_df[TOKENIZED_FIELD])
    document_matrix = cvec.transform(tokenized_df[TOKENIZED_FIELD])
    batches = np.linspace(0, tokenized_df.shape[0], _NUM_VOCAB_BATCHES).astype(int)
    i = 0
    tf = []
    while i < len(batches) - 1:
        logs.info("%s of %s vocab batches" % (str(i), str(len(batches))))
        batch_result = np.sum(
            document_matrix[batches[i] : batches[i + 1]].toarray(), axis=0
        )
        tf.append(batch_result)
        i += 1
    word_count_df = pd.DataFrame(
        [np.sum(tf, axis=0)], columns=cvec.get_feature_names()
    ).transpose()
    # Now organize everything into the dataframes
    word_count_df.columns = [CNT]
    word_count_df.index.name = WORD
    return word_count_df


def calc_p_word(word_count_df):
    # p(word)
    word_count_df[PROP] = word_count_df[CNT] / float(sum(word_count_df[CNT]))
    vocab_counts_df = pd.DataFrame(word_count_df.sort_values(by=CNT, ascending=False))
    vocab_counts_df[VOCAB] = vocab_counts_df.index
    return vocab_counts_df


def filter_vocab(vocab_counts_df):
    # TODO: Add warnings (which words are missing) to log file?
    filtered_vocab_counts_df = vocab_counts_df.drop(_CLOSED_CLASS, errors="ignore")
    filtered_count = filtered_vocab_counts_df[CNT]
    filtered_count_denom = float(sum(filtered_vocab_counts_df[CNT]))
    filtered_vocab_counts_df[PROP] = filtered_count / filtered_count_denom
    return filtered_vocab_counts_df


## Figures ##

def make_fig_lengths(tokenized_df, length_field):
    fig_tok_length, axs = plt.subplots(figsize=(15, 6), dpi=150)
    sns.histplot(data=tokenized_df[length_field], kde=True, bins=100, ax=axs)
    sns.rugplot(data=tokenized_df[length_field], ax=axs)
    return fig_tok_length


def make_fig_labels(label_df, label_names, label_field):
    labels = label_df[label_field].unique()
    label_sums = [len(label_df[label_df[label_field] == label]) for label in labels]
    fig_labels = px.pie(label_df, values=label_sums, names=label_names)
    return fig_labels


def make_npmi_fig(paired_results, subgroup_pair):
    subgroup1, subgroup2 = subgroup_pair
    UI_results = pd.DataFrame()
    if "npmi-bias" in paired_results:
        UI_results["npmi-bias"] = paired_results["npmi-bias"].astype(float)
    UI_results[subgroup1 + "-npmi"] = paired_results["npmi"][
        subgroup1 + "-npmi"
    ].astype(float)
    UI_results[subgroup1 + "-count"] = paired_results["count"][
        subgroup1 + "-count"
    ].astype(int)
    if subgroup1 != subgroup2:
        UI_results[subgroup2 + "-npmi"] = paired_results["npmi"][
            subgroup2 + "-npmi"
        ].astype(float)
        UI_results[subgroup2 + "-count"] = paired_results["count"][
            subgroup2 + "-count"
        ].astype(int)
    return UI_results.sort_values(by="npmi-bias", ascending=True)




## Input/Output ###


def define_subgroup_files(subgroup_list, pmi_cache_path):
    """
    Sets the file ids for the input identity terms
    :param subgroup_list: List of identity terms
    :return:
    """
    subgroup_files = {}
    for subgroup in subgroup_list:
        # TODO: Should the pmi, npmi, and count just be one file?
        subgroup_npmi_fid = pjoin(pmi_cache_path, subgroup + "_npmi.csv")
        subgroup_pmi_fid = pjoin(pmi_cache_path, subgroup + "_pmi.csv")
        subgroup_cooc_fid = pjoin(pmi_cache_path, subgroup + "_vocab_cooc.csv")
        subgroup_files[subgroup] = (
            subgroup_npmi_fid,
            subgroup_pmi_fid,
            subgroup_cooc_fid,
        )
    return subgroup_files


## Input/Output ##


def intersect_dfs(df_dict):
    started = 0
    new_df = None
    for key, df in df_dict.items():
        if df is None:
            continue
        for key2, df2 in df_dict.items():
            if df2 is None:
                continue
            if key == key2:
                continue
            if started:
                new_df = new_df.join(df2, how="inner", lsuffix="1", rsuffix="2")
            else:
                new_df = df.join(df2, how="inner", lsuffix="1", rsuffix="2")
                started = 1
    return new_df.copy()


def write_subgroup_npmi_data(subgroup, subgroup_dict, subgroup_files):
    """
    Saves the calculated nPMI statistics to their output files.
    Includes the npmi scores for each identity term, the pmi scores, and the
    co-occurrence counts of the identity term with all the other words
    :param subgroup: Identity term
    :return:
    """
    subgroup_fids = subgroup_files[subgroup]
    subgroup_npmi_fid, subgroup_pmi_fid, subgroup_cooc_fid = subgroup_fids
    subgroup_dfs = subgroup_dict[subgroup]
    subgroup_cooc_df, subgroup_pmi_df, subgroup_npmi_df = subgroup_dfs
    with open(subgroup_npmi_fid, "w+") as f:
        subgroup_npmi_df.to_csv(f)
    with open(subgroup_pmi_fid, "w+") as f:
        subgroup_pmi_df.to_csv(f)
    with open(subgroup_cooc_fid, "w+") as f:
        subgroup_cooc_df.to_csv(f)


