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
from os import mkdir
from os.path import exists, isdir
from os.path import join as pjoin
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import pyarrow.feather as feather
from datasets import load_from_disk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

from .dataset_utils import (
    CNT,
    DEDUP_TOT,
    EMBEDDING_FIELD,
    LENGTH_FIELD,
    OUR_LABEL_FIELD,
    OUR_TEXT_FIELD,
    PROP,
    TEXT_NAN_CNT,
    TOKENIZED_FIELD,
    TXT_LEN,
    VOCAB,
    WORD,
    extract_field,
    load_truncated_dataset,
)
from .embeddings import Embeddings
from .npmi import nPMI
from .zipf import Zipf

pd.options.display.float_format = "{:,.3f}".format

logs = logging.getLogger(__name__)
logs.setLevel(logging.WARNING)
logs.propagate = False

if not logs.handlers:

    Path('./log_files').mkdir(exist_ok=True)

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


_CVEC = CountVectorizer(token_pattern="(?u)\\b\\w+\\b", lowercase=True)

num_rows = 200000


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
    ):
        # This is only used for standalone runs for each kind of measurement.
        self.calculation = calculation
        self.our_text_field = OUR_TEXT_FIELD
        self.our_length_field = LENGTH_FIELD
        self.our_label_field = OUR_LABEL_FIELD
        self.our_tokenized_field = TOKENIZED_FIELD
        self.our_embedding_field = EMBEDDING_FIELD
        self.cache_dir = cache_dir
        ### What are we analyzing?
        # name of the Hugging Face dataset
        self.dset_name = dset_name
        # name of the dataset config
        self.dset_config = dset_config
        # name of the split to analyze
        self.split_name = split_name
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
        # HF dataset with text embeddings in the same order as self.text_dset
        self.embeddings_dset = None
        # HF dataset with all of the self.label_field instances in self.dset
        self.label_dset = None
        ## Data frames
        # Tokenized text
        self.tokenized_df = []
        # save sentence length histogram in the class so it doesn't ge re-computed
        self.fig_tok_length = None
        # Data Frame version of self.label_dset
        self.label_df = None
        # save label pie chart in the class so it doesn't ge re-computed
        self.fig_labels = None
        # Vocabulary with word counts in the dataset
        self.vocab_counts_df = None
        # Vocabulary filtered to remove stopwords
        self.vocab_counts_filtered_df = None
        ## General statistics and duplicates
        # Number of NaN values (NOT empty strings)
        self.text_nan_count = 0
        # Number of text items that appear more than once in the dataset
        self.dedup_total = 0
        # Duplicated text items along with their number of occurences ("count")
        self.text_dup_counts_df = None
        self.avg_length = None
        self.std_length = None
        self.general_stats_dict = None
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
        # TODO: Users ideally can type in whatever words they want.
        self.termlist = _IDENTITY_TERMS
        # termlist terms that are available more than _MIN_VOCAB_COUNT times
        self.available_terms = _IDENTITY_TERMS
        # TODO: Have lowercase be an option for a user to set.
        self.to_lowercase = True
        # The minimum amount of times a word should occur to be included in
        # word-count-based calculations (currently just relevant to nPMI)
        self.min_vocab_count = _MIN_VOCAB_COUNT
        # zipf
        self.z = None
        self.zipf_fig = None
        self.cvec = _CVEC
        # File definitions
        # path to the directory used for caching
        if not isinstance(text_field, str):
            text_field = "-".join(text_field)
        if isinstance(label_field, str):
            label_field = label_field
        else:
            label_field = "-".join(label_field)
        self.cache_path = pjoin(
            self.cache_dir,
            f"{dset_name}_{dset_config}_{split_name}_{text_field}_{label_field}",
        )
        if not isdir(self.cache_path):
            logs.warning("Creating cache directory %s." % self.cache_path)
            mkdir(self.cache_path)
        self.dset_fid = pjoin(self.cache_path, "base_dset")
        self.text_dset_fid = pjoin(self.cache_path, "text_dset")
        self.tokenized_df_fid = pjoin(self.cache_path, "tokenized_df.feather")
        self.label_dset_fid = pjoin(self.cache_path, "label_dset")
        self.vocab_counts_df_fid = pjoin(self.cache_path, "vocab_counts.feather")
        self.general_stats_fid = pjoin(self.cache_path, "general_stats.json")
        self.text_duplicate_counts_df_fid = pjoin(
            self.cache_path, "text_dup_counts_df.feather"
        )
        self.zipf_fid = pjoin(self.cache_path, "zipf_basic_stats.json")

    def get_base_dataset(self):
        """Gets a pointer to the truncated base dataset object."""
        if not self.dset:
            self.dset = load_truncated_dataset(
                self.dset_name,
                self.dset_config,
                self.split_name,
                cache_name=self.dset_fid,
                use_cache=True,
                use_streaming=True,
            )

    def get_dataset_peek(self):
        self.get_base_dataset()
        return self.dset[:100]

    def load_or_prepare_general_stats(self, use_cache=False):
        """Data structures used in calculating general statistics and duplicates"""

        # TODO: These probably don't need to be feather files, could be csv.
        # General statistics
        if (
            use_cache
            and exists(self.general_stats_fid)
            and exists(self.text_duplicate_counts_df_fid)
        ):
            self.load_general_stats(
                self.general_stats_fid, self.text_duplicate_counts_df_fid
            )
        else:
            (
                self.text_nan_count,
                self.dedup_total,
                self.text_dup_counts_df,
            ) = self.prepare_general_text_stats()
            self.general_stats_dict = {
                TEXT_NAN_CNT: self.text_nan_count,
                DEDUP_TOT: self.dedup_total,
            }
            write_df(self.text_dup_counts_df, self.text_duplicate_counts_df_fid)
            write_json(self.general_stats_dict, self.general_stats_fid)

    def load_or_prepare_text_lengths(self, use_cache=False):
        if len(self.tokenized_df) == 0:
            self.tokenized_df = self.do_tokenization()
        self.tokenized_df[LENGTH_FIELD] = self.tokenized_df[TOKENIZED_FIELD].apply(len)
        self.avg_length = round(
            sum(self.tokenized_df[self.our_length_field])
            / len(self.tokenized_df[self.our_length_field]),
            1,
        )
        self.std_length = round(
            statistics.stdev(self.tokenized_df[self.our_length_field]), 1
        )
        self.fig_tok_length = make_fig_lengths(self.tokenized_df, self.our_length_field)

    def load_or_prepare_embeddings(self, use_cache=False):
        self.embeddings = Embeddings(self, use_cache=use_cache)
        self.embeddings.make_hierarchical_clustering()
        self.fig_tree = self.embeddings.fig_tree
        self.node_list = self.embeddings.node_list

    # get vocab with word counts
    def load_or_prepare_vocab(self, use_cache=True, save=True):
        """
        Calculates the vocabulary count from the tokenized text.
        The resulting dataframes may be used in nPMI calculations, zipf, etc.
        :param use_cache:
        :return:
        """
        if (
            use_cache
            and exists(self.vocab_counts_df_fid)
        ):
            logs.info("Reading vocab from cache")
            self.load_vocab()
            self.vocab_counts_filtered_df = filter_words(self.vocab_counts_df)
        else:
            logs.info("Calculating vocab afresh")
            if len(self.tokenized_df) == 0:
                self.tokenized_df = self.do_tokenization()
                if save:
                    logs.info("Writing out.")
                    write_df(self.tokenized_df, self.tokenized_df_fid)
            word_count_df = count_vocab_frequencies(self.tokenized_df)
            logs.info("Making dfs with proportion.")
            self.vocab_counts_df = calc_p_word(word_count_df)
            self.vocab_counts_filtered_df = filter_words(self.vocab_counts_df)
            if save:
                logs.info("Writing out.")
                write_df(self.vocab_counts_df, self.vocab_counts_df_fid)
        logs.info("unfiltered vocab")
        logs.info(self.vocab_counts_df)
        logs.info("filtered vocab")
        logs.info(self.vocab_counts_filtered_df)

    def load_or_prepare_npmi_terms(self, use_cache=False):
        self.npmi_stats = nPMIStatisticsCacheClass(self, use_cache=use_cache)
        self.npmi_stats.load_or_prepare_npmi_terms()

    def load_or_prepare_zipf(self, use_cache=False):
        if use_cache and exists(self.zipf_fid):
            # TODO: Read zipf data so that the vocab is there.
            with open(self.zipf_fid, "r") as f:
                zipf_dict = json.load(f)
            self.z = Zipf()
            self.z.load(zipf_dict)
        else:
            self.z = Zipf(self.vocab_counts_df)
            write_zipf_data(self.z, self.zipf_fid)
        self.zipf_fig = make_zipf_fig(self.vocab_counts_df, self.z)

    def prepare_general_text_stats(self):
        text_nan_count = int(self.tokenized_df.isnull().sum().sum())
        dup_df = self.tokenized_df[self.tokenized_df.duplicated([self.our_text_field])]
        dedup_df = pd.DataFrame(
            dup_df.pivot_table(
                columns=[self.our_text_field], aggfunc="size"
            ).sort_values(ascending=False),
            columns=[CNT],
        )
        dedup_df.index = dedup_df.index.map(str)
        dedup_df[OUR_TEXT_FIELD] = dedup_df.index
        dedup_total = sum(dedup_df[CNT])
        return text_nan_count, dedup_total, dedup_df

    def load_general_stats(self, general_stats_fid, text_duplicate_counts_df_fid):
        general_stats = json.load(open(general_stats_fid, encoding="utf-8"))
        self.text_nan_count = general_stats[TEXT_NAN_CNT]
        self.dedup_total = general_stats[DEDUP_TOT]
        with open(text_duplicate_counts_df_fid, "rb") as f:
            self.text_dup_counts_df = feather.read_feather(f)

    def load_or_prepare_dataset(self, use_cache=True, use_df=False, save=True):
        """
         Prepares the HF datasets and data frames containing the untokenized and tokenized
         text as well as the label values. If cache is not being used (use_cache=False), writes the datasets to text.
        :param use_cache:
        :param use_df: Whether to used stored dataframes rather than dset files
        :return:
        """
        ## Raw text first, then tokenization.
        # Use what has been previously stored in DataFrame form or Dataset form.
        if (
            use_cache
            and use_df
            and exists(self.tokenized_df_fid)
        ):
            self.tokenized_df = feather.read_feather(self.tokenized_df_fid)
        elif (
            use_cache and exists(self.text_dset_fid)):
            # load extracted text
            self.text_dset = load_from_disk(self.text_dset_fid)
            logs.warning("Loaded dataset from disk")
            logs.info(self.text_dset)
        # ...Or load it from the server and store it anew
        else:
            self.get_base_dataset()
            # extract all text instances
            self.text_dset = self.dset.map(
                lambda examples: extract_field(
                    examples, self.text_field, OUR_TEXT_FIELD
                ),
                batched=True,
                remove_columns=list(self.dset.features),
            )
            if save:
                # save extracted text instances
                logs.warning("Saving dataset to disk")
                self.text_dset.save_to_disk(self.text_dset_fid)
            # tokenize all text instances
            self.tokenized_df = self.do_tokenization()
            if save:
                # save tokenized text
                write_df(self.tokenized_df, self.tokenized_df_fid)

    def do_tokenization(self):
        """
        Tokenizes the dataset
        :return:
        """
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

    def load_or_prepare_labels(self, use_cache=False, save=True):
        """
        Extracts labels from the Dataset
        :param use_cache:
        :return:
        """
        # extracted labels
        if len(self.label_field) > 0:
            if use_cache and exists(self.label_dset_fid):
                # load extracted labels
                self.label_dset = load_from_disk(self.label_dset_fid)
            else:
                self.get_base_dataset()
                self.label_dset = self.dset.map(
                    lambda examples: extract_field(
                        examples, self.label_field, OUR_LABEL_FIELD
                    ),
                    batched=True,
                    remove_columns=list(self.dset.features),
                )
                if save:
                    # save extracted label instances
                    self.label_dset.save_to_disk(self.label_dset_fid)
            self.label_df = self.label_dset.to_pandas()

            self.fig_labels = make_fig_labels(
                self.label_df, self.label_names, OUR_LABEL_FIELD
            )

    def load_vocab(self):
        with open(self.vocab_counts_df_fid, "rb") as f:
            self.vocab_counts_df = feather.read_feather(f)
        # Handling for changes in how the index is saved.
        self.vocab_counts_df = self._set_idx_col_names(self.vocab_counts_df)

    def _set_idx_col_names(self, input_vocab_df):
        if input_vocab_df.index.name != VOCAB and VOCAB in input_vocab_df.columns:
            input_vocab_df = input_vocab_df.set_index([VOCAB])
            input_vocab_df[VOCAB] = input_vocab_df.index
        return input_vocab_df


class nPMIStatisticsCacheClass:
    """ "Class to interface between the app and the nPMI class
    by calling the nPMI class with the user's selections."""

    def __init__(self, dataset_stats, use_cache=False):
        self.dstats = dataset_stats
        self.pmi_cache_path = pjoin(self.dstats.cache_path, "pmi_files")
        if not isdir(self.pmi_cache_path):
            logs.warning("Creating pmi cache directory %s." % self.pmi_cache_path)
            # We need to preprocess everything.
            mkdir(self.pmi_cache_path)
        self.joint_npmi_df_dict = {}
        self.termlist = self.dstats.termlist
        logs.info(self.termlist)
        self.use_cache = use_cache
        # TODO: Let users specify
        self.open_class_only = True
        self.min_vocab_count = self.dstats.min_vocab_count
        self.subgroup_files = {}
        self.npmi_terms_fid = pjoin(self.dstats.cache_path, "npmi_terms.json")
        self.available_terms = self.dstats.available_terms
        logs.info(self.available_terms)

    def load_or_prepare_npmi_terms(self, use_cache=False):
        """
        Figures out what identity terms the user can select, based on whether
        they occur more than self.min_vocab_count times
        :param use_cache:
        :return: Identity terms occurring at least self.min_vocab_count times.
        """
        # TODO: Add the user's ability to select subgroups.
        # TODO: Make min_vocab_count here value selectable by the user.
        if (
            use_cache
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

    def load_or_prepare_joint_npmi(self, subgroup_pair, use_cache=True):
        """
        Run on-the fly, while the app is already open,
        as it depends on the subgroup terms that the user chooses
        :param subgroup_pair:
        :return:
        """
        # Canonical ordering for subgroup_list
        subgroup_pair = sorted(subgroup_pair)
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
        if use_cache and exists(joint_npmi_fid):
            # When everything is already computed for the selected subgroups.
            logs.info("Loading cached joint npmi")
            joint_npmi_df = self.load_joint_npmi_df(joint_npmi_fid)
            # When maybe some things have been computed for the selected subgroups.
        else:
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
        logs.info("The joint npmi df is")
        logs.info(joint_npmi_df)
        return joint_npmi_df

    def load_joint_npmi_df(self, joint_npmi_fid):
        """
        Reads in a saved dataframe with all of the paired results.
        :param joint_npmi_fid:
        :return: paired results
        """
        with open(joint_npmi_fid, "rb") as f:
            joint_npmi_df = pd.read_csv(f)
        joint_npmi_df = self._set_idx_cols_from_cache(joint_npmi_df)
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
        npmi_obj = nPMI(self.dstats.vocab_counts_df, self.dstats.tokenized_df)
        return npmi_obj

    def load_or_fail_cached_npmi_scores(self, subgroup, subgroup_fids):
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
            subgroup_cooc_df = self._set_idx_cols_from_cache(
                subgroup_cooc_df, subgroup, "count"
            )
            subgroup_pmi_df = self._set_idx_cols_from_cache(
                subgroup_pmi_df, subgroup, "pmi"
            )
            subgroup_npmi_df = self._set_idx_cols_from_cache(
                subgroup_npmi_df, subgroup, "npmi"
            )
            return subgroup_cooc_df, subgroup_pmi_df, subgroup_npmi_df
        return False

    def _set_idx_cols_from_cache(self, csv_df, subgroup=None, calc_str=None):
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

    def get_available_terms(self, use_cache=False):
        return self.load_or_prepare_npmi_terms(use_cache=use_cache)

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
    logs.info("Fitting dummy tokenization to make matrix using the previous tokenization")
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


def filter_words(vocab_counts_df):
    # TODO: Add warnings (which words are missing) to log file?
    filtered_vocab_counts_df = vocab_counts_df.drop(_CLOSED_CLASS,
                                                    errors="ignore")
    filtered_count = filtered_vocab_counts_df[CNT]
    filtered_count_denom = float(sum(filtered_vocab_counts_df[CNT]))
    filtered_vocab_counts_df[PROP] = filtered_count / filtered_count_denom
    return filtered_vocab_counts_df


## Figures ##


def make_fig_lengths(tokenized_df, length_field):
    fig_tok_length = px.histogram(
        tokenized_df, x=length_field, marginal="rug", hover_data=[length_field]
    )
    return fig_tok_length


def make_fig_labels(label_df, label_names, label_field):
    labels = label_df[label_field].unique()
    label_sums = [len(label_df[label_df[label_field] == label]) for label in labels]
    fig_labels = px.pie(label_df, values=label_sums, names=label_names)
    return fig_labels


def make_zipf_fig_ranked_word_list(vocab_df, unique_counts, unique_ranks):
    ranked_words = {}
    for count, rank in zip(unique_counts, unique_ranks):
        vocab_df[vocab_df[CNT] == count]["rank"] = rank
        ranked_words[rank] = ",".join(
            vocab_df[vocab_df[CNT] == count].index.astype(str)
        )  # Use the hovertext kw argument for hover text
    ranked_words_list = [wrds for rank, wrds in sorted(ranked_words.items())]
    return ranked_words_list


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


def make_zipf_fig(vocab_counts_df, z):
    zipf_counts = z.calc_zipf_counts(vocab_counts_df)
    unique_counts = z.uniq_counts
    unique_ranks = z.uniq_ranks
    ranked_words_list = make_zipf_fig_ranked_word_list(
        vocab_counts_df, unique_counts, unique_ranks
    )
    zmin = z.get_xmin()
    logs.info("zipf counts is")
    logs.info(zipf_counts)
    layout = go.Layout(xaxis=dict(range=[0, 100]))
    fig = go.Figure(
        data=[
            go.Bar(
                x=z.uniq_ranks,
                y=z.uniq_counts,
                hovertext=ranked_words_list,
                name="Word Rank Frequency",
            )
        ],
        layout=layout,
    )
    fig.add_trace(
        go.Scatter(
            x=z.uniq_ranks[zmin : len(z.uniq_ranks)],
            y=zipf_counts[zmin : len(z.uniq_ranks)],
            hovertext=ranked_words_list[zmin : len(z.uniq_ranks)],
            line=go.scatter.Line(color="crimson", width=3),
            name="Zipf Predicted Frequency",
        )
    )
    # Customize aspect
    # fig.update_traces(marker_color='limegreen',
    #                  marker_line_width=1.5, opacity=0.6)
    fig.update_layout(title_text="Word Counts, Observed and Predicted by Zipf")
    fig.update_layout(xaxis_title="Word Rank")
    fig.update_layout(yaxis_title="Frequency")
    fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.10))
    return fig


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


def write_df(df, df_fid):
    feather.write_feather(df, df_fid)


def write_json(json_dict, json_fid):
    with open(json_fid, "w", encoding="utf-8") as f:
        json.dump(json_dict, f)


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


def write_zipf_data(z, zipf_fid):
    zipf_dict = {}
    zipf_dict["xmin"] = int(z.xmin)
    zipf_dict["xmax"] = int(z.xmax)
    zipf_dict["alpha"] = float(z.alpha)
    zipf_dict["ks_distance"] = float(z.distance)
    zipf_dict["p-value"] = float(z.ks_test.pvalue)
    zipf_dict["uniq_counts"] = [int(count) for count in z.uniq_counts]
    zipf_dict["uniq_ranks"] = [int(rank) for rank in z.uniq_ranks]
    with open(zipf_fid, "w+", encoding="utf-8") as f:
        json.dump(zipf_dict, f)
