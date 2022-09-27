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
import sys
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import statistics
import utils
import utils.dataset_utils as ds_utils
from data_measurements.tokenize import Tokenize
from data_measurements.labels import labels
from data_measurements.lengths import lengths
from data_measurements.npmi import npmi
from data_measurements.perplexity import perplexity
from data_measurements.text_duplicates import text_duplicates as td
from data_measurements.npmi import npmi
from data_measurements.zipf import zipf
from datasets import load_from_disk
from nltk.corpus import stopwords
from os import mkdir, getenv
from os.path import exists, isdir
from os.path import join as pjoin
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
# Unfortunately, caching computations for the streamlit UI requires bringing st
# into the main data statistics class, which is separate from the UI.
import streamlit as st

from utils.dataset_utils import (CNT, TEXT_FIELD, PROP, TEXT_NAN_CNT,
                                 TOKENIZED_FIELD, TOT_OPEN_WORDS, TOT_WORDS,
                                 VOCAB, WORD)

logs = utils.prepare_logging(__file__)

# TODO: Read this in depending on chosen language / expand beyond english
nltk.download("stopwords", quiet=True)
_CLOSED_CLASS = (
        stopwords.words("english")
        + ["t", "n", "ll", "d", "s"]
        + ["wasn", "weren", "won", "aren", "wouldn", "shouldn", "didn", "don",
           "hasn", "ain", "couldn", "doesn", "hadn", "haven", "isn", "mightn",
           "mustn", "needn", "shan", "would", "could", "dont"]
        + [str(i) for i in range(0, 99)]
)
IDENTITY_TERMS = ["man", "woman", "non-binary", "gay", "lesbian", "queer",
                  "trans", "straight", "cis", "she", "her", "hers", "he", "him",
                  "his", "they", "them", "their", "theirs", "himself", "herself"]
# treating inf values as NaN as well
pd.set_option("use_inf_as_na", True)

MIN_VOCAB_COUNT = 10
_NUM_VOCAB_BATCHES = 2000
_TOP_N = 100


class DatasetStatisticsCacheClass:

    def __init__(self, dset_name, dset_config,split_name, text_field, 
                 label_field, label_names, cache_dir="cache_dir",
                 dset_cache_dir=None, use_cache=False, save=True):
        ### What are we analyzing?
        # name of the Hugging Face dataset
        self.dset_name = dset_name
        # name of the dataset config
        self.dset_config = dset_config
        # name of the split to analyze
        self.split_name = split_name
        # which text/feature fields are we analysing?
        self.text_field = text_field
        ## Label variables
        # which label fields are we analysing?
        self.label_field = label_field
        # what are the names of the classes?
        self.label_names = label_names
        ## Hugging Face dataset objects
        self.dset = None  # original dataset
        # HF dataset with all of the self.text_field instances in self.dset
        self.text_dset = None
        self.dset_peek = None
        ## Data frames
        # Tokenized text
        self.tokenized_df = None

        # Widgets:
        self.length_obj = None
        self.label_obj = None

        ## Caching
        if not dset_cache_dir:
            dataset_cache_name, self.dset_cache_dir = ds_utils.get_cache_dir_naming(cache_dir,
                                                                      dset_name,
                                                                      dset_config,
                                                                      split_name,
                                                                      text_field)
        else:
            self.dset_cache_dir = dset_cache_dir

        # Use stored data if there; otherwise calculate afresh
        self.use_cache = use_cache
        # Save newly calculated results.
        self.save = save

        self.dset_peek = None
        # Tokenized text
        self.tokenized_df = None

        ## Zipf
        # Save zipf fig so it doesn't need to be recreated.
        self.zipf_fig = None
        # Zipf object
        self.z = None

        ## Vocabulary
        # Vocabulary with word counts in the dataset
        self.vocab_counts_df = None
        # Vocabulary filtered to remove stopwords
        self.vocab_counts_filtered_df = None
        self.sorted_top_vocab_df = None

        # Text Duplicates
        self.duplicates_results = None
        self.duplicates_files = {}
        self.dups_frac = 0
        self.dups_dict = {}

        ## Perplexity
        self.perplexities_df = None

        ## Lengths
        self.avg_length = None
        self.std_length = None
        self.length_stats_dict = None
        self.length_df = None
        self.num_uniq_lengths = 0

        ## "General" stats
        self.general_stats_dict = {}
        self.total_words = 0
        self.total_open_words = 0
        # Number of NaN values (NOT empty strings)
        self.text_nan_count = 0

        # nPMI
        self.npmi_obj = None
        # The minimum amount of times a word should occur to be included in
        # word-count-based calculations (currently just relevant to nPMI)
        self.min_vocab_count = MIN_VOCAB_COUNT

        # Files used in the DMT for the basic info, before fancier modules.
        self.hf_dset_cache_dir = pjoin(self.dset_cache_dir, "base_dset")
        self.text_dset_fid = pjoin(self.dset_cache_dir, "text_dset")
        self.tokenized_df_fid = pjoin(self.dset_cache_dir, "tokenized_df.json")
        self.dset_peek_json_fid = pjoin(self.dset_cache_dir, "dset_peek.json")
        self.vocab_counts_df_fid = pjoin(self.dset_cache_dir,
                                         "vocab_counts.json")
        self.general_stats_json_fid = pjoin(self.dset_cache_dir,
                                            "general_stats_dict.json")
        # Needed for UI
        self.sorted_top_vocab_df_fid = pjoin(
            self.dset_cache_dir, "sorted_top_vocab.json"
        )

        # Set the HuggingFace dataset object with the given arguments.
        self.dset = self._get_dataset()
        self.text_dset = None
        # Defines self.text_dset, a HF Dataset with just the TEXT_FIELD
        # instances in self.dset extracted
        self.load_or_prepare_text_dataset()

    def _get_dataset(self):
        """
        Gets the HuggingFace Dataset object.
        First tries to use the given cache directory if specified;
        otherwise saves to the given cache directory if specified.
        """
        dset = ds_utils.load_truncated_dataset(self.dset_name, self.dset_config,
                                               self.split_name,
                                               cache_dir=self.hf_dset_cache_dir,
                                               save=self.save)
        return dset

    @st.cache
    def load_or_prepare_text_dataset(self, load_only=False):
        """
        Prepares the HF dataset text/feature based on given config, split, etc.
        Args:
            load_only: Whether only a cached dataset can be used.
        """
        logs.info("Doing text dset.")
        if self.use_cache and exists(self.text_dset_fid):
            # load extracted text
            self.text_dset = load_from_disk(self.text_dset_fid)
            logs.info("Loaded dataset from disk")
            logs.info(self.text_dset)
        # ...Or load it from the server and store it anew
        elif not load_only:
            # Defines self.text_dset
            self.prepare_text_dset()
            if self.save:
                # save extracted text instances
                logs.info("Saving dataset to disk")
                self.text_dset.save_to_disk(self.text_dset_fid)

    def prepare_text_dset(self):
        logs.info("Working with dataset:")
        logs.info(self.dset)
        # Extract all text instances from the user-specified self.text_field,
        # which is a dataset-specific text/feature field;
        # create a new feature called TEXT_FIELD, which is a constant shared
        # across DMT logic.
        self.text_dset = self.dset.map(
            lambda examples: ds_utils.extract_field(
                examples, self.text_field, TEXT_FIELD
            ),
            batched=True,
            remove_columns=list(self.dset.features),
        )

    @st.cache
    def load_or_prepare_general_stats(self, load_only=False):
        """
        Content for expander_general_stats widget.
        Provides statistics for total words, total open words,
        the sorted top vocab, the NaN count, and the duplicate count.
        Args:

        Returns:

        """
        # General statistics
        # For the general statistics, text duplicates are not saved in their
        # own files, but rather just the text duplicate fraction is saved in the
        # "general" file. We therefore set save=False for
        # the text duplicate files in this case.
        # Similarly, we don't get the full list of duplicates
        # in general stats, so set list_duplicates to False
        self.load_or_prepare_text_duplicates(load_only=load_only, save=False,
                                             list_duplicates=False)
        logs.info("Duplicates results:")
        logs.info(self.duplicates_results)
        self.general_stats_dict.update(self.duplicates_results)
        # TODO: Tighten the rest of this similar to text_duplicates.
        if (
                self.use_cache
                and exists(self.general_stats_json_fid)
                and exists(self.sorted_top_vocab_df_fid)
        ):
            logs.info("Loading cached general stats")
            self.load_general_stats()
        elif not load_only:
            logs.info("Preparing general stats")
            self.prepare_general_stats()
            if self.save:
                ds_utils.write_df(self.sorted_top_vocab_df,
                               self.sorted_top_vocab_df_fid)
                ds_utils.write_json(self.general_stats_dict,
                                 self.general_stats_json_fid)

    @st.cache
    def load_or_prepare_text_lengths(self, load_only=False):
        """
        The text length widget relies on this function, which provides
        a figure of the text lengths, some text length statistics, and
        a text length dataframe to peruse.
        Args:
            load_only (Bool): Whether we can compute anew, or just need to try to grab cache.
        Returns:

        """
        # We work with the already tokenized dataset
        self.load_or_prepare_tokenized_df()
        self.length_obj = lengths.DMTHelper(self, load_only=load_only, save=self.save)
        self.length_obj.run_DMT_processing()

    ## Labels functions
    @st.cache
    def load_or_prepare_labels(self, load_only=False):
        """Uses a generic Labels class, with attributes specific to this
        project as input.
        Computes results for each label column,
        or else uses what's available in the cache.
        Currently supports Datasets with just one label column.
        """
        self.label_obj = labels.DMTHelper(self, load_only=load_only, save=self.save)
        self.label_obj.run_DMT_processing()

    # Get vocab with word counts
    @st.cache
    def load_or_prepare_vocab(self, load_only=False):
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
        elif not load_only:
            if self.tokenized_df is None:
                # Building the vocabulary starts with tokenizing.
                self.load_or_prepare_tokenized_df(load_only=False)
            logs.info("Calculating vocab afresh")
            word_count_df = count_vocab_frequencies(self.tokenized_df)
            self.vocab_counts_df = calc_p_word(word_count_df)
            self.vocab_counts_filtered_df = filter_vocab(self.vocab_counts_df)
            if self.save:
                logs.info("Writing out.")
                ds_utils.write_df(self.vocab_counts_df, self.vocab_counts_df_fid)
        logs.info("unfiltered vocab")
        logs.info(self.vocab_counts_df)
        logs.info("filtered vocab")
        logs.info(self.vocab_counts_filtered_df)

    def load_vocab(self):
        self.vocab_counts_df = ds_utils.read_df(self.vocab_counts_df_fid)

    @st.cache
    def load_or_prepare_text_duplicates(self, load_only=False, save=True, list_duplicates=True):
        """Uses a text duplicates library, which
        returns strings with their counts, fraction of data that is duplicated,
        or else uses what's available in the cache.
        """
        dups_obj = td.DMTHelper(self, load_only=load_only, save=save)
        dups_obj.run_DMT_processing(list_duplicates=list_duplicates)
        self.duplicates_results = dups_obj.duplicates_results
        self.dups_frac = self.duplicates_results[td.DUPS_FRAC]
        if list_duplicates and td.DUPS_DICT in self.duplicates_results:
            self.dups_dict = self.duplicates_results[td.DUPS_DICT]
        self.duplicates_files = dups_obj.get_duplicates_filenames()

    @st.cache
    def load_or_prepare_text_perplexities(self, load_only=False):
        perplex_obj = perplexity.DMTHelper(self, load_only=load_only)
        perplex_obj.run_DMT_processing()
        self.perplexities_df = perplex_obj.df


    def load_general_stats(self):
        self.general_stats_dict = json.load(
            open(self.general_stats_json_fid, encoding="utf-8")
        )
        self.sorted_top_vocab_df = ds_utils.read_df(self.sorted_top_vocab_df_fid)
        self.text_nan_count = self.general_stats_dict[TEXT_NAN_CNT]
        self.dups_frac = self.general_stats_dict[td.DUPS_FRAC]
        self.total_words = self.general_stats_dict[TOT_WORDS]
        self.total_open_words = self.general_stats_dict[TOT_OPEN_WORDS]

    def prepare_general_stats(self):
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
        self.load_or_prepare_text_duplicates()
        self.general_stats_dict = {
            TOT_WORDS: self.total_words,
            TOT_OPEN_WORDS: self.total_open_words,
            TEXT_NAN_CNT: self.text_nan_count,
            td.DUPS_FRAC: self.dups_frac
        }

    @st.cache
    def load_or_prepare_dataset(self, load_only=False):
        """
        Prepares the HF datasets and data frames containing the untokenized and
        tokenized text as well as the label values.
        self.tokenized_df is used further for calculating text lengths,
        word counts, etc.
        Args:
            load_only (Bool): Whether we should only use cache, no new prep.

        Returns:

        """
        logs.info("Doing text dset.")
        self.load_or_prepare_text_dset(load_only=load_only)

    @st.cache
    def load_or_prepare_dset_peek(self, load_only=False):
        if self.use_cache and exists(self.dset_peek_json_fid):
            with open(self.dset_peek_json_fid, "r") as f:
                self.dset_peek = json.load(f)["dset peek"]
        elif not load_only:
            self.dset_peek = self.dset[:100]
            if self.save:
                ds_utils.write_json({"dset peek": self.dset_peek},
                                 self.dset_peek_json_fid)

    @st.cache
    def load_or_prepare_tokenized_df(self, load_only=False):
        if self.use_cache and exists(self.tokenized_df_fid):
            self.tokenized_df = ds_utils.read_df(self.tokenized_df_fid)
        elif not load_only:
            # tokenize all text instances
            self.tokenized_df = Tokenize(self.text_dset, feature=TEXT_FIELD,
                                         tok_feature=TOKENIZED_FIELD).get_df()
            logs.info("tokenized df is")
            logs.info(self.tokenized_df)
            if self.save:
                logs.warning("Saving tokenized dataset to disk")
                # save tokenized text
                ds_utils.write_df(self.tokenized_df, self.tokenized_df_fid)

    @st.cache
    def load_or_prepare_npmi(self, load_only=False):
        self.npmi_obj = npmi.DMTHelper(self, IDENTITY_TERMS, load_only=load_only, use_cache=self.use_cache, save=self.save)
        self.npmi_obj.run_DMT_processing()

    @st.cache
    def load_or_prepare_zipf(self, load_only=False):
        self.zipf_obj = zipf.DMTHelper(self, load_only=load_only, use_cache=self.use_cache, save=self.save)
        self.zipf_obj.run_DMT_processing()

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
    batches = np.linspace(0, tokenized_df.shape[0], _NUM_VOCAB_BATCHES).astype(
        int)
    i = 0
    tf = []
    while i < len(batches) - 1:
        if i % 100 == 0:
            logs.info("%s of %s vocab batches" % (str(i), str(len(batches))))
        batch_result = np.sum(
            document_matrix[batches[i]: batches[i + 1]].toarray(), axis=0
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
    vocab_counts_df = pd.DataFrame(
        word_count_df.sort_values(by=CNT, ascending=False))
    vocab_counts_df[VOCAB] = vocab_counts_df.index
    return vocab_counts_df


def filter_vocab(vocab_counts_df):
    # TODO: Add warnings (which words are missing) to log file?
    filtered_vocab_counts_df = vocab_counts_df.drop(_CLOSED_CLASS,
                                                    errors="ignore")
    filtered_count = filtered_vocab_counts_df[CNT]
    filtered_count_denom = float(sum(filtered_vocab_counts_df[CNT]))
    filtered_vocab_counts_df[PROP] = filtered_count / filtered_count_denom
    return filtered_vocab_counts_df
