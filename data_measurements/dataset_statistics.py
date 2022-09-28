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
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import statistics
import utils
import utils.dataset_utils as ds_utils
from data_measurements.tokenize import Tokenize
from data_measurements.labels import labels
from data_measurements.perplexity import perplexity
from data_measurements.lengths import lengths
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

from utils.dataset_utils import (CNT, TEXT_FIELD, PROP, TEXT_NAN_CNT, TOP_VOCAB, DUPS_FRAC,
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
NUM_VOCAB_BATCHES = 2000


class DatasetStatisticsCacheClass:

    def __init__(self, dset_name, dset_config, split_name, text_field,
                 label_field, label_names, cache_dir="cache_dir",
                 dset_cache_dir=None, use_cache=False, load_only=False, save=True):
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
        self.npmi_obj = None

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
        # Whether we should only use cache, and not use cache
        self.load_only = load_only
        self.check_cache_load_flags()
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
        self.top_vocab = None

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
        self.word_counts = 0
        self.open_word_counts = 0
        # Number of NaN values (NOT empty strings)
        self.text_nan_count = 0
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

        # Load the HuggingFace dataset object with the given arguments.
        self.dset = self._get_dataset()
        self.text_dset = None
        # Defines self.text_dset, a HF Dataset with just the TEXT_FIELD
        # instances in self.dset extracted
        self.load_or_prepare_text_dataset()

    def check_cache_load_flags(self):
        if self.load_only and not self.use_cache:
            logs.warning("You asked only for loading from cache, but also "
                         "specified not to use cache. Bravely changing the flag "
                         "to allow using cache.")
            self.use_cache = True

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

    def load_or_prepare_text_dataset(self):
        """
        Prepares the HF dataset text/feature based on given config, split, etc.
        """
        logs.info("Doing text dset.")
        if self.use_cache and exists(self.text_dset_fid):
            # load extracted text
            self.text_dset = load_from_disk(self.text_dset_fid)
            logs.info("Loaded dataset from disk")
            logs.info(self.text_dset)
        # ...Or load it from the server and store it anew
        elif not self.load_only:
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

    def load_or_prepare_tokenized_df(self):
        if self.use_cache and exists(self.tokenized_df_fid):
            self.tokenized_df = ds_utils.read_df(self.tokenized_df_fid)
        elif not self.load_only:
            self.prepare_tokenized_df()
            if self.save:
                logs.warning("Saving tokenized dataset to disk")
                # save tokenized text
                ds_utils.write_df(self.tokenized_df, self.tokenized_df_fid)

    def prepare_tokenized_df(self):
        # tokenize all text instances
        self.tokenized_df = Tokenize(self.text_dset, feature=TEXT_FIELD,
                                     tok_feature=TOKENIZED_FIELD).get_df()
        logs.info("tokenized df is")
        logs.info(self.tokenized_df)

    # Get vocab with word counts
    def load_or_prepare_vocab(self):
        """
        Calculates the vocabulary count from the tokenized text.
        The resulting dataframes may be used in nPMI calculations, zipf, etc.
        :param
        :return:
        """
        if self.use_cache and exists(self.vocab_counts_df_fid):
            logs.info("Reading vocab from cache")
            self.load_vocab()
        elif not self.load_only:
            self.prepare_vocab()
            if self.save:
                logs.info("Writing out.")
                ds_utils.write_df(self.vocab_counts_df, self.vocab_counts_df_fid)
        logs.info("unfiltered vocab")
        logs.info(self.vocab_counts_df)
        logs.info("filtered vocab")
        logs.info(self.vocab_counts_filtered_df)

    def load_vocab(self):
        self.vocab_counts_df = ds_utils.read_df(self.vocab_counts_df_fid)
        self.vocab_counts_filtered_df = filter_vocab(self.vocab_counts_df)

    def prepare_vocab(self):
        if self.tokenized_df is None:
            # Building the vocabulary starts with tokenizing.
            self.load_or_prepare_tokenized_df()
        logs.info("Calculating vocab afresh")
        word_count_df = count_vocab_frequencies(self.tokenized_df, NUM_VOCAB_BATCHES)
        self.vocab_counts_df = calc_p_word(word_count_df)
        self.vocab_counts_filtered_df = filter_vocab(self.vocab_counts_df)

    def load_or_prepare_dset_peek(self):
        if self.use_cache and exists(self.dset_peek_json_fid):
            self.load_dset_peek()
        elif not self.load_only:
            self.prepare_dset_peek()
            if self.save:
                ds_utils.write_json({"dset peek": self.dset_peek},
                                    self.dset_peek_json_fid)

    def load_dset_peek(self):
        dset_peek_all = ds_utils.read_json(self.dset_peek_json_fid)
        self.dset_peek = dset_peek_all["dset peek"]

    def prepare_dset_peek(self, peek_num=100):
        self.dset_peek = self.dset[:peek_num]

    def load_or_prepare_general_stats(self):
        """
        Content for expander_general_stats widget.
        Provides statistics for total words, total open words,
        the sorted top vocab, the NaN count, and the duplicate count.
        Args:

        Returns:

        """
        # General statistics
        if self.use_cache and exists(self.general_stats_json_fid):
            logs.info("Loading cached general stats")
            self.load_general_stats()
        else:
            self.prepare_general_stats()
            if self.save:
                ds_utils.write_json(self.general_stats_dict, 
                                    self.general_stats_json_fid)

    def load_general_stats(self):
        self.general_stats_dict = ds_utils.read_json(self.general_stats_json_fid)
        self.text_nan_count = self.general_stats_dict[TEXT_NAN_CNT]
        self.dups_frac = self.general_stats_dict[DUPS_FRAC]
        self.word_counts = self.general_stats_dict[TOT_WORDS]
        self.open_word_counts = self.general_stats_dict[TOT_OPEN_WORDS]
        self.top_vocab = self.general_stats_dict[TOP_VOCAB]

    def prepare_general_stats(self):
        self.prepare_word_counts()
        self.general_stats_dict[TOT_WORDS] = self.word_counts
        self.general_stats_dict[TOT_OPEN_WORDS] = self.open_word_counts
        self.prepare_nan_count()
        self.general_stats_dict[TEXT_NAN_CNT] = self.text_nan_count
        self.prepare_top_vocab()
        self.general_stats_dict[TOP_VOCAB] = self.top_vocab
        # Text duplicates are not saved in their
        # own files, but rather just the text duplicate fraction is saved in the
        # "general" file. We therefore set save=False for
        # the text duplicate files in this case.
        # Similarly, we don't need the full list of duplicates
        # in general stats, so set list_duplicates to False
        self.load_or_prepare_text_duplicates(save=False,
                                             list_duplicates=False)
        self.general_stats_dict[DUPS_FRAC] = self.dups_frac

    def prepare_nan_count(self):
        if self.tokenized_df is None:
            self.load_or_prepare_tokenized_df()
        self.text_nan_count = int(self.tokenized_df.isnull().sum().sum())

    def prepare_word_counts(self):
        if self.vocab_counts_df is None:
            self.load_or_prepare_vocab()
        self.word_counts = len(self.vocab_counts_df)
        self.open_word_counts = len(self.vocab_counts_filtered_df)

    def prepare_top_vocab(self, top_n=100):
        if self.vocab_counts_filtered_df is None:
            self.load_or_prepare_vocab()
        self.top_vocab = self.vocab_counts_filtered_df.sort_values(CNT, ascending=False).head(top_n)
        logs.info("vocab counts")
        logs.info(self.vocab_counts_filtered_df)
        logs.info("top vocab")
        logs.info(self.top_vocab)

    def load_or_prepare_text_duplicates(self, save=True, list_duplicates=True):
        """Uses a text duplicates library, which
        returns strings with their counts, fraction of data that is duplicated,
        or else uses what's available in the cache.
        """
        dups_obj = td.DMTHelper(self, save=save)
        dups_obj.run_DMT_processing(list_duplicates=list_duplicates)
        self.duplicates_results = dups_obj.duplicates_results
        self.dups_frac = self.duplicates_results[td.DUPS_FRAC]
        if list_duplicates and td.DUPS_DICT in self.duplicates_results:
            self.dups_dict = self.duplicates_results[td.DUPS_DICT]
        self.duplicates_files = dups_obj.get_duplicates_filenames()

    def load_or_prepare_text_lengths(self):
        """
        The text length widget relies on this function, which provides
        a figure of the text lengths, some text length statistics, and
        a text length dataframe to peruse.
        """
        # We work with the already tokenized dataset
        self.load_or_prepare_tokenized_df()
        self.length_obj = lengths.DMTHelper(self, save=self.save)
        self.length_obj.run_DMT_processing()

    ## Labels functions
    def load_or_prepare_labels(self):
        """Uses a generic Labels class, with attributes specific to this
        project as input.
        Computes results for each label column,
        or else uses what's available in the cache.
        Currently supports Datasets with just one label column.
        """
        self.label_obj = labels.DMTHelper(self, save=self.save)
        self.label_obj.run_DMT_processing()

    def load_or_prepare_text_perplexities(self):
        perplex_obj = perplexity.DMTHelper(self)
        perplex_obj.run_DMT_processing()
        self.perplexities_df = perplex_obj.df

    def load_or_prepare_npmi(self):
        self.npmi_obj = npmi.DMTHelper(self, IDENTITY_TERMS, use_cache=self.use_cache, save=self.save)
        self.npmi_obj.run_DMT_processing()

    def load_or_prepare_zipf(self):
        zipf_json_fid, zipf_fig_json_fid, zipf_fig_html_fid = zipf.get_zipf_fids(
            self.dset_cache_dir)
        if self.use_cache and exists(zipf_json_fid):
            # Zipf statistics
            # Read Zipf statistics: Alpha, p-value, etc.
            with open(zipf_json_fid, "r") as f:
                zipf_dict = json.load(f)
            self.z = zipf.Zipf(self.vocab_counts_df)
            self.z.load(zipf_dict)
            # Zipf figure
            if exists(zipf_fig_json_fid):
                self.zipf_fig = ds_utils.read_plotly(zipf_fig_json_fid)
            elif not self.load_only:
                self.zipf_fig = zipf.make_zipf_fig(self.z)
                if self.save:
                    ds_utils.write_plotly(self.zipf_fig)
        elif not self.load_only:
            self.prepare_zipf()
            if self.save:
                zipf_dict = self.z.get_zipf_dict()
                ds_utils.write_json(zipf_dict, zipf_json_fid)
                ds_utils.write_plotly(self.zipf_fig, zipf_fig_json_fid)
                self.zipf_fig.write_html(zipf_fig_html_fid)

    def prepare_zipf(self):
        # Calculate zipf from scratch
        # TODO: Does z even need to be self?
        self.z = zipf.Zipf(self.vocab_counts_df)
        self.z.calc_fit()
        self.zipf_fig = zipf.make_zipf_fig(self.z)

def dummy(doc):
    return doc

def count_vocab_frequencies(tokenized_df, num_vocab_batches=2000):
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
    batches = np.linspace(0, tokenized_df.shape[0], num_vocab_batches).astype(
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



# =======
#     def load_or_prepare_dataset(self):
# >>>>>>> main