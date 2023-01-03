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

import numpy as np
import pandas as pd
import utils
import utils.dataset_utils as ds_utils
import warnings
from collections import defaultdict
from os.path import exists
from os.path import join as pjoin
from sklearn.preprocessing import MultiLabelBinarizer
from utils.dataset_utils import (CNT, TOKENIZED_FIELD)
from abc import ABC, abstractmethod

# Might be nice to print to log instead? Happens when we drop closed class.
warnings.filterwarnings(action="ignore", category=UserWarning)
# When we divide by 0 in log
np.seterr(divide="ignore")
# treating inf values as NaN as well
pd.set_option("use_inf_as_na", True)
logs = utils.prepare_logging(__file__)
# TODO: Should be possible for a user to specify this.
NUM_BATCHES = 500
# For the associations of an identity term
SING = "associations"
# For the difference between the associations of identity terms
DIFF = "biases"
# Used in the figures we show in DMT
DMT = "combined"


def pair_terms(id_terms):
    """Creates alphabetically ordered paired terms based on the given terms."""
    pairs = []
    for i in range(len(id_terms)):
        term1 = id_terms[i]
        for j in range(i + 1, len(id_terms)):
            term2 = id_terms[j]
            # Use one ordering for a pair.
            pair = tuple(sorted([term1, term2]))
            pairs += [pair]
    return pairs


class DMTHelper:
    """Helper class for the Data Measurements Tool.
    This allows us to keep all variables and functions related to labels
    in one file.
    """

    def __init__(self, dstats, identity_terms, load_only=False, use_cache=False,
                 save=True):
        # The data measurements tool settings (dataset, config, etc.)
        self.dstats = dstats
        # Whether we can use caching (when live, no).
        self.load_only = load_only
        # Whether to first try using cache before calculating
        self.use_cache = use_cache
        # Whether to save results
        self.save = save
        # Tokenized dataset
        tokenized_df = dstats.tokenized_df
        self.tokenized_sentence_df = tokenized_df[TOKENIZED_FIELD]
        # Dataframe of shape #vocab x 1 (count)
        self.vocab_counts_df = dstats.vocab_counts_df
        # Cutoff for the number of times something must occur to be included
        self.min_count = dstats.min_vocab_count
        self.cache_path = pjoin(dstats.dataset_cache_dir, SING)
        self.avail_terms_json_fid = pjoin(self.cache_path,
                                          "identity_terms.json")
        # TODO: Users ideally can type in whatever words they want.
        # This is the full list of terms.
        self.identity_terms = identity_terms
        logs.info("Using term list:")
        logs.info(self.identity_terms)
        # identity_terms terms that are available more than MIN_VOCAB_COUNT
        self.avail_identity_terms = []
        # TODO: Let users specify
        self.open_class_only = True
        # Single-word associations
        self.assoc_results_dict = defaultdict(dict)
        # Paired term association bias
        self.bias_results_dict = defaultdict(dict)
        # Dataframes used in displays.
        self.bias_dfs_dict = defaultdict(dict)
        # Results of the single word associations and their paired bias values.
        # Formatted as:
        # {(s1,s2)): {pd.DataFrame({s1-s2:diffs, s1:assoc, s2:assoc})}}
        self.results_dict = defaultdict(lambda: defaultdict(dict))
        # Filenames for cache, based on the results
        self.filenames_dict = defaultdict(dict)

    def run_DMT_processing(self):
        # The identity terms that can be used
        self.load_or_prepare_avail_identity_terms()
        # Association measurements & pair-wise differences for identity terms.
        self.load_or_prepare_dmt_results()

    def load_or_prepare_avail_identity_terms(self):
        """
        Figures out what identity terms the user can select, based on whether
        they occur more than self.min_vocab_count times
        Provides identity terms -- uniquely and in pairs -- occurring at least
        self.min_vocab_count times.
        """
        # If we're trying to use the cache of available terms
        if self.use_cache:
            self.avail_identity_terms = self._load_identity_cache()
            if self.avail_identity_terms:
                logs.info(
                    "Loaded identity terms occuring >%s times" % self.min_count)
        # Figure out the identity terms if we're not just loading from cache
        if not self.load_only:
            if not self.avail_identity_terms:
                self.avail_identity_terms = self._prepare_identity_terms()
            # Finish
            if self.save:
                self._write_term_cache()

    def _load_identity_cache(self):
        if exists(self.avail_terms_json_fid):
            avail_identity_terms = ds_utils.read_json(self.avail_terms_json_fid)
            return avail_identity_terms
        return []

    def _prepare_identity_terms(self):
        """Uses DataFrame magic to return those terms that appear
        greater than min_vocab times."""
        # Mask to get the identity terms
        true_false = [term in self.vocab_counts_df.index for term in
                      self.identity_terms]
        # List of identity terms
        word_list_tmp = [x for x, y in zip(self.identity_terms, true_false) if
                         y]
        # Whether said identity terms have a count > min_count
        true_false_counts = [
            self.vocab_counts_df.loc[word, CNT] >= self.min_count for word in
            word_list_tmp]
        # List of identity terms with a count higher than min_count
        avail_identity_terms = [word for word, y in
                                zip(word_list_tmp, true_false_counts) if y]
        logs.debug("Identity terms that occur > %s times are:" % self.min_count)
        logs.debug(avail_identity_terms)
        return avail_identity_terms

    def load_or_prepare_dmt_results(self):
        # Initialize with no results (reset).
        self.results_dict = {}
        # Filenames for caching and saving
        self._make_fids()
        # If we're trying to use the cache of already computed results
        if self.use_cache:
            # Loads the association results and dataframes used in the display.
            logs.debug("Trying to load...")
            self.results_dict = self._load_dmt_cache()
        # Compute results if we can
        if not self.load_only:
            # If there isn't a solution using cache
            if not self.results_dict:
                # Does the actual computations
                self.prepare_results()
            # Finish
            if self.save:
                # Writes the paired & singleton dataframe out.
                self._write_dmt_cache()

    def _load_dmt_cache(self):
        """
        Loads dataframe with paired differences and individual item scores.
        """
        results_dict = defaultdict(lambda: defaultdict(dict))
        pairs = pair_terms(self.avail_identity_terms)
        for pair in pairs:
            combined_fid = self.filenames_dict[DMT][pair]
            if exists(combined_fid):
                results_dict[pair] = ds_utils.read_df(combined_fid)
        return results_dict

    def prepare_results(self):
        assoc_obj = nPMI(self.dstats.vocab_counts_df,
                         self.tokenized_sentence_df,
                         self.avail_identity_terms)
        self.assoc_results_dict = assoc_obj.assoc_results_dict
        self.results_dict = assoc_obj.bias_results_dict

    def _write_term_cache(self):
        ds_utils.make_path(self.cache_path)
        if self.avail_identity_terms:
            ds_utils.write_json(self.avail_identity_terms,
                                self.avail_terms_json_fid)

    def _write_dmt_cache(self, measure="npmi"):
        ds_utils.make_path(pjoin(self.cache_path, measure))
        for pair, bias_df in self.results_dict.items():
            logs.debug("Results for pair is:")
            logs.debug(bias_df)
            fid = self.filenames_dict[DMT][pair]
            logs.debug("Writing to %s" % fid)
            ds_utils.write_df(bias_df, fid)

    def _make_fids(self, measure="npmi"):
        """
        Utility function to create filename/path strings for the different
        result caches. This include single identity term results as well
        as the difference between them. Also includes the datastructure used in
        the DMT, which is a dataframe that has:
        (term1, term2) difference, term1 (scores), term2 (scores)
        """
        self.filenames_dict = {SING: {}, DIFF: {}, DMT: {}}
        # When we have the available identity terms,
        # we can make cache filenames for them.
        for id_term in self.avail_identity_terms:
            filename = SING + "-" + id_term + ".json"
            json_fid = pjoin(self.cache_path, measure, filename)
            self.filenames_dict[SING][id_term] = json_fid
        paired_terms = pair_terms(self.avail_identity_terms)
        for id_term_tuple in paired_terms:
            # The paired association results (bias) are stored with these files.
            id_term_str = '-'.join(id_term_tuple)
            filename = DIFF + "-" + id_term_str + ".json"
            json_fid = pjoin(self.cache_path, measure, filename)
            self.filenames_dict[DIFF][id_term_tuple] = json_fid
            # The display dataframes in the DMT are stored with these files.
            filename = DMT + "-" + id_term_str + ".json"
            json_fid = pjoin(self.cache_path, measure, filename)
            self.filenames_dict[DMT][id_term_tuple] = json_fid

    def get_display(self, s1, s2):
        pair = tuple(sorted([s1, s2]))
        display_df = self.results_dict[pair]
        logs.debug(self.results_dict)
        display_df.columns = ["bias", s1, s2]
        return display_df

    def get_filenames(self):
        filenames = {"available terms": self.avail_terms_json_fid,
                     "results": self.filenames_dict}
        return filenames


class Association(ABC):
    """
     Uses the vocabulary dataframe and tokenized sentences to calculate
     co-occurrence statistics, PMI, and nPMI
     """

    def __init__(self, vocab_counts_df, tokenized_sentence_df, given_id_terms, measure=None):
        logs.debug("Initiating assoc class.")
        self.vocab_counts_df = vocab_counts_df
        # TODO: Change this logic so just the vocabulary is given.
        self.vocabulary = list(vocab_counts_df.index)
        self.vocab_counts = pd.DataFrame([0] * len(self.vocabulary))
        logs.debug("vocabulary is is")
        logs.debug(self.vocab_counts_df)
        self.tokenized_sentence_df = tokenized_sentence_df
        logs.debug("tokenized sentences are")
        logs.debug(self.tokenized_sentence_df)
        self.given_id_terms = given_id_terms
        logs.info("identity terms are")
        logs.info(self.given_id_terms)
        # Terms we calculate the difference between
        self.paired_terms = pair_terms(given_id_terms)

        # Matrix of # sentences x vocabulary size
        self.word_cnts_per_sentence = self.count_words_per_sentence()
        logs.info("Calculating results...")
        # Formatted as {subgroup:{"count":{...},"npmi":{...}}}
        self.assoc_results_dict = self.calc_measures()
        # Dictionary keyed by pair tuples. Each value is a dataframe with
        # vocab terms as the index, and columns of paired difference and
        # individual scores for the two identity terms.
        self.bias_results_dict = self.calc_bias(self.assoc_results_dict, measure)

    def count_words_per_sentence(self):
        # Counts the number of each vocabulary item per-sentence in batches.
        logs.info("Creating co-occurrence matrix for nPMI calculations.")
        word_cnts_per_sentence = []
        logs.info(self.tokenized_sentence_df)
        batches = np.linspace(0, self.tokenized_sentence_df.shape[0],
                              NUM_BATCHES).astype(int)
        # Creates matrix of size # batches x # sentences
        for batch_num in range(len(batches) - 1):
            # Makes matrix shape: batch size (# sentences) x # words,
            # with the occurrence of each word per sentence.
            # vocab_counts_df.index is the vocabulary.
            mlb = MultiLabelBinarizer(classes=self.vocabulary)
            if batch_num % 100 == 0:
                logs.debug(
                    "%s of %s sentence binarize batches." % (
                        str(batch_num), str(len(batches)))
                )
            # Per-sentence word counts
            sentence_batch = self.tokenized_sentence_df[
                             batches[batch_num]:batches[batch_num + 1]]
            mlb_series = mlb.fit_transform(sentence_batch)
            word_cnts_per_sentence.append(mlb_series)
        return word_cnts_per_sentence

    @abstractmethod
    def calculate(self, subgroup):
        pass

    def calc_measures(self):
        id_results = {}
        for subgroup in self.given_id_terms:
            id_results[subgroup] = self.calculate(subgroup)
        return id_results

    def calc_bias(self, measurements_dict, measure):
        """Uses the subgroup dictionaries to compute the differences across pairs.
        Uses dictionaries rather than dataframes due to the fact that dicts seem
        to be preferred amongst evaluate users so far.
        :return: Dict of (id_term1, id_term2):{term1:diff, term2:diff ...}"""
        paired_results_dict = {}
        for pair in self.paired_terms:
            paired_results = pd.DataFrame()
            s1 = pair[0]
            s2 = pair[1]
            s1_results = measurements_dict[s1][measure]
            s2_results = measurements_dict[s2][measure]
            # !!! This is the final result of all the work !!!
            word_diffs = s1_results[s1] - s2_results[s2]
            paired_results[("%s - %s" % (s1, s2))] = word_diffs
            paired_results[s1] = s1_results
            paired_results[s2] = s2_results
            paired_results_dict[pair] = paired_results.dropna()
        logs.debug("Paired bias results from the main nPMI class are ")
        logs.debug(paired_results_dict)
        return paired_results_dict

    def _write_debug_msg(self, batch_id, subgroup_df=None,
                         subgroup_sentences=None, msg_type="batching"):
        if msg_type == "batching":
            if not batch_id % 100:
                logs.debug(
                    "%s of %s co-occurrence count batches"
                    % (str(batch_id), str(len(self.word_cnts_per_sentence)))
                )
        elif msg_type == "transpose":
            if not batch_id % 100:
                logs.debug("Removing 0 counts, subgroup_df is")
                logs.debug(subgroup_df)
                logs.debug("subgroup_sentences is")
                logs.debug(subgroup_sentences)
                logs.debug(
                    "Now we do the transpose approach for co-occurrences")


class Cooccurence(Association):
    def calculate(self, subgroup):
        logs.info("Calculating for %s " % subgroup)
        # Index of the identity term in the vocabulary
        subgroup_idx = self.vocabulary.index(subgroup)
        # print("idx is %s" % subgroup_idx)
        logs.debug("Calculating co-occurrences...")

        initialize = True
        coo_df = None
        # Big computation here!  Should only happen once.
        logs.debug(
            "Approaching big computation! Here, we binarize all words in the "
            "sentences, making a sparse matrix of sentences."
        )
        for batch_id in range(len(self.word_cnts_per_sentence)):
            # Every 100 batches, print out the progress.
            if not batch_id % 100:
                logs.debug(
                    "%s of %s co-occurrence count batches"
                    % (str(batch_id), str(len(self.word_cnts_per_sentence)))
                )
            # List of all the sentences (list of vocab) in that batch
            batch_sentence_row = self.word_cnts_per_sentence[batch_id]
            # Dataframe of # sentences in batch x vocabulary size
            sent_batch_df = pd.DataFrame(batch_sentence_row)
            # Subgroup counts per-sentence for the given batch
            subgroup_df = sent_batch_df[subgroup_idx]
            subgroup_df.columns = [subgroup]
            # Remove the sentences where the count of the subgroup is 0.
            # This way we have less computation & resources needs.
            subgroup_df = subgroup_df[subgroup_df > 0]
            mlb_subgroup_only = sent_batch_df[sent_batch_df[subgroup_idx] > 0]
            # Create cooccurrence matrix for the given subgroup and all words.
            batch_coo_df = pd.DataFrame(mlb_subgroup_only.T.dot(subgroup_df))

            # Creates a batch-sized dataframe of co-occurrence counts.
            # Note these could just be summed rather than be batch size.
            if initialize:
                coo_df = batch_coo_df
            else:
                coo_df = coo_df.add(batch_coo_df, fill_value=0)
            initialize = False
        logs.debug("Made co-occurrence matrix")
        logs.debug(coo_df)
        count_df = coo_df.set_index(self.vocab_counts_df.index)
        count_df.columns = ["count"]
        count_df["count"] = count_df["count"].astype(int)

        # Create a data structure for the identity term associations
        # logs.debug("results_dict is:")
        return {
            "count": count_df,
        }


class PMI(Cooccurence):
    def __init__(self, *args, measure="pmi", **kwargs):
        super().__init__(*args, **kwargs, measure=measure)

    def calculate(self, subgroup):
        base_results = super().calculate(subgroup)
        vocab_cooc_df = base_results["count"]

        """A
        # PMI(x;y) = h(y) - h(y|x)
        #          = h(subgroup) - h(subgroup|word)az
        #          = log (p(subgroup|word) / p(subgroup))
        # nPMI additionally divides by -log(p(x,y)) = -log(p(x|y)p(y))
        """
        # Calculation of p(subgroup)
        subgroup_prob = self.vocab_counts_df.loc[subgroup]["proportion"]
        # Calculation of p(subgroup|word) = count(subgroup,word) / count(word)
        # Because the indices match (the vocab words),
        # this division doesn't need to specify the index (I think?!)
        vocab_cooc_df.columns = ["cooc"]
        p_subgroup_g_word = (
                vocab_cooc_df["cooc"] / self.vocab_counts_df["count"])
        logs.info("p_subgroup_g_word is")
        logs.info(p_subgroup_g_word)
        pmi_df = pd.DataFrame()
        pmi_df[subgroup] = np.log(p_subgroup_g_word / subgroup_prob).dropna()
        # Note: A potentially faster solution for adding count, npmi,
        # can be based on this zip idea:
        # df_test['size_kb'],  df_test['size_mb'], df_test['size_gb'] =
        # zip(*df_test['size'].apply(sizes))

        return {**base_results, "pmi": pmi_df}


class nPMI(PMI):
    def __init__(self, *args, measure="npmi", **kwargs):
        super().__init__(*args, **kwargs, measure=measure)

    def calculate(self, subgroup):
        base_results = super().calculate(subgroup)

        pmi_df = base_results["pmi"]
        vocab_cooc_df = base_results["count"]

        """
        # nPMI additionally divides by -log(p(x,y)) = -log(p(x|y)p(y))
        #                                           = -log(p(word|subgroup)p(word))
        """
        p_word_g_subgroup = vocab_cooc_df["cooc"] / sum(vocab_cooc_df["cooc"])
        logs.debug("p_word_g_subgroup")
        logs.debug(p_word_g_subgroup)
        p_word = pmi_df.apply(
            lambda x: self.vocab_counts_df.loc[x.name]["proportion"], axis=1
        )
        logs.debug("p word is")
        logs.debug(p_word)
        normalize_pmi = -np.log(p_word_g_subgroup * p_word)
        npmi_df = pd.DataFrame()
        npmi_df[subgroup] = pmi_df[subgroup] / normalize_pmi

        return {**base_results, "npmi": npmi_df.dropna()}
