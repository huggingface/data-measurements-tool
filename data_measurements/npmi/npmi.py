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

import warnings
import utils
import utils.dataset_utils as ds_utils
from utils.dataset_utils import (CNT, TOKENIZED_FIELD)

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from os.path import join as pjoin

# Might be nice to print to log instead? Happens when we drop closed class.
warnings.filterwarnings(action="ignore", category=UserWarning)
# When we divide by 0 in log
np.seterr(divide="ignore")
# treating inf values as NaN as well
pd.set_option("use_inf_as_na", True)
logs = utils.prepare_logging(__file__)
# TODO: Should be possible for a user to specify this.
NUM_BATCHES = 500


class DMTHelper:
    """Helper class for the Data Measurements Tool.
    This allows us to keep all variables and functions related to labels
    in one file.
    """

    def __init__(self, dstats, identity_terms, load_only=False, use_cache=False, save=True):
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
        self.cache_path = pjoin(dstats.dataset_cache_dir, "associations")
        self.avail_terms_json_fid = pjoin(self.cache_path, "identity_terms.json")
        # TODO: Users ideally can type in whatever words they want.
        # This is the full list of terms.
        self.identity_terms = identity_terms
        logs.info("Using term list:")
        logs.info(self.identity_terms)
        # identity_terms terms that are available more than MIN_VOCAB_COUNT times
        self.avail_identity_terms = []
        # TODO: Let users specify
        self.open_class_only = True
        # Results of the single word associations and their paired bias values.
        self.results_dict = {}
        # Filenames for cache, based on the results
        self.fid_dict = {}

    def run_DMT_processing(self):
        # The identity terms that can be used
        self.load_or_prepare_avail_identity_terms()
        # Gets the association (npmi) scores and pair-wise differences for the
        # identity terms.
        # TODO: Implement more association measures.
        self.load_or_prepare_association_results(measure="npmi")

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
                    "Loaded identity terms that occur > %s times" % self.min_count)
        # Figure out the identity terms if we're not just loading from cache
        if not self.load_only:
            if not self.avail_identity_terms:
                self.avail_identity_terms = self._prepare_identity_terms()
            # Finish
            if self.save:
                self._write_term_cache()

    def _load_identity_cache(self):
        if exists(self.avail_terms_json_fid):
            avail_identity_terms = json.load(open(self.avail_terms_json_fid))
            return avail_identity_terms
        return []

    def load_or_prepare_association_results(self, measure="npmi"):
        # Filenames for caching and saving
        # Format is {measure:{subgroup:{term1:val, term2:val}...}}}
        bias_str = self._make_bias_str(measure)
        # File names for cache and saving
        fid_dict = {measure: {}, bias_str: {}}
        self.fid_dict.update(self._make_fids(fid_dict))
        # If we're trying to use the cache of already computed results
        if self.use_cache:
            self.results_dict = self._load_assoc_results_cache(measure=measure)
        # Compute results if we're not just loading from cache or the cache didn't have the results.
        if not self.load_only:
            if not self.results_dict:
                assoc_obj = nPMI(self.dstats.vocab_counts_df,
                                self.tokenized_sentence_df,
                                self.avail_identity_terms)
                self.results_dict = assoc_obj.results_dict
            # Finish
            if self.save:
                self._write_results_cache()

    def _load_assoc_results_cache(self, measure="npmi"):
        """
        :return: Dict of {single/pair:{subgroup:{measure:{word:valye}}}}
        """
        bias_str = self._make_bias_str(measure)
        for subgroup, fid in self.fid_dict[measure].items():
            if exists(fid):
                results_dict[measure] = ds_utils.read_json(fid)
        for subgroup, fid in self.fid_dict[bias_str].items():
            if exists(fid):
                results_dict[bias_str] = ds_utils.read_json(fid)
        return results_dict

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

    def _write_term_cache(self):
        ds_utils.make_path(self.cache_path)
        if self.avail_identity_terms:
            ds_utils.write_json(self.avail_identity_terms, self.avail_terms_json_fid)

    def _write_results_cache(self, measure="npmi"):
        ds_utils.make_path(pjoin(self.cache_path, measure))
        bias_str = self._make_bias_str(measure)
        if self.results_dict:
            for subgroup, fid in self.fid_dict[measure].items():
                ds_utils.write_json(self.results_dict[measure][subgroup], fid)
            for subgroups, fid in self.fid_dict[bias_str].items():
                ds_utils.write_json(self.results_dict[bias_str][subgroups], fid)

    def _make_bias_str(self, measure):
        # Utility function so that the key we use for association biases
        # is always the same w/o having to type it over and over.
        return measure + "-bias"

    def _make_fids(self, fid_dict, measure="npmi"):
        bias_measure = self._make_bias_str(measure)
        prefix = "word_associations-"
        for id_term in self.avail_identity_terms:
            fid = prefix + id_term + ".json"
            json_fid = pjoin(self.cache_path, measure, fid)
            fid_dict[measure][id_term] = json_fid
        paired_terms = pair_terms(self.avail_identity_terms)
        for id_term_tuple in paired_terms:
            id_term_str = '-'.join(id_term_tuple)
            fid = prefix + id_term_str + ".json"
            json_fid = pjoin(self.cache_path, measure, fid)
            fid_dict[bias_measure][id_term_tuple] = json_fid
        return fid_dict

    def get_filenames(self):
        filenames = {"available terms":self.avail_terms_json_fid, "results":self.fid_dict}
        return filenames


class nPMI:
    """
    Uses the vocabulary dataframe and tokenized sentences to calculate
    co-occurrence statistics, PMI, and nPMI
    """

    def __init__(self, vocab_counts_df, tokenized_sentence_df, given_id_terms):
        logs.debug("Initiating assoc class.")
        self.vocab_counts_df = vocab_counts_df
        # TODO: Change this logic so just the vocabulary is given.
        self.vocabulary = list(vocab_counts_df.index)
        self.vocab_counts = pd.DataFrame([0] * len(self.vocabulary))
        logs.debug("vocabulary is is")
        logs.debug(self.vocab_counts_df)
        self.tokenized_sentence_df = tokenized_sentence_df
        logs.debug("tokenized dataframe is")
        logs.debug(self.tokenized_sentence_df)
        self.given_id_terms = given_id_terms
        logs.info("identity terms are")
        logs.info(self.given_id_terms)
        # self.word_cnt_per_sentence holds # sentences x # words
        self.word_cnt_per_sentence = self.count_words_per_sentence()
        logs.debug(self.word_cnt_per_sentence)
        logs.info("Calculating results...")
        # Formatted as {subgroup:{"count":{...},"npmi":{...}}}
        self.id_results = self.calc_measures()
        self.results_dict = self.id_results
        # Terms we calculate the difference between
        self.paired_terms = pair_terms(given_id_terms)
        # Formatted as {(subgroup1,subgroup2):{"npmi bias":{...}}}
        self.paired_results = self.calc_bias(self.id_results, self.paired_terms)
        self.results_dict.update(self.paired_results)
        logs.debug("Results dict is")
        logs.debug(self.results_dict)


    def count_words_per_sentence(self):
        # Counts the number of each vocabulary item per-sentence in batches.
        logs.info("Creating co-occurrence matrix for nPMI calculations.")
        word_cnt_per_sentence = []
        logs.debug(self.vocab_counts)
        logs.info(self.tokenized_sentence_df)
        batches = np.linspace(0, self.tokenized_sentence_df.shape[0],
                              NUM_BATCHES).astype(int)
        # Creates matrix of size # batches x # sentences
        for batch_num in range(len(batches)-1):
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
            mlb_series = mlb.fit_transform(
                self.tokenized_sentence_df[batches[batch_num]:batches[batch_num + 1]])
            word_cnt_per_sentence.append(mlb_series)
            print(np.sum(mlb_series, axis=0))
        return word_cnt_per_sentence

    def calc_measures(self):
        id_results = {}
        for subgroup in self.given_id_terms:
            logs.info("Calculating for %s " % subgroup)
            # Index of the identity term in the vocabulary
            subgroup_idx = self.vocabulary.index(subgroup)
            print("idx is %s" % subgroup_idx)
            logs.debug("Calculating co-occurrences...")
            vocab_cooc_df = self.calc_cooccurrences(subgroup, subgroup_idx)
            logs.debug("Converting from vocab indexes to vocab, we now have:")
            logs.debug(vocab_cooc_df)
            logs.debug("Calculating PMI...")
            pmi_df = self.calc_PMI(vocab_cooc_df, subgroup)
            logs.debug(pmi_df)
            logs.debug("Calculating nPMI...")
            npmi_df = self.calc_nPMI(pmi_df, vocab_cooc_df, subgroup)
            logs.debug(npmi_df)
            # Create a data structure for the identity term associations
            id_results[subgroup] = {"count": vocab_cooc_df.to_dict(),
                                      "pmi": pmi_df.to_dict(),
                                      "npmi": npmi_df.to_dict()}
            logs.debug("results_dict is:")
            print(id_results)
        return id_results

    def calc_cooccurrences(self, subgroup, subgroup_idx):
        # Big computation here!  Should only happen once per subgroup.
        logs.debug(
            "Approaching big computation! Here, we binarize all words in the "
            "sentences, making a sparse matrix of sentences."
        )
        coo_series = []
        for batch_num in range(len(self.word_cnt_per_sentence)):
            self._write_debug_msg(batch_num, msg_type="batching")
            # Sparse-matrix sentences (columns=all vocabulary items) in batch
            batch_sentence_row = self.word_cnt_per_sentence[batch_num]
            # Co-occurrence counts
            batch_coo_df = self._transpose_counts(batch_num, batch_sentence_row,
                                                  subgroup, subgroup_idx)
            coo_series += [batch_coo_df]
        # Just get those terms that co-occur with the subgroup term (eg cooc>0).
        count_df = self._isolate_subgroup_coo(coo_series)
        logs.debug("Returning co-occurrence matrix")
        logs.debug(count_df)
        return count_df

    def _sum_coo(self, batch_coo_df, coo_list):
        """
        Creates a row of co-occurrence counts with the subgroup term for the
        given batch.
        """
        print(batch_coo_df)
        coo_sums = batch_coo_df.sum()
        print(coo_sums)
        coo_list += coo_sums
        return coo_list

    def _transpose_counts(self, batch_id, batch_sentence_row, subgroup,
                          subgroup_idx):
        # TODO: You add the vocab counts here by summing the columns.
        # Dataframe of # sentences in batch x vocabulary size
        sent_batch_df = pd.DataFrame(batch_sentence_row)
        # Size is # sentences x 1.
        # identity term count per-sentence for the given batch
        subgroup_df = self._isolate_subgroup_sentences(sent_batch_df,
                                                       subgroup,
                                                       subgroup_idx)
        # Size is # sentences x # vocab
        subgroup_sentences = sent_batch_df[sent_batch_df[subgroup_idx] > 0]
        # Debugging messages
        self._write_debug_msg(batch_id, subgroup_df, subgroup_sentences, msg_type="transpose")
        # Create cooccurrence matrix for the given subgroup and all words.
        batch_coo_df = pd.DataFrame(subgroup_sentences.T.dot(subgroup_df))
        return batch_coo_df

    def _isolate_subgroup_coo(self, coo_list):
        count_df = pd.DataFrame(coo_list, index=self.vocabulary)
        count_df = count_df.loc[~(count_df == 0).all(axis=1)]
        return count_df

    def _isolate_subgroup_sentences(self, sent_batch_df, subgroup,
                                    subgroup_idx):
        subgroup_df = sent_batch_df[subgroup_idx]
        subgroup_df.columns = [subgroup]
        # Remove the sentences where the count of the subgroup is 0.
        # This way we have less computation & resources needs.
        # Note however that we could use this to get the counts of each
        # vocab item, making the use of the vocab_counts df unnecessary.
        subgroup_df = subgroup_df[subgroup_df > 0]
        return subgroup_df

    def calc_PMI(self, vocab_cooc_df, subgroup):
        """
        # PMI(x;y) = h(y) - h(y|x)
        #          = h(subgroup) - h(subgroup|word)
        #          = log (p(subgroup|word) / p(subgroup))
        # nPMI additionally divides by -log(p(x,y)) = -log(p(x|y)p(y))
        """
        # Calculation of p(subgroup)
        subgroup_prob = self.vocab_counts_df.loc[subgroup]["proportion"]
        # Calculation of p(subgroup|word) = count(subgroup,word) / count(word)
        # Because the inidices match (the vocab words),
        # this division doesn't need to specify the index (I think?!)
        p_subgroup_g_word = (vocab_cooc_df[subgroup] / self.vocab_counts_df["count"]).dropna()
        logs.info("p_subgroup_g_word is")
        logs.info(p_subgroup_g_word)
        pmi_df = pd.DataFrame()
        pmi_df[subgroup] = np.log(p_subgroup_g_word / subgroup_prob).dropna()
        # Note: A potentially faster solution for adding count, npmi,
        # can be based on this zip idea:
        # df_test['size_kb'],  df_test['size_mb'], df_test['size_gb'] =
        # zip(*df_test['size'].apply(sizes))
        return pmi_df

    def calc_nPMI(self, pmi_df, vocab_cooc_df, subgroup):
        """
        # nPMI additionally divides by -log(p(x,y)) = -log(p(x|y)p(y))
        #                                           = -log(p(word|subgroup)p(word))
        """
        p_word_g_subgroup = vocab_cooc_df[subgroup] / sum(vocab_cooc_df[subgroup])
        p_word = pmi_df.apply(
            lambda x: self.vocab_counts_df.loc[x.name]["proportion"], axis=1
        )
        normalize_pmi = -np.log(p_word_g_subgroup * p_word)
        npmi_df = pd.DataFrame()
        npmi_df[subgroup] = pmi_df[subgroup] / normalize_pmi
        return npmi_df.dropna()

    def calc_bias(self, subgroup_results, paired_terms, measure="npmi"):
        """Uses the subgroup dictionaries to compute the differences across pairs.
        Uses dictionaries rather than dataframes due to the fact that dicts seem
        to be preferred amongst evaluate users so far.
        :return: Dict of (id_term1, id_term2):{measure-bias:{term1:diff, term2:diff ...}}"""
        bias_measure = self._make_bias_str(measure)
        paired_results_dict = {}
        for pair in paired_terms:
            paired_results_dict[pair] = {}
            s1 = pair[0]
            s2 = pair[1]
            s1_results = subgroup_results[s1][measure]
            s2_results = subgroup_results[s2][measure]
            shared_words = set(s1_results).intersection(s2_results)
            # This is the final result of all the work!
            word_diffs = {word: s1_results[word] - s2_results[word] for word in shared_words}
            paired_results_dict[pair][bias_measure] = word_diffs
        logs.debug("Paired results are ")
        logs.debug(paired_results_dict)
        return paired_results_dict

    def _write_debug_msg(self, batch_id, subgroup_df=None,
                         subgroup_sentences=None, msg_type="batching"):
        if msg_type == "batching":
            if not batch_id % 100:
                logs.debug(
                    "%s of %s co-occurrence count batches"
                    % (str(batch_id), str(len(self.word_cnt_per_sentence)))
                )
        elif msg_type == "transpose":
            if not batch_id % 100:
                logs.debug("Removing 0 counts, subgroup_df is")
                logs.debug(subgroup_df)
                logs.debug("subgroup_sentences is")
                logs.debug(subgroup_sentences)
                logs.debug(
                    "Now we do the transpose approach for co-occurrences")

def pair_terms(id_terms):
    """Creates the paired terms based on the given terms."""
    pairs = []
    for i in range(len(id_terms)):
        term1 = id_terms[i]
        for j in range(i + 1, len(id_terms)):
            term2 = id_terms[j]
            # Use one ordering for a pair.
            pair = tuple(sorted([term1, term2]))
            pairs += [pair]
    return pairs

