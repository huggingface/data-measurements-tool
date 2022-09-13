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
from collections import defaultdict

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
# For the associations of an identity term
SING = "associations"
# For the difference between the associations of identity terms
DIFF = "biases"
# Used in the figures we show in DMT
COMBINED = "combined"

def _make_bias_str(measure):
    """Utility function so that the key we use for association biases
    # is always the same w/o having to type it over and over."""
    return measure + "-bias"

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
        self.cache_path = pjoin(dstats.dataset_cache_dir, SING)
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
        # Single-word associations
        self.assoc_results_dict = {}
        # Paired term association bias
        self.bias_results_dict = {}
        # Results of the single word associations and their paired bias values.
        self.results_dict = {}
        # Filenames for cache, based on the results
        self.results_fid_dict = {SING:defaultdict(dict), DIFF:defaultdict(dict)}
        # Filenames for the combined dataframes used in display.
        self.combined_fid_dict = defaultdict(dict)
        # Dataframes used in displays.
        self.combined_dfs_dict = None

    def run_DMT_processing(self):
        # The identity terms that can be used
        self.load_or_prepare_avail_identity_terms()
        # Association measurements & pair-wise differences for identity terms.
        # TODO: Provide functionality for more assoiation measures
        measure = "npmi"
        self.load_or_prepare_association_results(measure=measure)

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

    def load_or_prepare_association_results(self, measure="npmi"):
        # Filenames for caching and saving
        # Format is {measure:{subgroup:{term1:val, term2:val}...}}}
        self._make_fids()
        # If we're trying to use the cache of already computed results
        if self.use_cache:
            # Loads the association results and the dataframes
            # used in the display.
           self.results_dict = self._load_results_cache(measure=measure)
        # Compute results if we're not just loading from cache or the cache didn't have the results.
        if not self.load_only:
            if not self.results_dict:
                # Does the actual computations
                self.results_dict = self._prepare_assoc_results()
                # Combines identity terms and pairs into one dataframe.
                self.results_dict[COMBINED] = self._prepare_combined_dfs()
            # Finish
            if self.save:
                self._write_results_cache()
                self._write_combined_cache()

    def _load_results_cache(self, measure="npmi"):
        """
        self.results_dict holds {single/pair/combined:{subgroup:{measure:{word:value}}}}
        """
        results_dict = {}
        bias_str = _make_bias_str(measure)
        pairs = pair_terms(self.avail_identity_terms)
        for subgroup in self.avail_identity_terms:
            fid = self.results_fid_dict[SING][subgroup][measure]
            if exists(fid):
                self.assoc_results_dict[subgroup][measure] = ds_utils.read_json(fid)
        for pair in pairs:
            bias_fid = self.results_fid_dict[DIFF][pair][bias_str]
            if exists(bias_fid):
                self.bias_results_dict[pair][bias_str] = ds_utils.read_json(bias_fid)
            combined_fid = self.combined_fid_dict[pair][bias_str]
            if exists(combined_fid):
                self.combined_df_dict[pair][bias_str] = ds_utils.read_df(combined_fid)
        results_dict[SING] = self.assoc_results_dict
        results_dict[DIFF] = self.bias_results_dict
        results_dict[COMBINED] = self.combined_dfs_dict
        return results_dict

    def _prepare_assoc_results(self):
        assoc_obj = nPMI(self.dstats.vocab_counts_df,
                         self.tokenized_sentence_df,
                         self.avail_identity_terms)
        self.assoc_results_dict = assoc_obj.assoc_results_dict
        self.bias_results_dict = assoc_obj.bias_results_dict
        results_dict = {SING: self.assoc_results_dict,
                        DIFF: self.bias_results_dict}
        return results_dict

    def _prepare_combined_dfs(self, measure="npmi"):
        bias_str = _make_bias_str(measure)
        combined_dfs_dict = defaultdict(dict)
        for pair in sorted(self.bias_results_dict):
            combined_df = pd.DataFrame()
            combined_df[pair] = pd.DataFrame(self.bias_results_dict[pair][bias_str])
            s1 = pair[0]
            s2 = pair[1]
            combined_df[s1] = pd.DataFrame(self.assoc_results_dict[s1][measure])
            combined_df[s2] = pd.DataFrame(self.assoc_results_dict[s2][measure])
            combined_dfs_dict[pair][bias_str] = combined_df
        # {pair: bias_str : {pd.DataFrame({pair:diffs, s1:assoc, s2:assoc})}}
        return combined_dfs_dict

    def _write_term_cache(self):
        ds_utils.make_path(self.cache_path)
        if self.avail_identity_terms:
            ds_utils.write_json(self.avail_identity_terms, self.avail_terms_json_fid)

    def _write_results_cache(self, measure="npmi"):
        ds_utils.make_path(pjoin(self.cache_path, measure))
        bias_measure = _make_bias_str(measure)
        logs.debug("fid dict is")
        logs.debug(self.results_fid_dict)
        logs.debug("assoc results dict is")
        logs.debug(self.assoc_results_dict)
        for subgroup, measure_cache_dict in self.results_fid_dict[SING].items():
            fid = measure_cache_dict[measure]
            ds_utils.write_json(self.assoc_results_dict[subgroup][measure].to_json(), fid)
        for subgroup_pair, measure_cache_dict in self.results_fid_dict[DIFF].items():
            fid = measure_cache_dict[bias_measure]
            ds_utils.write_json(self.bias_results_dict[subgroup_pair][bias_measure].to_json(), fid)

    def _write_combined_cache(self, measure="npmi"):
        bias_str = _make_bias_str(measure)
        ds_utils.make_path(pjoin(self.cache_path, measure))
        for pair, measure_dict in self.combined_dfs_dict.items():
            df = measure_dict[bias_str]
            fid = self.combined_fid_dict[pair][bias_str]
            ds_utils.write_df(df, fid)

    def _make_fids(self, measure="npmi"):
        bias_str = _make_bias_str(measure)
        # When we have the available identity terms,
        # we can make cache filenames for them.
        for id_term in self.avail_identity_terms:
            filename = SING + "-" + id_term + ".json"
            json_fid = pjoin(self.cache_path, measure, filename)
            self.results_fid_dict[SING][id_term][measure] = json_fid
        paired_terms = pair_terms(self.avail_identity_terms)
        for id_term_tuple in paired_terms:
            id_term_str = '-'.join(id_term_tuple)
            filename = DIFF + "-" + id_term_str + ".json"
            json_fid = pjoin(self.cache_path, measure, filename)
            self.results_fid_dict[DIFF][id_term_tuple][bias_str] = json_fid
            # The images are created for each pair.
            combined_filename = id_term_str + ".feather"
            feather_fid = pjoin(self.cache_path, measure, combined_filename)
            self.combined_fid_dict[id_term_tuple][bias_str] = feather_fid

    def load_combined_df(self, pair, measure="npmi"):
        """Returns the dataframe used in the DMT streamlit app to display the results"""
        bias_str = _make_bias_str(measure)
        combined_df = self.combined_dfs_dict[pair][bias_str]
        return combined_df

    def get_filenames(self):
        filenames = {"available terms": self.avail_terms_json_fid,
                     "results": self.results_fid_dict,
                     "dataframes for streamlit display": self.combined_fid_dict}
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
        #logs.debug(self.vocab_counts_df)
        self.tokenized_sentence_df = tokenized_sentence_df
        logs.debug("tokenized dataframe is")
        #logs.debug(self.tokenized_sentence_df)
        self.given_id_terms = given_id_terms
        logs.info("identity terms are")
        logs.info(self.given_id_terms)
        # Terms we calculate the difference between
        self.paired_terms = pair_terms(given_id_terms)

        # Matrix of # sentences x vocabulary size
        self.word_cnts_per_sentence = self.count_words_per_sentence()
        #logs.debug(self.word_cnts_per_sentence)
        logs.info("Calculating results...")
        # Formatted as {subgroup:{"count":{...},"npmi":{...}}}
        self.assoc_results_dict = self.calc_measures()
        # Formatted as {(subgroup1,subgroup2):{"npmi bias":{...}}}
        self.bias_results_dict = self.calc_bias(self.assoc_results_dict)


    def count_words_per_sentence(self):
        # Counts the number of each vocabulary item per-sentence in batches.
        logs.info("Creating co-occurrence matrix for nPMI calculations.")
        word_cnts_per_sentence = []
        #logs.debug(self.vocab_counts)
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
            sentence_batch = self.tokenized_sentence_df[batches[batch_num]:batches[batch_num + 1]]
            mlb_series = mlb.fit_transform(sentence_batch)
            word_cnts_per_sentence.append(mlb_series)
            #print(np.sum(mlb_series, axis=0))
        return word_cnts_per_sentence

    def calc_measures(self):
        id_results = {}
        for subgroup in self.given_id_terms:
            logs.info("Calculating for %s " % subgroup)
            # Index of the identity term in the vocabulary
            subgroup_idx = self.vocabulary.index(subgroup)
            print("idx is %s" % subgroup_idx)
            logs.debug("Calculating co-occurrences...")
            vocab_cooc_df = self.calc_cooccurrences(subgroup, subgroup_idx)
            logs.debug("Calculating PMI...")
            pmi_df = self.calc_PMI(vocab_cooc_df, subgroup)
            logs.debug("PMI dataframe is:")
            logs.debug(pmi_df)
            logs.debug("Calculating nPMI...")
            npmi_df = self.calc_nPMI(pmi_df, vocab_cooc_df, subgroup)
            logs.debug("npmi df is")
            logs.debug(npmi_df)
            # Create a data structure for the identity term associations
            id_results[subgroup] = {"count": vocab_cooc_df,
                                    "pmi": pmi_df,
                                    "npmi": npmi_df}
            logs.debug("results_dict is:")
            print(id_results)
        return id_results

    def calc_cooccurrences(self, subgroup, subgroup_idx):
        initialize = True
        coo_df = None
        # Big computation here!  Should only happen once.
        logs.debug(
            "Approaching big computation! Here, we binarize all words in the sentences, making a sparse matrix of sentences."
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
            # logs.info('sent batch df is')
            # logs.info(sent_batch_df)
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
        print("Now it is:")
        print(count_df)
        return count_df

    #def _sum_coo(self, batch_coo_df, coo_list):
    #    """
    #    Creates a row of co-occurrence counts with the subgroup term for the
    #    given batch.
    #    """
    #    #print(batch_coo_df)
    #    coo_sums = batch_coo_df.sum()
    #    #print(coo_sums)
    #    coo_list += coo_sums
    #    return coo_list

    #def _transpose_counts(self, batch_id, batch_sentence_row, subgroup,
    #                      subgroup_idx):
    #    # TODO: You add the vocab counts here by summing the columns.
    #    # Dataframe of # sentences in batch x vocabulary size
    #    sent_batch_df = pd.DataFrame(batch_sentence_row)
    #    # Size is # sentences x 1.
    #    # identity term count per-sentence for the given batch
    #    subgroup_df = self._isolate_subgroup_sentences(sent_batch_df,
    #                                                   subgroup,
    #                                                   subgroup_idx)
    #    # Size is # sentences x # vocab
    #    subgroup_sentences = sent_batch_df[sent_batch_df[subgroup_idx] > 0]
    #    # Debugging messages
    #    self._write_debug_msg(batch_id, subgroup_df, subgroup_sentences, msg_type="transpose")
    #    # Create cooccurrence matrix for the given subgroup and all words.
    #    batch_coo_df = pd.DataFrame(subgroup_sentences.T.dot(subgroup_df))
    #    return batch_coo_df

    #def _isolate_subgroup_coo(self, coo_list):
    #    count_df = pd.DataFrame(coo_list, index=self.vocabulary)
    #    count_df = count_df.loc[~(count_df == 0).all(axis=1)]
     #   return count_df

    #def _isolate_subgroup_sentences(self, sent_batch_df, subgroup, subgroup_idx):
    #    subgroup_df = sent_batch_df[subgroup_idx]
    #    subgroup_df.columns = [subgroup]
    #    # Remove the sentences where the count of the subgroup is 0.
    #    # This way we have less computation & resources needs.
    #    # Note however that we could use this to get the counts of each
    #    # vocab item, making the use of the vocab_counts df unnecessary.
    #    subgroup_df = subgroup_df[subgroup_df > 0]
    #    return subgroup_df

    def calc_PMI(self, vocab_cooc_df, subgroup):
        """A
        # PMI(x;y) = h(y) - h(y|x)
        #          = h(subgroup) - h(subgroup|word)az
        #          = log (p(subgroup|word) / p(subgroup))
        # nPMI additionally divides by -log(p(x,y)) = -log(p(x|y)p(y))
        """
        print("vocab cooc df")
        print(vocab_cooc_df)
        print("vocab counts")
        print(self.vocab_counts_df["count"])
        # Calculation of p(subgroup)
        subgroup_prob = self.vocab_counts_df.loc[subgroup]["proportion"]
        # Calculation of p(subgroup|word) = count(subgroup,word) / count(word)
        # Because the indices match (the vocab words),
        # this division doesn't need to specify the index (I think?!)
        vocab_cooc_df.columns = ["cooc"]
        p_subgroup_g_word = (vocab_cooc_df["cooc"]/self.vocab_counts_df["count"])
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
        return npmi_df.dropna()

    def calc_bias(self, measurements_dict, measure="npmi"):
        """Uses the subgroup dictionaries to compute the differences across pairs.
        Uses dictionaries rather than dataframes due to the fact that dicts seem
        to be preferred amongst evaluate users so far.
        :return: Dict of (id_term1, id_term2):{measure-bias:{term1:diff, term2:diff ...}}"""
        bias_measure = measure + "-bias"
        paired_results_dict = {}
        logs.debug("measurements dict is")
        logs.debug(measurements_dict)
        for pair in self.paired_terms:
            logs.debug("pair is")
            logs.debug(pair)
            paired_results_dict[pair] = {}
            s1 = pair[0]
            s2 = pair[1]
            s1_results = measurements_dict[s1][measure]
            logs.debug("s1 results are")
            logs.debug(s1_results)
            s2_results = measurements_dict[s2][measure]
            #shared_words = set(s1_results).intersection(s2_results)
            #logs.debug("Shared words are")
            #logs.debug(shared_words)
            # This is the final result of all the work!
            word_diffs = s1_results[s1] - s2_results[s2]
            word_diffs.columns = [("%s - %s") % (s1, s2)]
            logs.debug("word diffs are")
            logs.debug(word_diffs)
            #word_diffs = {word: s1_results[word] - s2_results[word] for word in shared_words}
            paired_results_dict[pair][bias_measure] = word_diffs.dropna()
        logs.debug("Paired results are ")
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
