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

_NUM_BATCHES = 500


class DMTHelper:
    """Helper class for the Data Measurements Tool.
    This allows us to keep all variables and functions related to labels
    in one file.
    """

    def __init__(self, dstats, identity_terms, use_cache=False, save=True):
        self.dstats = dstats
        self.use_cache = use_cache
        self.save = save
        tokenized_df = dstats.tokenized_df
        self.tokenized_sentence_df = tokenized_df[TOKENIZED_FIELD]
        self.vocab_counts_df = dstats.vocab_counts_df
        # Cutoff for the number of times something must occur to be included
        self.min_count = dstats.min_vocab_count
        self.npmi_cache_path = pjoin(dstats.dataset_cache_dir, "npmi")
        ds_utils.make_path(self.npmi_cache_path)
        self.npmi_terms_json_fid = pjoin(self.npmi_cache_path, "npmi_terms.json")
        #self.npmi_df_fid = pjoin(self.npmi_cache_path, "npmi_results.feather")
        self.npmi_results_json_fid_dict = {}
        for i in range(len(identity_terms)):
            identity_term = identity_terms[i]
            json_fid = pjoin(self.npmi_cache_path, "word_associations-" + identity_term + ".json")
            self.npmi_results_json_fid_dict[identity_term] = json_fid
            for j in range(i+1,len(identity_terms)):
                identity_term2 = identity_terms[j]
                identity_pair = tuple(sorted([identity_term, identity_term2]))
                identity_pair_fid = identity_pair[0] + "-" + identity_pair[1]
                paired_json_fid = pjoin(self.npmi_cache_path, "word_associations-" + identity_pair_fid + ".json")
                self.npmi_results_json_fid_dict[identity_pair] = paired_json_fid
        #self.npmi_results_json_prefix = self.npmi_cache_path + "/word_associations-"
        # TODO: Users ideally can type in whatever words they want.
        # This is the full list of terms.
        self.identity_terms = identity_terms
        logs.info("Using term list:")
        logs.info(self.identity_terms)
        # identity_terms terms that are available more than _MIN_VOCAB_COUNT times
        self.avail_identity_terms = []
        # pairs of identity terms between which differences are computed.
        self.paired_identity_terms = []
        # TODO: Let users specify
        self.open_class_only = True
        self.joint_npmi_df_dict = {}
        self.subgroup_results_dict = {}
        self.subgroup_files = {}
        self.results_dict = {}

    def run_DMT_processing(self, load_only=False):
        # Sets the identity terms that can be used
        self.load_or_prepare_avail_identity_terms(load_only=load_only)
        # Gets the npmi scores and pair-wise differences for the identity terms.
        self.load_or_prepare_npmi_results()

    def load_or_prepare_npmi_results(self, load_only=False):
        # If we're trying to use the cache of available terms
        if self.use_cache:
            self.results_dict = self._load_npmi_results_cache()
        # Figure out the identity terms if we're not just loading from cache
        if not load_only:
            if not self.results_dict:
                npmi_obj = nPMI(self.dstats.vocab_counts_df,
                                self.tokenized_sentence_df,
                                self.avail_identity_terms)
                self.results_dict = npmi_obj.results_dict
            # Finish
            if self.save:
                self._write_cache()

    def load_or_prepare_avail_identity_terms(self, load_only=False):
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
        if not load_only:
            if not self.avail_identity_terms:
                self.avail_identity_terms = self._prepare_identity_terms()
            # Finish
            if self.save:
                self._write_cache()
        # Construct the identity pairs available based on this.
        self.paired_identity_terms = self._pair_avail_identity_terms()

    def _pair_avail_identity_terms(self):
        paired_identity_terms = []
        for i in range(len(self.identity_terms)):
            s1 = self.identity_terms[i]
            for j in range(i+1, len(self.identity_terms)):
                s2 = self.identity_terms[j]
                # Use the same (alphabetical) ordering for subgroup pairs.
                pair = tuple(sorted([s1, s2]))
                paired_identity_terms += [pair]
        return paired_identity_terms


    def _load_identity_cache(self):
        if exists(self.npmi_terms_json_fid):
            avail_identity_terms = json.load(open(self.npmi_terms_json_fid))
            return avail_identity_terms
        return []

    def _load_npmi_results_cache(self):
        results_dict = {}
        for subgroups, fid in self.npmi_results_json_fid_dict.items():
            if exists(fid):
                results_dict[subgroups] = ds_utils.read_json(fid)
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

    def _write_cache(self):
        ds_utils.make_path(self.npmi_cache_path)
        if self.avail_identity_terms:
            ds_utils.write_json(self.avail_identity_terms, self.npmi_terms_json_fid)
        for subgroups in self.results_dict:
            npmi_results_json_subgroup_fid = self.npmi_results_json_fid_dict[subgroups]
            ds_utils.write_json(self.results_dict[subgroups], npmi_results_json_subgroup_fid)

    def get_available_terms(self):
        return self.load_or_prepare_identity_terms()

    def get_filenames(self):
        filenames = {"available terms":self.npmi_terms_json_fid, "results":self.npmi_results_json_fid_dict}
        return filenames


class nPMI:
    """
    Uses the vocabulary dataframe and tokenized sentences to calculate co-occurrence
    statistics.
    """

    def __init__(self, vocab_counts_df, tokenized_sentence_df, identity_terms):
        logs.debug("Initiating npmi class.")
        self.vocab_counts_df = vocab_counts_df
        logs.debug("vocabulary is is")
        logs.debug(self.vocab_counts_df)
        self.tokenized_sentence_df = tokenized_sentence_df
        logs.debug("tokenized dataframe is")
        logs.debug(self.tokenized_sentence_df)
        self.identity_terms = identity_terms
        logs.info("Counting word coocurrences....")
        # self.word_cnt_per_sentence holds # sentences x # words
        self.word_cnt_per_sentence = count_words_per_sentence(
            tokenized_sentence_df, vocab_counts_df, _NUM_BATCHES)
        logs.info(self.word_cnt_per_sentence)
        logs.info("identity terms are")
        logs.info(self.identity_terms)
        self.results_dict = self.calc_metrics()
        self.paired_results_dict = self.calc_paired_metrics()
        logs.debug("npmi bias is")
        logs.debug(self.paired_results_dict)
        #print(self.npmi_bias_df)

    def calc_paired_metrics(self):
        """Uses the subgroup dictionaries to compute the differences across pairs.
        Uses dictionaries rather than dataframes due to the fact that dicts seem
        to be preferred amongst evaluate users so far."""
        paired_results_dict = {}
        for i in range(len(self.identity_terms)):
            s1 = self.identity_terms[i]
            s1_dict = self._get_subgroup_dict(s1)
            for j in range(i+1, len(self.identity_terms)):
                s2 = self.identity_terms[j]
                s2_dict = self._get_subgroup_dict(s2)
                shared_keys = set(s1_dict).intersection(s2_dict)
                npmi_diff = {key: s1_dict[key] - s2_dict[key] for key in shared_keys}
                # Use the same (alphabetical) ordering for subgroup pairs.
                pair = tuple(sorted([s1, s2]))
                paired_results_dict[pair] = npmi_diff
        return paired_results_dict

    def _get_subgroup_dict(self, id_term, association="npmi"):
        """Helper function to extract the dictionaries of {word:association_measure} for each
        subgroup."""
        id_term_association = self.results_dict[id_term][association]
        id_term_npmi_key = id_term + "-" + association
        id_term_dict = id_term_association[id_term_npmi_key]
        return id_term_dict

    def calc_metrics(self):
        npmi_bias_dict = {}
        # The columns should be identical sizes, so this hopefully won't have an effect.
        take_bigger = lambda s1, s2: s1 if s1.sum() < s2.sum() else s2
        for subgroup in self.identity_terms:
            logs.info("Calculating for %s " % subgroup)
            # Index of the subgroup word in the sparse vector
            subgroup_idx = self.vocab_counts_df.index.get_loc(subgroup)
            logs.debug("Calculating co-occurrences...")
            df_coo = self.calc_cooccurrences(subgroup, subgroup_idx)
            vocab_cooc_df = self.set_columm_names(df_coo, subgroup)
            logs.debug("Converting from vocab indexes to vocab, we now have:")
            logs.debug(vocab_cooc_df)
            logs.debug("Calculating PMI...")
            pmi_df = self.calc_PMI(vocab_cooc_df, subgroup)
            logs.debug(pmi_df)
            logs.debug("Calculating nPMI...")
            npmi_df = self.calc_nPMI(pmi_df, vocab_cooc_df, subgroup)
            logs.debug(npmi_df)
            # Create a data structure for the identity term associations
            npmi_bias_dict[subgroup] = {"count":vocab_cooc_df.to_dict(), "pmi":pmi_df.to_dict(), "npmi": npmi_df.to_dict()}
            #vocab_cooc_df.combine(pmi_df, take_bigger, fill_value=0).combine(npmi_df, take_bigger, fill_value=0)
            logs.debug("npmi_bias_dict is:")
            print(npmi_bias_dict)
        #npmi_bias_df = pd.DataFrame(npmi_bias_dict, index=["count", "npmi", "pmi"])
        return npmi_bias_dict


    def calc_cooccurrences(self, subgroup, subgroup_idx):
        initialize = True
        coo_df = None
        # Big computation here!  Should only happen once per subgroup.
        logs.debug(
            "Approaching big computation! Here, we binarize all words in the "
            "sentences, making a sparse matrix of sentences."
        )
        for batch_id in range(len(self.word_cnt_per_sentence)):
            if not batch_id % 100:
                logs.debug(
                    "%s of %s co-occurrence count batches"
                    % (str(batch_id), str(len(self.word_cnt_per_sentence)))
                )
            # List of all the sentences (list of vocab) in that batch
            batch_sentence_row = self.word_cnt_per_sentence[batch_id]
            # Dataframe of # sentences in batch x vocabulary size
            sent_batch_df = pd.DataFrame(batch_sentence_row)
            # Subgroup counts per-sentence for the given batch
            subgroup_df = sent_batch_df[subgroup_idx]
            subgroup_df.columns = [subgroup]
            # Remove the sentences where the count of the subgroup is 0.
            # This way we have less computation & resources needs.
            subgroup_df = subgroup_df[subgroup_df > 0]
            mlb_subgroup_only = sent_batch_df[sent_batch_df[subgroup_idx] > 0]
            if not batch_id % 100:
                logs.debug("Removing 0 counts, subgroup_df is")
                logs.debug(subgroup_df)
                logs.debug("mlb subgroup only is")
                logs.debug(mlb_subgroup_only)
                # Create cooccurrence matrix for the given subgroup and all words.
                logs.debug("Now we do the transpose approach for co-occurrences")
            batch_coo_df = pd.DataFrame(mlb_subgroup_only.T.dot(subgroup_df))
            # Creates a batch-sized dataframe of co-occurrence counts.
            # Note these could just be summed rather than be batch size.
            if initialize:
                coo_df = batch_coo_df
            else:
                coo_df = coo_df.add(batch_coo_df, fill_value=0)
            if not batch_id % 100:
                logs.debug("coo_df is")
                logs.debug(coo_df)
            initialize = False
        logs.debug("Returning co-occurrence matrix")
        logs.debug(coo_df)
        return pd.DataFrame(coo_df)


    def set_columm_names(self, df_coo, subgroup):
        """
        :param df_coo: Co-occurrence counts for subgroup, length is num_words
        :return:
        """
        count_df = df_coo.set_index(self.vocab_counts_df.index)
        count_df = count_df.loc[~(count_df == 0).all(axis=1)]
        count_df.columns = [subgroup + "-count"]
        count_df[subgroup + "-count"] = count_df[subgroup + "-count"].astype(
            int)
        print("count_df is")
        print(count_df)
        return count_df

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
        p_subgroup_g_word = (
                vocab_cooc_df[subgroup + "-count"] / self.vocab_counts_df[
            "count"]
        ).dropna()
        logs.info("p_subgroup_g_word is")
        logs.info(p_subgroup_g_word)
        pmi_df = pd.DataFrame()
        pmi_df[subgroup + "-pmi"] = np.log(p_subgroup_g_word / subgroup_prob)
        # Note: A potentially faster solution for adding count, npmi,
        # can be based on this zip idea:
        # df_test['size_kb'],  df_test['size_mb'], df_test['size_gb'] =
        # zip(*df_test['size'].apply(sizes))
        return pmi_df.dropna()

    def calc_nPMI(self, pmi_df, vocab_cooc_df, subgroup):
        """
        # nPMI additionally divides by -log(p(x,y)) = -log(p(x|y)p(y))
        #                                           = -log(p(word|subgroup)p(word))
        """
        p_word_g_subgroup = vocab_cooc_df[subgroup + "-count"] / sum(
            vocab_cooc_df[subgroup + "-count"]
        )
        p_word = pmi_df.apply(
            lambda x: self.vocab_counts_df.loc[x.name]["proportion"], axis=1
        )
        normalize_pmi = -np.log(p_word_g_subgroup * p_word)
        npmi_df = pd.DataFrame()
        npmi_df[subgroup + "-npmi"] = pmi_df[subgroup + "-pmi"] / normalize_pmi
        return npmi_df.dropna()


def count_words_per_sentence(tokenized_sentence_df, vocab_counts_df,
                             num_batches):
    # Counts the number of each vocabulary item per-sentence in batches.
    logs.info("Creating co-occurrence matrix for nPMI calculations.")
    word_cnt_per_sentence = []
    logs.info(tokenized_sentence_df)
    batches = np.linspace(0, tokenized_sentence_df.shape[0],
                          num_batches).astype(int)
    i = 0
    # Creates list of size (# batches x # sentences)
    while i < len(batches) - 1:
        # Makes matrix shape: batch size (# sentences) x # words,
        # with the occurrence of each word per sentence.
        # vocab_counts_df.index is the vocabulary.
        mlb = MultiLabelBinarizer(classes=vocab_counts_df.index)
        if i % 100 == 0:
            logs.debug(
                "%s of %s sentence binarize batches." % (
                str(i), str(len(batches)))
            )
        mlb_series = mlb.fit_transform(
            tokenized_sentence_df[batches[i]:batches[i + 1]])
        i += 1
        word_cnt_per_sentence.append(mlb_series)
    return word_cnt_per_sentence
