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
        self.npmi_terms_json_fid = pjoin(dstats.dataset_cache_dir, "npmi_terms.json")
        self.npmi_df_fid = pjoin(dstats.dataset_cache_dir, "npmi_results.feather")
        # TODO: Users ideally can type in whatever words they want.
        # This is the full list of terms.
        self.identity_terms = identity_terms
        logs.info("Using term list:")
        logs.info(self.identity_terms)
        # identity_terms terms that are available more than _MIN_VOCAB_COUNT times
        self.avail_identity_terms = []
        # TODO: Let users specify
        self.open_class_only = True
        self.joint_npmi_df_dict = {}
        self.subgroup_results_dict = {}
        self.subgroup_files = {}
        self.npmi_results = None

    def run_DMT_processing(self, load_only=False):
        self.load_or_prepare_avail_identity_terms(
            load_only=load_only)
        self.load_or_prepare_npmi_results(load_only)

    def load_or_prepare_npmi_results(self, load_only):
        # If we're trying to use the cache of available terms
        if self.use_cache:
            self.npmi_results = self._load_npmi_results_cache()
        # Figure out the identity terms if we're not just loading from cache
        if not load_only:
            if not self.npmi_results:
                npmi_obj = nPMI(self.dstats.vocab_counts_df,
                                self.tokenized_sentence_df,
                                self.avail_identity_terms)
                self.npmi_results = npmi_obj.npmi_bias_df
            # Finish
            if self.save:
                self._write_cache()

    def load_or_prepare_avail_identity_terms(self, load_only=False):
        """
        Figures out what identity terms the user can select, based on whether
        they occur more than self.min_vocab_count times
        :return: Identity terms occurring at least self.min_vocab_count times.
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

    def _load_identity_cache(self):
        if exists(self.npmi_terms_json_fid):
            avail_identity_terms = json.load(open(self.npmi_terms_json_fid))
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

    def _write_cache(self):
        ds_utils.make_path(self.npmi_cache_path)
        if self.avail_identity_terms:
            ds_utils.write_json(self.avail_identity_terms, self.npmi_terms_json_fid)
        if self.npmi_results is not None:
            ds_utils.write_df(self.npmi_results, self.npmi_df_fid)

    def get_available_terms(self):
        return self.load_or_prepare_identity_terms()


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
        # Data structure that holds the npmi scores and differences
        # for different subgroups.
        self.npmi_bias_df = pd.DataFrame(index=vocab_counts_df.index)
        self.calc_metrics()
        #for idx1 in range(len(self.identity_terms)):
        #    subgroup1 = self.identity_terms[idx1]
        #    vocab_cooc_df, pmi_df, npmi_df1 = self.calc_metrics(subgroup1)
        #    for idx2 in range(idx1+1,len(self.identity_terms)):
        #        subgroup2 = self.identity_terms[idx2]
        #        vocab_cooc_df2, pmi_df2, npmi_df2 = self.calc_metrics(subgroup2)
        #        # Canonical ordering
        #        subgroup_tuple = sorted([subgroup1, subgroup2])
        #        self.npmi_bias_df[subgroup_tuple] = npmi_df1 - npmi_df2
        #        print(self.npmi_bias_df)
        # Dataframe of word co-occurrences
        # self.coo_df = self.count_cooccurrences(self.identity_terms, self.word_cnt_per_sentence)

    def calc_metrics(self):
        # Index of the subgroup word in the sparse vector
        subgroup_idxs = [self.vocab_counts_df.index.get_loc(subgroup) for subgroup in self.identity_terms]
        #logs.info("Calculating co-occurrences of %s..." % subgroup)
        self.calc_cooccurrences(subgroup_idxs)
        vocab_cooc_df = self.set_idx_cols(self.coo_df, subgroup)
        logs.info(vocab_cooc_df)
        logs.info("Calculating PMI...")
        pmi_df = self.calc_PMI(vocab_cooc_df, subgroup)
        logs.info(pmi_df)
        logs.info("Calculating nPMI...")
        npmi_df = self.calc_nPMI(pmi_df, vocab_cooc_df, subgroup)
        logs.info(npmi_df)
        return vocab_cooc_df, pmi_df, npmi_df

    def calc_cooccurrences(self, subgroup_idxs):
        """
        Returns the co-occurrence matrix with dimensions vocab x vocab
        """
        # Big computation here!  Should only happen once per subgroup.
        logs.info(
            "Approaching big computation! Here, we count all words in the sentences, making a matrix of sentences x vocab."
        )
        # For each batch, calculate how many times the subgroup identity term
        # co-occurs with the rest of the vocabulary
        for batch_id in range(len(self.word_cnt_per_sentence)):
            logs.debug(
                "%s of %s co-occurrence count batches"
                % (str(batch_id), str(len(self.word_cnt_per_sentence)))
            )
            # List of all the sentences (list of vocab) in that batch
            batch_sentence_row = self.word_cnt_per_sentence[batch_id]
            # Dataframe of # sentences in batch x vocabulary size
            sent_batch_df = pd.DataFrame(batch_sentence_row)
            print(subgroup_idxs)
            print(sent_batch_df[[35]])
            # sent_batch_df.loc[(sent_batch_df['col1'] == value) & (df['col2'] < value)]
            # Extract the set of sentences where the identity term appears
            identity_sentences_df = sent_batch_df[sent_batch_df[subgroup_idxs] > 0]
            print(identity_sentences_df)
            # Remove the rows for sentences where the term counts are all NaN.
            no_na = identity_sentences_df.dropna(how='all')
            # Remove the rows where the term counts are all 0.git add .
            mlb_subgroup_only = no_na.loc[(no_na != 0).any(axis=1)]
            #mlb_subgroup_only.columns = self.identity_terms
            print(mlb_subgroup_only)
            print(sent_batch_df)
            # Extract the sentences where the identity terms occur.
            #subgroup_df = subgroup_df[subgroup_df > 0]
            #print(subgroup_df)
            # Subgroup counts per-sentence for the given batch
            #mlb_subgroup_only = sent_batch_df[sent_batch_df[subgroup_idxs] > 0]
            # Calculate how much the subgroup term co-occurs with all the others
            self.update_cooc_matrix(batch_id, mlb_subgroup_only, sent_batch_df)

    def update_cooc_matrix(self, batch_id, mlb_subgroup_only, subgroup_df):
        # Create cooccurrence matrix for the given subgroup and all words.
        logs.debug("Doing the transpose for co-occurrences")
        batch_coo_df = pd.DataFrame(mlb_subgroup_only.T.dot(subgroup_df))
        # Creates a batch-sized dataframe of co-occurrence counts.
        # Note these could just be summed rather than be batch size.
        # Initialize the co-occurrence matrix with the correct number of columns
        if batch_id == 0:
            self.coo_df = batch_coo_df
        else:
            self.coo_df = self.coo_df.add(batch_coo_df, fill_value=0)
        logs.debug("Have co-occurrence matrix")
        logs.debug(self.coo_df)

    def set_idx_cols(self, df_coo, subgroup):
        """
        :param df_coo: Co-occurrence counts for subgroup, length is num_words
        :return:
        """
        count_df = df_coo.set_index(self.vocab_counts_df.index)
        count_df.columns = [subgroup + "-count"]
        count_df[subgroup + "-count"] = count_df[subgroup + "-count"].astype(
            int)
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
        )
        logs.info("p_subgroup_g_word is")
        logs.info(p_subgroup_g_word)
        pmi_df = pd.DataFrame()
        pmi_df[subgroup + "-pmi"] = np.log(p_subgroup_g_word / subgroup_prob)
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
        p_word_g_subgroup = vocab_cooc_df[subgroup + "-count"] / sum(
            vocab_cooc_df[subgroup + "-count"]
        )
        p_word = pmi_df.apply(
            lambda x: self.vocab_counts_df.loc[x.name]["proportion"], axis=1
        )
        normalize_pmi = -np.log(p_word_g_subgroup * p_word)
        npmi_df = pd.DataFrame()
        npmi_df[subgroup + "-npmi"] = pmi_df[subgroup + "-pmi"] / normalize_pmi
        return npmi_df


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
