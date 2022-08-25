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

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

# Might be nice to print to log instead? Happens when we drop closed class.
warnings.filterwarnings(action="ignore", category=UserWarning)
# When we divide by 0 in log
np.seterr(divide="ignore")

# treating inf values as NaN as well
pd.set_option("use_inf_as_na", True)

logs = logging.getLogger(__name__)
logs.setLevel(logging.INFO)
logs.propagate = False

if not logs.handlers:

    Path("./log_files").mkdir(exist_ok=True)

    # Logging info to log file
    file = logging.FileHandler("./log_files/npmi.log")
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

_NUM_BATCHES = 500


class nPMI:
    # TODO: Expand beyond pairwise
    def __init__(
        self,
        vocab_counts_df,
        tokenized_df,
        tokenized_col_name="tokenized_text",
    ):
        logs.info("Initiating npmi class.")
        logs.info("vocab is")
        logs.info(vocab_counts_df)
        self.vocab_counts_df = vocab_counts_df
        logs.info("tokenized is")
        self.tokenized_df = tokenized_df
        logs.info(self.tokenized_df)
        self.tokenized_col_name = tokenized_col_name
        # self.mlb_list holds num batches x num_sentences
        self.mlb_list = []

    def binarize_words_in_sentence(self):
        logs.info("Creating co-occurrence matrix for PMI calculations.")
        batches = np.linspace(0, self.tokenized_df.shape[0], _NUM_BATCHES).astype(int)
        i = 0
        # Creates list of size (# batches x # sentences)
        while i < len(batches) - 1:
            # Makes a sparse matrix (shape: # sentences x # words),
            # with the occurrence of each word per sentence.
            mlb = MultiLabelBinarizer(classes=self.vocab_counts_df.index)
            logs.info(
                "%s of %s sentence binarize batches." % (str(i), str(len(batches)))
            )
            # Returns series: batch size x num_words
            mlb_series = mlb.fit_transform(
                self.tokenized_df[self.tokenized_col_name][batches[i]:batches[i + 1]]
            )
            i += 1
            self.mlb_list.append(mlb_series)

    def calc_cooccurrences(self, subgroup, subgroup_idx):
        initialize = True
        coo_df = None
        # Big computation here!  Should only happen once.
        logs.info(
            "Approaching big computation! Here, we binarize all words in the sentences, making a sparse matrix of sentences."
        )
        if not self.mlb_list:
            self.binarize_words_in_sentence()
        for batch_id in range(len(self.mlb_list)):
            logs.info(
                "%s of %s co-occurrence count batches"
                % (str(batch_id), str(len(self.mlb_list)))
            )
            # List of all the sentences (list of vocab) in that batch
            batch_sentence_row = self.mlb_list[batch_id]
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
            logs.info("Removing 0 counts, subgroup_df is")
            logs.info(subgroup_df)
            mlb_subgroup_only = sent_batch_df[sent_batch_df[subgroup_idx] > 0]
            logs.info("mlb subgroup only is")
            logs.info(mlb_subgroup_only)
            # Create cooccurrence matrix for the given subgroup and all words.
            logs.info("Now we do the T.dot approach for co-occurrences")
            batch_coo_df = pd.DataFrame(mlb_subgroup_only.T.dot(subgroup_df))

            # Creates a batch-sized dataframe of co-occurrence counts.
            # Note these could just be summed rather than be batch size.
            if initialize:
                coo_df = batch_coo_df
            else:
                coo_df = coo_df.add(batch_coo_df, fill_value=0)
            logs.info("coo_df is")
            logs.info(coo_df)
            initialize = False
        logs.info("Returning co-occurrence matrix")
        logs.info(coo_df)
        return pd.DataFrame(coo_df)

    @staticmethod
    def calc_paired_metrics(subgroup_pair, subgroup_npmi_dict):
        """
        Calculates nPMI metrics between paired subgroups.
        Special handling for a subgroup paired with itself.
        :param subgroup_npmi_dict: vocab, pmi, and npmi for each subgroup.
        :return:

        Args:
            subgroup_pair:
        """
        paired_results_dict = {"npmi": {}, "pmi": {}, "count": {}}
        # Canonical ordering. This is done previously, but just in case...
        subgroup1, subgroup2 = sorted(subgroup_pair)
        vocab_cooc_df1, pmi_df1, npmi_df1 = subgroup_npmi_dict[subgroup1]
        logs.info("vocab cooc")
        logs.info(vocab_cooc_df1)
        if subgroup1 == subgroup2:
            shared_npmi_df = npmi_df1
            shared_pmi_df = pmi_df1
            shared_vocab_cooc_df = vocab_cooc_df1
        else:
            vocab_cooc_df2, pmi_df2, npmi_df2 = subgroup_npmi_dict[subgroup2]
            logs.info("vocab cooc2")
            logs.info(vocab_cooc_df2)
            # Note that lsuffix and rsuffix should not come into play.
            shared_npmi_df = npmi_df1.join(
                npmi_df2, how="inner", lsuffix="1", rsuffix="2"
            )
            shared_pmi_df = pmi_df1.join(pmi_df2, how="inner", lsuffix="1", rsuffix="2")
            shared_vocab_cooc_df = vocab_cooc_df1.join(
                vocab_cooc_df2, how="inner", lsuffix="1", rsuffix="2"
            )
            shared_vocab_cooc_df = shared_vocab_cooc_df.dropna()
            shared_vocab_cooc_df = shared_vocab_cooc_df[
                shared_vocab_cooc_df.index.notnull()
            ]
            logs.info("shared npmi df")
            logs.info(shared_npmi_df)
            logs.info("shared vocab df")
            logs.info(shared_vocab_cooc_df)
        npmi_bias = (
            shared_npmi_df[subgroup1 + "-npmi"] - shared_npmi_df[subgroup2 + "-npmi"]
        )
        paired_results_dict["npmi-bias"] = npmi_bias.dropna()
        paired_results_dict["npmi"] = shared_npmi_df.dropna()
        paired_results_dict["pmi"] = shared_pmi_df.dropna()
        paired_results_dict["count"] = shared_vocab_cooc_df.dropna()
        return paired_results_dict

    def calc_metrics(self, subgroup):
        # Index of the subgroup word in the sparse vector
        subgroup_idx = self.vocab_counts_df.index.get_loc(subgroup)
        logs.info("Calculating co-occurrences...")
        df_coo = self.calc_cooccurrences(subgroup, subgroup_idx)
        vocab_cooc_df = self.set_idx_cols(df_coo, subgroup)
        logs.info(vocab_cooc_df)
        logs.info("Calculating PMI...")
        pmi_df = self.calc_PMI(vocab_cooc_df, subgroup)
        logs.info(pmi_df)
        logs.info("Calculating nPMI...")
        npmi_df = self.calc_nPMI(pmi_df, vocab_cooc_df, subgroup)
        logs.info(npmi_df)
        return vocab_cooc_df, pmi_df, npmi_df

    def set_idx_cols(self, df_coo, subgroup):
        """
        :param df_coo: Co-occurrence counts for subgroup, length is num_words
        :return:
        """
        count_df = df_coo.set_index(self.vocab_counts_df.index)
        count_df.columns = [subgroup + "-count"]
        count_df[subgroup + "-count"] = count_df[subgroup + "-count"].astype(int)
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
            vocab_cooc_df[subgroup + "-count"] / self.vocab_counts_df["count"]
        )
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

class nPMIStatisticsCacheClass:
    """ "Class to interface between the app and the nPMI class
    by calling the nPMI class with the user's selections."""

    def __init__(self, dataset_stats, use_cache=False):
        self.live = dataset_stats.live
        self.dstats = dataset_stats
        self.pmi_cache_path = pjoin(self.dstats.cache_path, "pmi_files")
        if not isdir(self.pmi_cache_path):
            logs.warning(
                "Creating pmi cache directory %s." % self.pmi_cache_path)
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
                and json.load(open(self.npmi_terms_fid))[
            "available terms"] != []
        ):
            available_terms = json.load(open(self.npmi_terms_fid))[
                "available terms"]
        else:
            true_false = [
                term in self.dstats.vocab_counts_df.index for term in
                self.termlist
            ]
            word_list_tmp = [x for x, y in zip(self.termlist, true_false) if y]
            true_false_counts = [
                self.dstats.vocab_counts_df.loc[
                    word, CNT] >= self.min_vocab_count
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
        subgroup_files = define_subgroup_files(subgroup_pair,
                                               self.pmi_cache_path)
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
                    write_subgroup_npmi_data(subgroup, subgroup_dict,
                                             subgroup_files)
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
        joint_npmi_df, subgroup_dict = self.do_npmi(subgroup_pair,
                                                    subgroup_dict)
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
        paired_results = npmi_obj.calc_paired_metrics(subgroup_pair,
                                                      subgroup_dict)
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