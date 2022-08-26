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
import numpy as np
import os
import pandas as pd
import plotly.graph_objects as go
import powerlaw
from os.path import join as pjoin
import utils
from scipy.stats import ks_2samp
from scipy.stats import zipf as zipf_lib

# treating inf values as NaN as well

pd.set_option("use_inf_as_na", True)

logs = utils.prepare_logging(__file__)


class Zipf:
    def __init__(self, vocab_counts_df, count_str="count",
                 proportion_str="prop"):
        self.vocab_counts_df = vocab_counts_df
        # Strings used in the input dictionary
        self.cnt_str = count_str
        self.prop_str = proportion_str
        self.alpha = None
        self.xmin = None
        self.xmax = None
        self.p = None
        self.ks_distance = None
        self.observed_counts = None
        self.word_counts_unique = None
        self.word_ranks_unique = None
        if self.vocab_counts_df is not None:
            self.observed_counts = self.vocab_counts_df[self.cnt_str].values
            self.word_counts_unique = list(set(self.observed_counts))
            self.word_ranks_unique = list(
                np.arange(1, len(self.word_counts_unique) + 1))
        self.zipf_dict = {"xmin": None, "xmax": None, "alpha": None,
                          "ks_distance": None, "p-value": None,
                          "word_ranks_unique": self.word_ranks_unique,
                          "word_counts_unique": self.word_counts_unique}
        self.fit = None
        self.predicted_counts = None

    def load(self, zipf_dict):
        self.zipf_dict = zipf_dict
        self.xmin = zipf_dict["xmin"]
        self.xmax = zipf_dict["xmax"]
        self.alpha = zipf_dict["alpha"]
        self.ks_distance = zipf_dict["ks_distance"]
        self.p = zipf_dict["p-value"]
        self.word_ranks_unique = zipf_dict["word_ranks_unique"]
        self.word_counts_unique = zipf_dict["word_counts_unique"]

    def get_zipf_dict(self):
        zipf_dict = {"xmin": int(self.xmin), "xmax": int(self.xmax),
                     "alpha": float(self.alpha),
                     "ks_distance": float(self.ks_distance),
                     "p-value": float(self.ks_test.pvalue),
                     "word_counts_unique": [int(count) for count in
                                            self.word_counts_unique],
                     "word_ranks_unique": [int(rank) for rank in
                                           self.word_ranks_unique]}
        return zipf_dict

    def calc_fit(self):
        """
        Uses the powerlaw package to fit the observed frequencies
        to a zipfian distribution.
        We use the KS-distance to fit, as that seems more appropriate that MLE.
        """
        logs.info("Fitting based on input vocab counts.")

        self._make_rank_column()
        # Note another method for determining alpha might be defined by
        # (Newman, 2005): alpha = 1 + n * sum(ln( xi / xmin )) ^ -1
        self.fit = powerlaw.Fit(self.observed_counts, fit_method="KS",
                                discrete=True)
        # This should probably be a pmf (not pdf); using discrete=True above.
        # original_data=False uses only the fitted data (within xmin and xmax).
        # pdf_bin_edges: The portion of the data within the bin.
        # observed_pdf: The probability density function (normalized histogram)
        # of the data.
        pdf_bin_edges, observed_pdf = self.fit.pdf(original_data=False)
        # See the 'Distribution' class described here for info:
        # https://pythonhosted.org/powerlaw/#powerlaw.Fit.pdf
        theoretical_distro = self.fit.power_law
        # The probability density function (normalized histogram) of the
        # theoretical distribution.
        predicted_pdf = theoretical_distro.pdf()
        self._set_fit_vars(observed_pdf, predicted_pdf, theoretical_distro)

    def _set_fit_vars(self, observed_pdf, predicted_pdf, theoretical_distro):
        # !!!! CRITICAL VALUE FOR ZIPF !!!!
        self.alpha = theoretical_distro.alpha
        # Exclusive xmin: The optimal xmin *beyond which* the scaling regime of
        # the power law fits best.
        self.xmin = int(theoretical_distro.xmin)
        self.xmax = theoretical_distro.xmax
        # Can be None if there isn't an xmax returned;
        # this handles that.
        self._set_xmax()
        self.ks_distance = theoretical_distro.KS()
        self.ks_test = ks_2samp(observed_pdf, predicted_pdf)
        self.p = self.ks_test[1]
        logs.info("KS test:")
        logs.info(self.ks_test)
        self.predicted_counts = self._calc_zipf_counts()

    def _make_rank_column(self):
        # TODO: These proportions may have already been calculated.
        prop_denom = float(sum(self.vocab_counts_df[self.cnt_str]))
        count_prop = self.vocab_counts_df[self.cnt_str] / prop_denom
        self.vocab_counts_df[self.prop_str] = count_prop
        rank_column = self.vocab_counts_df[self.cnt_str].rank(
            method="dense", numeric_only=True, ascending=False
        )
        self.vocab_counts_df["rank"] = rank_column.astype("int64")

    def _calc_zipf_counts(self):
        """
        The fit is based on an optimal xmin (minimum rank)
        Let's use this to make count estimates for the zipf fit,
        by multiplying the fitted pmf value by the sum of counts above xmin.
        :return: array of count values following the fitted pmf.
        """
        logs.info("Getting predicted counts.")
        if not self.alpha:
            print("Have not yet fit -- need the alpha value.")
            print("Fitting now...")
            self.calc_fit()
        logs.info(self.word_counts_unique)
        logs.info(self.xmin)
        logs.info(self.xmax)
        # The subset of words that fit
        word_counts_fit_unique = self.word_counts_unique[
                                 self.xmin + 1: self.xmax]
        pmf_mass = float(sum(word_counts_fit_unique))
        zipf_counts = np.array(
            [self._estimate_count(rank, pmf_mass) for rank in
             self.word_ranks_unique]
        )
        return zipf_counts

    def _estimate_count(self, rank, pmf_mass):
        return int(round(zipf_lib.pmf(rank, self.alpha) * pmf_mass))

    def _set_xmax(self):
        """
        xmax is usually None, so we add some handling to set it as the
        maximum rank in the dataset.
        :param xmax:
        :return:
        """
        if self.xmax is not None:
            self.xmax = int(xmax)
        elif self.word_counts_unique:
            self.xmax = int(len(self.word_counts_unique))
        elif self.word_ranks_unique:
            self.xmax = int(len(self.word_ranks_unique))


# TODO: This might fit better in its own file handling class?
def get_zipf_fids(cache_path):
    zipf_cache_dir = pjoin(cache_path, "zipf")
    os.makedirs(zipf_cache_dir, exist_ok=True)
    # Zipf cache files
    zipf_fid = pjoin(zipf_cache_dir, "zipf_basic_stats.json")
    zipf_fig_fid = pjoin(zipf_cache_dir, "zipf_fig.json")
    zipf_fig_html_fid = pjoin(zipf_cache_dir, "zipf_fig.html")
    return zipf_fid, zipf_fig_fid, zipf_fig_html_fid


def make_unique_rank_word_list(z):
    """
    Function to help with the figure, creating strings for the hovertext.
    """
    ranked_words = {}
    word_counts = z.word_counts_unique
    word_ranks = z.word_ranks_unique
    for count, rank in zip(word_counts, word_ranks):
        z.vocab_counts_df[z.vocab_counts_df[z.cnt_str] == count]["rank"] = rank
        ranked_words[rank] = ",".join(
            z.vocab_counts_df[
                z.vocab_counts_df[z.cnt_str] == count].index.astype(str)
        )  # Use the hovertext kw argument for hover text
    ranked_words_list = [wrds for rank, wrds in
                         sorted(ranked_words.items())]
    return ranked_words_list


def make_zipf_fig(z):
    xmin = z.xmin
    word_ranks_unique = z.word_ranks_unique
    observed_counts = z.observed_counts
    zipf_counts = z.predicted_counts  # "] #self.calc_zipf_counts()
    ranked_words_list = make_unique_rank_word_list(z)
    layout = go.Layout(xaxis=dict(range=[0, 100]))
    fig = go.Figure(
        data=[
            go.Bar(
                x=word_ranks_unique,
                y=observed_counts,
                hovertext=ranked_words_list,
                name="Word Rank Frequency",
            )
        ],
        layout=layout,
    )
    fig.add_trace(
        go.Scatter(
            x=word_ranks_unique[xmin: len(word_ranks_unique)],
            y=zipf_counts[xmin: len(word_ranks_unique)],
            hovertext=ranked_words_list[xmin: len(word_ranks_unique)],
            line=go.scatter.Line(color="crimson", width=3),
            name="Zipf Predicted Frequency",
        )
    )
    # Customize aspect
    # fig.update_traces(marker_color='limegreen',
    #                  marker_line_width=1.5, opacity=0.6)
    fig.update_layout(
        title_text="Word Counts, Observed and Predicted by Zipf")
    fig.update_layout(xaxis_title="Word Rank")
    fig.update_layout(yaxis_title="Frequency")
    fig.update_layout(
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.10))
    return fig
