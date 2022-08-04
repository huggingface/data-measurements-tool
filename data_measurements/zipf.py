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

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import powerlaw
from scipy.stats import ks_2samp
from scipy.stats import zipf as zipf_lib
import json
import os
from os.path import join as pjoin
import plotly.graph_objects as go


# treating inf values as NaN as well

pd.set_option("use_inf_as_na", True)

logs = logging.getLogger(__name__)
logs.setLevel(logging.INFO)
logs.propagate = False

if not logs.handlers:

    Path("./log_files").mkdir(exist_ok=True)

    # Logging info to log file
    file = logging.FileHandler("./log_files/zipf.log")
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


class Zipf:
    def __init__(self, vocab_counts_df, zipf_dict=None, CNT="count", PROP="prop"):
        # TODO: Should handle just loading the dict instead?
        self.vocab_counts_df = vocab_counts_df
        # Strings used in the input dictionary
        self.cnt_str = CNT
        self.prop_str = PROP
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
            self.word_ranks_unique = list(np.arange(1, len(self.word_counts_unique) + 1))
        self.predicted_counts = None #self.calc_zipf_counts()
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

    def calc_fit(self):
        """
        Uses the powerlaw package to fit the observed frequencies to a zipfian distribution.
        We use the KS-distance to fit, as that seems more appropriate that MLE.
        :param vocab_counts_df:
        :return:
        """
        logs.info("Fitting based on input vocab counts.")

        self._make_rank_column()
        # Note another method for determining alpha might be defined by
        # (Newman, 2005): alpha = 1 + n * sum(ln( xi / xmin )) ^ -1
        self.fit = powerlaw.Fit(self.observed_counts, fit_method="KS", discrete=True)
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
        self.predicted_counts = self.calc_zipf_counts()


    def _make_rank_column(self):
        # TODO: These proportions may have already been calculated.
        prop_denom = float(sum(self.vocab_counts_df[self.cnt_str]))
        count_prop = self.vocab_counts_df[self.cnt_str] / prop_denom
        self.vocab_counts_df[self.prop_str] = count_prop
        rank_column = self.vocab_counts_df[self.cnt_str].rank(
            method="dense", numeric_only=True, ascending=False
        )
        self.vocab_counts_df["rank"] = rank_column.astype("int64")

    def calc_zipf_counts(self):
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
        # TODO: Limit from above xmin to below xmax, not just above xmin.
        logs.info(self.word_counts_unique)
        logs.info(self.xmin)
        logs.info(self.xmax)
        # The subset of words that fit
        word_counts_fit_unique = self.word_counts_unique[self.xmin + 1: self.xmax]
        pmf_mass = float(sum(word_counts_fit_unique))
        zipf_counts = np.array(
            [self._estimate_count(rank, pmf_mass) for rank in self.word_ranks_unique]
        )
        return zipf_counts

    def _estimate_count(self, rank, pmf_mass):
        return int(round(zipf_lib.pmf(rank, self.alpha) * pmf_mass))

    def get_zipf_dict(self):
        zipf_dict = {"xmin": int(self.xmin), "xmax": int(self.xmax),
                     "alpha": float(self.alpha), "ks_distance": float(self.ks_distance),
                     "p-value": float(self.ks_test.pvalue),
                     "word_counts_unique": [int(count) for count in self.word_counts_unique],
                     "word_ranks_unique": [int(rank) for rank in self.word_ranks_unique]}
        return zipf_dict

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


def load_zipf(zipf_dict):
    """ Defines the zipf attributes from what's stored in the dictionary,
        rather than computing things anew.
    """
    z = Zipf(None)
    z.load(zipf_dict)
    return z

# TODO: This might fit better in its own file handling class
def get_zipf_fids(cache_path):
    zipf_cache_dir = pjoin(cache_path, "zipf")
    os.makedirs(zipf_cache_dir, exist_ok=True)
    ## Zipf cache files
    zipf_fid = pjoin(zipf_cache_dir, "zipf_basic_stats.json")
    zipf_fig_fid = pjoin(zipf_cache_dir, "zipf_fig.json")
    zipf_fig_html_fid = pjoin(zipf_cache_dir, "zipf_fig.html")
    return zipf_fid, zipf_fig_fid, zipf_fig_html_fid

def make_unique_rank_word_list(z):
    ranked_words = {}
    word_counts = z.word_counts_unique
    word_ranks = z.word_ranks_unique
    for count, rank in zip(word_counts, word_ranks):
        z.vocab_counts_df[z.vocab_counts_df[z.cnt_str] == count]["rank"] = rank
        ranked_words[rank] = ",".join(
            z.vocab_counts_df[z.vocab_counts_df[z.cnt_str] == count].index.astype(str)
        )  # Use the hovertext kw argument for hover text
    ranked_words_list = [wrds for rank, wrds in
                         sorted(ranked_words.items())]
    return ranked_words_list

def make_zipf_fig(z):
    xmin = z.xmin
    word_ranks_unique = z.word_ranks_unique
    observed_counts = z.observed_counts
    zipf_counts = z.predicted_counts #"] #self.calc_zipf_counts()
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

"""
    # TODO: Incorporate this function (not currently using)
    def fit_others(self, fit):
        st.markdown(
            "_Checking log likelihood ratio to see if the data is better explained by other well-behaved distributions..._"
        )
        # The first value returned from distribution_compare is the log likelihood ratio
        better_distro = False
        trunc = fit.distribution_compare("power_law", "truncated_power_law")
        if trunc[0] < 0:
            st.markdown("Seems a truncated power law is a better fit.")
            better_distro = True

        lognormal = fit.distribution_compare("power_law", "lognormal")
        if lognormal[0] < 0:
            st.markdown("Seems a lognormal distribution is a better fit.")
            st.markdown("But don't panic -- that happens sometimes with language.")
            better_distro = True

        exponential = fit.distribution_compare("power_law", "exponential")
        if exponential[0] < 0:
            st.markdown("Seems an exponential distribution is a better fit. Panic.")
            better_distro = True

        if not better_distro:
            st.markdown("\nSeems your data is best fit by a power law. Celebrate!!")
"""""