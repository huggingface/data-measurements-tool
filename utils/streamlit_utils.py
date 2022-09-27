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
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
import seaborn as sns
import statistics
import streamlit as st
import utils
import utils.dataset_utils as ds_utils
from st_aggrid import AgGrid, GridOptionsBuilder
from utils.dataset_utils import HF_DESC_FIELD, HF_FEATURE_FIELD, HF_LABEL_FIELD

logs = utils.prepare_logging(__file__)
st.set_option('deprecation.showPyplotGlobalUse', False)

# Note: Make sure to consider colorblind-friendly colors for your images! Ex:
# ["#332288", "#117733", "#882255", "#AA4499", "#CC6677", "#44AA99", "#DDCC77",
# "#88CCEE"]

pd.options.display.float_format = "{:,.3f}".format # '{:20,.2f}'.format

def sidebar_header():
    st.sidebar.markdown("""This demo showcases the 
    [dataset metrics as we develop them](https://huggingface.co/blog/data-measurements-tool).
    Right now this has:
    - dynamic loading of datasets in the lib
    - fetching config and info without downloading the dataset
    - propose the list of candidate text and label features to select.
    """, unsafe_allow_html=True,)


def sidebar_selection(ds_name_to_dict, column_id=""):
    ds_names = list(ds_name_to_dict.keys())
    with st.sidebar.expander(f"Choose dataset and field {column_id}",
                             expanded=True):
        # choose a dataset to analyze
        ds_name = st.selectbox(
            f"Choose dataset to explore{column_id}:",
            ds_names,
            index=ds_names.index("hate_speech18"),
        )
        # choose a config to analyze
        ds_configs = ds_name_to_dict[ds_name]
        # special handling for the largest-by-far dataset, C4
        if ds_name == "c4":
            config_names = ['en', 'en.noblocklist', 'realnewslike']
        else:
            config_names = list(ds_configs.keys())
        config_name = st.selectbox(
            f"Choose configuration{column_id}:",
            config_names,
            index=0,
        )
        # choose a subset of num_examples
        ds_config = ds_configs[config_name]
        text_features = ds_config[HF_FEATURE_FIELD]["string"]
        # TODO @yacine: Explain what this is doing and why eg tp[0] could = "id"
        text_field = st.selectbox(
            f"Which text feature from the {column_id} dataset would you like to"
            f" analyze?",
            [("text",)]
            if ds_name == "c4"
            else [tp for tp in text_features if tp[0] != "id"],
        )
        # Choose a split and dataset size
        avail_splits = list(ds_config["splits"].keys())
        # 12.Nov note: Removing "test" because those should not be examined
        # without discussion of pros and cons, which we haven't done yet.
        if "test" in avail_splits:
            avail_splits.remove("test")
        split = st.selectbox(
            f"Which split from the{column_id} dataset would you like to "
            f"analyze?",
            avail_splits,
            index=0,
        )
        label_field, label_names = (
            ds_name_to_dict[ds_name][config_name][HF_FEATURE_FIELD][
                HF_LABEL_FIELD][0]
            if len(
                ds_name_to_dict[ds_name][config_name][HF_FEATURE_FIELD][
                    HF_LABEL_FIELD]
            ) > 0
            else ((), [])
        )
        return {
            "dset_name": ds_name,
            "dset_config": config_name,
            "split_name": split,
            "text_field": text_field,
            "label_field": label_field,
            "label_names": label_names,
        }


def expander_header(dstats, ds_name_to_dict, column_id=""):
    with st.expander(f"Dataset Description{column_id}"):
        st.markdown(
            ds_name_to_dict[dstats.dset_name][dstats.dset_config][HF_DESC_FIELD]
        )
        st.dataframe(dstats.dset_peek)


def expander_general_stats(dstats, column_id=""):
    with st.expander(f"General Text Statistics{column_id}"):
        st.caption(
            "Use this widget to check whether the terms you see most "
            "represented in the dataset make sense for the goals of the dataset."
        )
        st.markdown("There are {0} total words".format(str(dstats.total_words)))
        st.markdown(
            "There are {0} words after removing closed "
            "class words".format(str(dstats.total_open_words))
        )
        st.markdown(
            "The most common "
            "[open class words](https://dictionary.apa.org/open-class-words) "
            "and their counts are: "
        )
        st.dataframe(dstats.sorted_top_vocab_df)
        st.markdown(
            "There are {0} missing values in the dataset.".format(
                str(dstats.text_nan_count)
            )
        )
        if dstats.dups_frac > 0:
            st.markdown(
                "The dataset is {0}% duplicates. "
                "For more information about the duplicates, "
                "click the 'Duplicates' tab below.".format(
                    str(round(dstats.dups_frac * 100, 2)))
            )
        else:
            st.markdown("There are 0 duplicate items in the dataset. ")


def expander_label_distribution(dstats, column_id=""):
    with st.expander(f"Label Distribution{column_id}", expanded=False):
        st.caption(
            "Use this widget to see how balanced the labels in your dataset are."
        )
        if dstats.label_obj.fig_labels:
            st.plotly_chart(dstats.label_obj.fig_labels, use_container_width=True)
        else:
            st.markdown("No labels were found in the dataset")


def expander_text_lengths(dstats, column_id=""):
    _TEXT_LENGTH_CAPTION = (
        "Use this widget to identify outliers, particularly suspiciously long "
        "outliers."
    )
    with st.expander(f"Text Lengths{column_id}", expanded=False):
        st.caption(_TEXT_LENGTH_CAPTION)
        st.markdown(
            "Below, you can see how the lengths of the text instances in your "
            "dataset are distributed."
        )
        st.markdown(
            "Any unexpected peaks or valleys in the distribution may help to "
            "identify instances you want to remove or augment."
        )
        st.markdown(
            "### Here is the count of different text lengths in "
            "your dataset:"
        )
        # When matplotlib first creates this, it's a Figure.
        # Once it's saved, then read back in,
        # it's an ndarray that must be displayed using st.image
        # (I know, lame).
        if isinstance(dstats.length_obj.fig_lengths, Figure):
            st.pyplot(dstats.length_obj.fig_lengths, use_container_width=True)
        else:
            try:
                st.image(dstats.length_obj.fig_lengths)
            except Exception as e:
                logs.exception("Hit exception for lengths figure:")
                logs.exception(e)
        st.markdown(
            "The average length of text instances is **"
            + str(round(dstats.length_obj.avg_length, 2))
            + " words**, with a standard deviation of **"
            + str(round(dstats.length_obj.std_length, 2))
            + "**."
        )
        if dstats.length_obj.lengths_df is not None:
            start_id_show_lengths = st.selectbox(
                "Show examples of length:",
                np.sort(dstats.length_obj.lengths_df["length"].unique())[::-1].tolist(),
                key=f"select_show_length_{column_id}",
            )
            st.table(
                dstats.length_obj.lengths_df[
                    dstats.length_obj.lengths_df["length"] == start_id_show_lengths
                ].set_index("length")
            )


def expander_text_duplicates(dstats, column_id=""):
    with st.expander(f"Text Duplicates{column_id}", expanded=False):
        st.caption(
            "Use this widget to identify text strings that appear more than "
            "once."
        )
        st.markdown(
            "A model's training and testing may be negatively affected by "
            "unwarranted duplicates "
            "([Lee et al., 2021](https://arxiv.org/abs/2107.06499))."
        )
        st.markdown("------")
        st.write(
            "### Here is the list of all the duplicated items and their counts "
            "in the dataset."
        )
        if not dstats.duplicates_results:
            st.write("There are no duplicates in this dataset! 🥳")
        else:
            st.write("The fraction of the data that is a duplicate is:")
            st.write(str(round(dstats.dups_frac, 4)))
            # TODO: Check if this is slow when the size is large --
            # Should we store as dataframes?
            # Dataframes allow this to be interactive.
            st.dataframe(ds_utils.counter_dict_to_df(dstats.dups_dict))


def expander_text_perplexities(dstats, column_id=""):
    with st.expander(f"Text Perplexities{column_id}", expanded=False):
        st.caption(
            "Use this widget to identify text perplexities from GPT-2."
        )
        st.markdown(
            """
            Outlier perplexities, especially very high values, could highlight 
            an issue with an example. Smaller variations should be interpreted 
            with more care, as they indicate how similar to the GPT-2 training 
            corpus the examples are rather than being reflective of general 
            linguistic properties.
            For more information on GPT-2, 
            see its [model card](https://hf.co/gpt2).
            """
        )
        st.markdown("------")
        st.write(
            "### Here is the list of the examples in the dataset, sorted by "
            "GPT-2 perplexity:"
        )
        if dstats.perplexities_df is None or dstats.perplexities_df.empty:
            st.write(
                "Perplexities have not been computed yet for this dataset, or "
                "this dataset is too large for the UI (> 1,000,000 examples).")
        else:
            st.dataframe(dstats.perplexities_df.reset_index(drop=True))


def expander_npmi_description(min_vocab):
    _NPMI_CAPTION = (
        "Use this widget to identify problematic biases and stereotypes in "
        "your data."
    )
    _NPMI_CAPTION1 = """
    nPMI scores for a word help to identify potentially
    problematic associations, ranked by how close the association is."""
    _NPMI_CAPTION2 = """
    nPMI bias scores for paired words help to identify how word
    associations are skewed between the selected selected words
    ([Aka et al., 2021](https://arxiv.org/abs/2103.03417)).
    """

    st.caption(_NPMI_CAPTION)
    st.markdown(_NPMI_CAPTION1)
    st.markdown(_NPMI_CAPTION2)
    st.markdown("  ")
    st.markdown(
        "You can select from gender and sexual orientation "
        "identity terms that appear in the dataset at least %s "
        "times." % min_vocab
    )
    st.markdown(
        "The resulting ranked words are those that co-occur with both "
        "identity terms.  "
    )
    st.markdown(
        "The more *positive* the score, the more associated the word is with "
        "the first identity term.  "
        "The more *negative* the score, the more associated the word is with "
        "the second identity term."
    )


def expander_zipf(dstats, column_id=""):
    z = dstats.z
    zipf_fig = dstats.zipf_fig
    with st.expander(
        f"Vocabulary Distribution{column_id}: Zipf's Law Fit", expanded=False
    ):
        try:
            _ZIPF_CAPTION = """This shows how close the observed language is to an ideal
            natural language distribution following [Zipf's law](https://en.wikipedia.org/wiki/Zipf%27s_law),
            calculated by minimizing the [Kolmogorov-Smirnov (KS) statistic](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test)."""

            powerlaw_eq = r"""p(x) \propto x^{- \alpha}"""
            zipf_summary = (
                    "The optimal alpha based on this dataset is: **"
                    + str(round(z.alpha, 2))
                    + "**, with a KS distance of: **"
                    + str(round(z.ks_distance, 2))
            )
            zipf_summary += (
                    "**.  This was fit with a minimum rank value of: **"
                    + str(int(z.xmin))
                    + "**, which is the optimal rank *beyond which* the scaling regime of the power law fits best."
            )

            alpha_warning = "Your alpha value is a bit on the high side, which means that the distribution over words in this dataset is a bit unnatural. This could be due to non-language items throughout the dataset."
            xmin_warning = "The minimum rank for this fit is a bit on the high side, which means that the frequencies of your most common words aren't distributed as would be expected by Zipf's law."
            fit_results_table = pd.DataFrame.from_dict(
                {
                    r"Alpha:": [str("%.2f" % z.alpha)],
                    "KS distance:": [str("%.2f" % z.ks_distance)],
                    "Min rank:": [str("%s" % int(z.xmin))],
                },
                columns=["Results"],
                orient="index",
            )
            fit_results_table.index.name = column_id
            st.caption(
                "Use this widget for the counts of different words in your dataset, measuring the difference between the observed count and the expected count under Zipf's law."
            )
            st.markdown(_ZIPF_CAPTION)
            st.write(
                """
            A Zipfian distribution follows the power law: $p(x) \propto x^{-α}$
    with an ideal α value of 1."""
            )
            st.markdown(
                "In general, an alpha greater than 2 or a minimum rank greater than 10 (take with a grain of salt) means that your distribution is relativaly _unnatural_ for natural language. This can be a sign of mixed artefacts in the dataset, such as HTML markup."
            )
            st.markdown(
                "Below, you can see the counts of each word in your dataset vs. the expected number of counts following a Zipfian distribution."
            )
            st.markdown("-----")
            st.write("### Here is your dataset's Zipf results:")
            st.dataframe(fit_results_table)
            st.write(zipf_summary)
            # TODO: Nice UI version of the content in the comments.
            # st.markdown("\nThe KS test p-value is < %.2f" % z.ks_test.pvalue)
            # if z.ks_test.pvalue < 0.01:
            #    st.markdown(
            #        "\n Great news! Your data fits a powerlaw with a minimum KS " "distance of %.4f" % z.distance)
            # else:
            #    st.markdown("\n Sadly, your data does not fit a powerlaw. =(")
            # st.markdown("Checking the goodness of fit of our observed distribution")
            # st.markdown("to the hypothesized power law distribution")
            # st.markdown("using a Kolmogorov–Smirnov (KS) test.")
            st.plotly_chart(zipf_fig, use_container_width=True)
            if z.alpha > 2:
                st.markdown(alpha_warning)
            if z.xmin > 5:
                st.markdown(xmin_warning)
        except:
            st.write("Under construction!")


def npmi_widget(dstats, column_id=""):
    """
    Part of the UI, but providing for interaction.
    :param column_id:
    :param dstats:
    :return:
    """
    min_vocab = dstats.min_vocab_count
    npmi_stats = dstats.npmi_obj
    available_terms = npmi_stats.avail_identity_terms
    with st.expander(f"Word Association{column_id}: nPMI", expanded=False):
        if npmi_stats and len(available_terms) > 0:
            expander_npmi_description(min_vocab)
            st.markdown("-----")
            term1 = st.selectbox(
                f"What is the first term you want to select?{column_id}",
                available_terms,
            )
            term2 = st.selectbox(
                f"What is the second term you want to select?{column_id}",
                reversed(available_terms),
            )
            try:
                joint_npmi_df = npmi_stats.get_display(term1, term2)
                npmi_show(joint_npmi_df)
            except Exception as e:
                logs.exception(e)
                st.markdown(
                    "**WARNING!** The nPMI for these terms has not been"
                    " pre-computed, please re-run caching."
                )
        else:
            st.markdown("No words found co-occurring with both of the selected identity"
                " terms.")


def npmi_show(paired_results):
    if paired_results.empty:
        st.markdown(
            "No words that co-occur enough times for results! Or there's a 🐛."
            "  Or we're still computing this one. 🤷")
    else:
        logs.debug("Results to be shown in streamlit are")
        logs.debug(paired_results)
        s = pd.DataFrame(
            paired_results.sort_values(paired_results.columns[0], ascending=True))
        s.index.name = "word"
        bias_col = s.filter(like="bias").columns
        #count_cols = s.filter(like="count").columns
        # Keep the dataframe from being crazy big.
        if s.shape[0] > 10000:
            bias_thres = max(abs(s[s[0]][5000]),
                             abs(s[s[0]][-5000]))
            logs.info(f"filtering with bias threshold: {bias_thres}")
            s_filtered = s[s[0].abs() > bias_thres]
        else:
            s_filtered = s
        cm = sns.palplot(sns.diverging_palette(270, 36, s=99, l=48, n=16))
        out_df = s_filtered.style.background_gradient(subset=bias_col, cmap=cm).format(formatter="{:,.3f}").set_properties(**{"align": "center", "width":"100em"}).set_caption("nPMI scores between the selected identity terms and the words they both co-occur with")
        #set_properties(subset=count_cols, **{"width": "10em", "text-align": "center"}).
        # .format(subset=count_cols, formatter=int).
        #.format(subset=bias_col, formatter="{:,.3f}")
        st.write("### Here is your dataset's bias results:")
        st.dataframe(out_df)
