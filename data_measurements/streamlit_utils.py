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

import statistics

import pandas as pd
import seaborn as sns
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder

from .dataset_utils import HF_DESC_FIELD, HF_FEATURE_FIELD, HF_LABEL_FIELD


def sidebar_header():
    st.sidebar.markdown(
        """
    This demo showcases the [dataset metrics as we develop them](https://github.com/huggingface/DataMeasurements).
    Right now this has:
    - dynamic loading of datasets in the lib
    - fetching config and info without downloading the dataset
    - propose the list of candidate text and label features to select
    We are still working on:
    - implementing all the current tools
    """,
        unsafe_allow_html=True,
    )


def sidebar_selection(ds_name_to_dict, column_id):
    ds_names = list(ds_name_to_dict.keys())
    with st.sidebar.expander(f"Choose dataset and field {column_id}", expanded=True):
        # choose a dataset to analyze
        ds_name = st.selectbox(
            f"Choose dataset to explore{column_id}:",
            ds_names,
            index=ds_names.index("hate_speech18"),
        )
        # choose a config to analyze
        ds_configs = ds_name_to_dict[ds_name]
        config_names = list(ds_configs.keys())
        config_name = st.selectbox(
            f"Choose configuration{column_id}:",
            config_names,
            index=0,
        )
        # choose a subset of num_examples
        # TODO: Handling for multiple text features
        ds_config = ds_configs[config_name]
        text_features = ds_config[HF_FEATURE_FIELD]["string"]
        # TODO @yacine: Explain what this is doing and why eg tp[0] could = "id"
        text_field = st.selectbox(
            f"Which text feature from the{column_id} dataset would you like to analyze?",
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
            f"Which split from the{column_id} dataset would you like to analyze?",
            avail_splits,
            index=0,
        )
        label_field, label_names = (
            ds_name_to_dict[ds_name][config_name][HF_FEATURE_FIELD][HF_LABEL_FIELD][0]
            if len(
                ds_name_to_dict[ds_name][config_name][HF_FEATURE_FIELD][HF_LABEL_FIELD]
            )
            > 0
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


def expander_header(dstats, ds_name_to_dict, column_id):
    with st.expander(f"Dataset Description{column_id}"):
        st.markdown(
            ds_name_to_dict[dstats.dset_name][dstats.dset_config][HF_DESC_FIELD]
        )
        st.dataframe(dstats.get_dataset_peek())


def expander_general_stats(dstats, column_id):
    with st.expander(f"General Text Statistics{column_id}"):
        st.caption(
            "Use this widget to check whether the terms you see most represented"
            " in the dataset make sense for the goals of the dataset."
        )
        st.markdown(
            "There are {0} total words".format(str(dstats.total_words))
        )
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
        st.markdown(
            "There are {0} duplicate items in the dataset. "
            "For more information about the duplicates, "
            "click the 'Duplicates' tab below.".format(
                str(dstats.dedup_total)
            )
        )


### Show the label distribution from the datasets
def expander_label_distribution(label_df, fig_labels, column_id):
    with st.expander(f"Label Distribution{column_id}", expanded=False):
        st.caption(
            "Use this widget to see how balanced the labels in your dataset are."
        )
        if label_df is not None:
            st.plotly_chart(fig_labels, use_container_width=True)
        else:
            st.markdown("No labels were found in the dataset")


def expander_text_lengths(
    tokenized_df,
    fig_tok_length,
    avg_length,
    std_length,
    text_field_name,
    length_field_name,
    column_id,
):
    _TEXT_LENGTH_CAPTION = (
        "Use this widget to identify outliers, particularly suspiciously long outliers."
    )
    with st.expander(f"Text Lengths{column_id}", expanded=False):
        st.caption(_TEXT_LENGTH_CAPTION)
        st.markdown(
            "Below, you can see how the lengths of the text instances in your dataset are distributed."
        )
        st.markdown(
            "Any unexpected peaks or valleys in the distribution may help to identify data instances you want to remove or augment."
        )
        st.markdown(
            "### Here is the relative frequency of different text lengths in your dataset:"
        )
        st.plotly_chart(fig_tok_length, use_container_width=True)
        data = tokenized_df[[length_field_name, text_field_name]].sort_values(
            by=["length"], ascending=True
        )
        st.markdown(
            "The average length of text instances is **"
            + str(avg_length)
            + " words**, with a standard deviation of **"
            + str(std_length)
            + "**."
        )

        start_id_show_lengths = st.slider(
            f"Show the shortest sentences{column_id} starting at:",
            0,
            len(data["length"].unique()),
            value=0,
            step=1,
        )
        st.dataframe(data[data["length"] == start_id_show_lengths].set_index("length"))


### Third, use a sentence embedding model
def expander_text_embeddings(
    text_dset, fig_tree, node_list, embeddings, text_field, column_id
):
    with st.expander(f"Text Embedding Clusters{column_id}", expanded=False):
        _EMBEDDINGS_CAPTION = """
        ### Hierarchical Clustering of Text Fields
        Taking in the diversity of text represented in a dataset can be
        challenging when it is made up of hundreds to thousands of sentences.
        Grouping these text items based on a measure of similarity can help
        users gain some insights into their distribution.
        The following figure shows a hierarchical clustering of the text fields
        in the dataset based on a
        [Sentence-Transformer](https://hf.co/sentence-transformers/all-mpnet-base-v2)
        model. Clusters are merged if any of the embeddings in cluster A has a
        dot product with any of the embeddings or with the centroid of cluster B
        higher than a threshold (one threshold per level, from 0.5 to 0.95).
        To explore the clusters, you can:
        - hover over a node to see the 5 most representative examples (deduplicated)
        - enter an example in the text box below to see which clusters it is most similar to
        - select a cluster by ID to show all of its examples
        """
        st.markdown(_EMBEDDINGS_CAPTION)
        st.plotly_chart(fig_tree, use_container_width=True)
        st.markdown("---\n")
        if st.checkbox(
            label="Enter text to see nearest clusters",
            key=f"search_clusters_{column_id}",
        ):
            compare_example = st.text_area(
                label="Enter some text here to see which of the clusters in the dataset it is closest to",
                key=f"search_cluster_input_{column_id}",
            )
            if compare_example != "":
                paths_to_leaves = embeddings.cached_clusters.get(
                    compare_example,
                    embeddings.find_cluster_beam(compare_example, beam_size=50),
                )
                clusters_intro = ""
                if paths_to_leaves[0][1] < 0.3:
                    clusters_intro += (
                        "**Warning: no close clusters found (best score <0.3). **"
                    )
                clusters_intro += "The closest clusters to the text entered aboce are:"
                st.markdown(clusters_intro)
                for path, score in paths_to_leaves[:5]:
                    example = text_dset[
                        node_list[path[-1]]["sorted_examples_centroid"][0][0]
                    ][text_field][:256]
                    st.write(
                        f"Cluster {path[-1]:5d} | Score: {score:.3f}  \n Example: {example}"
                    )
                show_node_default = paths_to_leaves[0][0][-1]
            else:
                show_node_default = len(node_list) // 2
        else:
            show_node_default = len(node_list) // 2
        st.markdown("---\n")
        show_node = st.selectbox(
            f"Choose a leaf node to explore in the{column_id} dataset:",
            range(len(node_list)),
            index=show_node_default,
        )
        node = node_list[show_node]
        start_id = st.slider(
            f"Show closest sentences in cluster to the centroid{column_id} starting at index:",
            0,
            len(node["sorted_examples_centroid"]) - 5,
            value=0,
            step=5,
        )
        for sid, sim in node["sorted_examples_centroid"][start_id : start_id + 5]:
            # only show the first 4 lines and the first 10000 characters
            show_text = text_dset[sid][text_field][:10000]
            show_text = "\n".join(show_text.split("\n")[:4])
            st.text(f"{sim:.3f} \t {show_text}")


### Then, show duplicates
def expander_text_duplicates(dstats, column_id):
    # TODO: Saving/loading figure
    with st.expander(f"Text Duplicates{column_id}", expanded=False):
        st.caption(
            "Use this widget to identify text strings that appear more than once."
        )
        st.markdown(
            "A model's training and testing may be negatively affected by unwarranted duplicates ([Lee et al., 2021](https://arxiv.org/abs/2107.06499))."
        )
        st.markdown("------")
        st.write(
            "### Here is the list of all the duplicated items and their counts in your dataset:"
        )
        # Eh...adding 1 because otherwise it looks too weird for duplicate counts when the value is just 1.
        if len(dstats.dup_counts_df) == 0:
            st.write("There are no duplicates in this dataset! 🥳")
        else:
            gb = GridOptionsBuilder.from_dataframe(dstats.dup_counts_df)
            gb.configure_column(
                f"text{column_id}",
                wrapText=True,
                resizable=True,
                autoHeight=True,
                min_column_width=85,
                use_container_width=True,
            )
            go = gb.build()
            AgGrid(dstats.dup_counts_df, gridOptions=go)


def expander_npmi_description(min_vocab):
    _NPMI_CAPTION = (
        "Use this widget to identify problematic biases and stereotypes in your data."
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
        "The more *positive* the score, the more associated the word is with the first identity term.  "
        "The more *negative* the score, the more associated the word is with the second identity term."
    )


### Finally, show Zipf stuff
def expander_zipf(z, zipf_fig, column_id):
    _ZIPF_CAPTION = """This shows how close the observed language is to an ideal
    natural language distribution following [Zipf's law](https://en.wikipedia.org/wiki/Zipf%27s_law),
    calculated by minimizing the [Kolmogorov-Smirnov (KS) statistic](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test)."""

    powerlaw_eq = r"""p(x) \propto x^{- \alpha}"""
    zipf_summary = (
        "The optimal alpha based on this dataset is: **"
        + str(round(z.alpha, 2))
        + "**, with a KS distance of: **"
        + str(round(z.distance, 2))
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
            "KS distance:": [str("%.2f" % z.distance)],
            "Min rank:": [str("%s" % int(z.xmin))],
        },
        columns=["Results"],
        orient="index",
    )
    fit_results_table.index.name = column_id
    with st.expander(
        f"Vocabulary Distribution{column_id}: Zipf's Law Fit", expanded=False
    ):
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


### Finally finally finally, show nPMI stuff.
def npmi_widget(column_id, available_terms, npmi_stats, min_vocab, use_cache=False):
    """
    Part of the main app, but uses a user interaction so pulled out as its own f'n.
    :param use_cache:
    :param column_id:
    :param npmi_stats:
    :param min_vocab:
    :return:
    """
    with st.expander(f"Word Association{column_id}: nPMI", expanded=False):
        if len(available_terms) > 0:
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
            # We calculate/grab nPMI data based on a canonical (alphabetic)
            # subgroup ordering.
            subgroup_pair = sorted([term1, term2])
            try:
                joint_npmi_df = npmi_stats.load_or_prepare_joint_npmi(subgroup_pair)
                npmi_show(joint_npmi_df)
            except KeyError:
                st.markdown(
                    "**WARNING!** The nPMI for these terms has not been pre-computed, please re-run caching."
                )
        else:
            st.markdown(
                "No words found co-occurring with both of the selected identity terms."
            )


def npmi_show(paired_results):
    if paired_results.empty:
        st.markdown("No words that co-occur enough times for results!  Or there's a 🐛.")
    else:
        s = pd.DataFrame(paired_results.sort_values(by="npmi-bias", ascending=True))
        # s.columns=pd.MultiIndex.from_arrays([['npmi','npmi','npmi','count', 'count'],['bias','man','straight','man','straight']])
        s.index.name = "word"
        npmi_cols = s.filter(like="npmi").columns
        count_cols = s.filter(like="count").columns
        # TODO: This is very different look than the duplicates table above. Should probably standardize.
        cm = sns.palplot(sns.diverging_palette(270, 36, s=99, l=48, n=16))
        out_df = (
            s.style.background_gradient(subset=npmi_cols, cmap=cm)
            .format(subset=npmi_cols, formatter="{:,.3f}")
            .format(subset=count_cols, formatter=int)
            .set_properties(
                subset=count_cols, **{"width": "10em", "text-align": "center"}
            )
            .set_properties(**{"align": "center"})
            .set_caption(
                "nPMI scores and co-occurence counts between the selected identity terms and the words they both co-occur with"
            )
        )  # s = pd.read_excel("output.xlsx", index_col="word")
        st.write("### Here is your dataset's nPMI results:")
        st.dataframe(out_df)


### Dumping unused functions here for now
### Second, show the distribution of text perplexities
def expander_text_perplexities(text_label_df, sorted_sents_loss, fig_loss):
    with st.expander("Show text perplexities A", expanded=False):
        st.markdown("### Text perplexities A")
        st.plotly_chart(fig_loss, use_container_width=True)
        start_id_show_loss = st.slider(
            "Show highest perplexity sentences in A starting at index:",
            0,
            text_label_df.shape[0] - 5,
            value=0,
            step=5,
        )
        for lss, sent in sorted_sents_loss[start_id_show_loss : start_id_show_loss + 5]:
            st.text(f"{lss:.3f} {sent}")
