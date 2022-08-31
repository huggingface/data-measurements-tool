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
import argparse
import logging
import streamlit as st
from os import mkdir
from os.path import isdir
from pathlib import Path

Path('./log_files').mkdir(exist_ok=True)

from data_measurements import dataset_statistics
import utils
from utils import dataset_utils
from utils import streamlit_utils as st_utils

logs = utils.prepare_logging(__file__)

st.set_page_config(
    page_title="Demo to showcase dataset metrics",
    page_icon="https://huggingface.co/front/assets/huggingface_logo.svg",
    layout="wide",
    initial_sidebar_state="auto",
)

# colorblind-friendly colors
colors = [
    "#332288",
    "#117733",
    "#882255",
    "#AA4499",
    "#CC6677",
    "#44AA99",
    "#DDCC77",
    "#88CCEE",
]

CACHE_DIR = dataset_utils.CACHE_DIR
# String names we are using (not coming from the stored dataset).
OUR_TEXT_FIELD = dataset_utils.OUR_TEXT_FIELD
TOKENIZED_FIELD = dataset_utils.TOKENIZED_FIELD
EMBEDDING_FIELD = dataset_utils.EMBEDDING_FIELD
LENGTH_FIELD = dataset_utils.LENGTH_FIELD
# TODO: Allow users to specify this.
_SHOW_TOP_N_WORDS = 10


@st.cache(
    hash_funcs={
        dataset_statistics.DatasetStatisticsCacheClass: lambda dstats: dstats.cache_path
    },
    allow_output_mutation=True,
)
def load_or_prepare(ds_args, show_embeddings, show_perplexities, use_cache=False):
    """
    Takes the dataset arguments from the GUI and uses them to load a dataset from the Hub or, if
    a cache for those arguments is available, to load it from the cache.
    Args:
        ds_args (dict): the dataset arguments defined via the streamlit app GUI
        show_embeddings (Bool): whether embeddings should we loaded and displayed for this dataset
        show_perplexities (Bool): whether perplexities should be loaded and displayed for this dataset
        use_cache (Bool) : whether the cache is used by default or not
    Returns:
        dstats: the computed dataset statistics (from the dataset_statistics class)
    """
    if not isdir(CACHE_DIR):
        logs.warning("Creating cache")
        # We need to preprocess everything.
        # This should eventually all go into a prepare_dataset CLI
        mkdir(CACHE_DIR)
    if use_cache:
        logs.warning("Using cache")
    dstats = dataset_statistics.DatasetStatisticsCacheClass(CACHE_DIR, **ds_args, use_cache=use_cache)
    if pull_cache_from_hub:
        logs.warning(dataset_utils.pull_cache_from_hub(dstats.cache_path, dstats.dataset_cache_dir))

    logs.warning("Loading dataset")
    dstats.load_or_prepare_dataset()
    logs.warning("Loading labels")
    dstats.load_or_prepare_labels()
    logs.warning("Loading text lengths")
    dstats.load_or_prepare_text_lengths()
    logs.warning("Loading duplicates")
    dstats.load_or_prepare_text_duplicates()
    logs.warning("Loading vocabulary")
    dstats.load_or_prepare_vocab()
    logs.warning("Loading general statistics...")
    dstats.load_or_prepare_general_stats()
    if show_embeddings:
        logs.warning("Loading Embeddings")
        dstats.load_or_prepare_embeddings()
    if show_perplexities:
        logs.warning("Loading Text Perplexities")
        dstats.load_or_prepare_text_perplexities()
    logs.warning("Loading nPMI")
    dstats.load_or_prepare_npmi()
    logs.warning("Loading Zipf")
    dstats.load_or_prepare_zipf()
    return dstats

@st.cache(
    hash_funcs={
        dataset_statistics.DatasetStatisticsCacheClass: lambda dstats: dstats.cache_path
    },
    allow_output_mutation=True,
)
def load_or_prepare_widgets(ds_args, show_embeddings, show_perplexities, live=True, pull_cache_from_hub=False, use_cache=False):
    """
    Loader specifically for the widgets used in the app.
    Args:
        ds_args:
        show_embeddings:
        show_perplexities:
        use_cache:

    Returns:

    """
    # When we're "live", cache is used.
    if live:
        use_cache = True
    if use_cache:
        logs.warning("Using cache")
    dstats = dataset_statistics.DatasetStatisticsCacheClass(CACHE_DIR, **ds_args, use_cache=use_cache)

    if pull_cache_from_hub:
        logs.warning(dataset_utils.pull_cache_from_hub(dstats.cache_path, dstats.dataset_cache_dir))

    if live:
        # checks whether the cache_dir exists in deployment mode
        if isdir(dstats.cache_path):
                try:
                    # Header widget
                    dstats.load_or_prepare_dset_peek(load_only=True)
                except:
                    logs.warning("Issue with %s." % "dataset peek")
                try:
                    dstats.load_or_prepare_vocab(load_only=True)
                except:
                    logs.warning("Issue with %s." % "vocabulary statistics")
                try:
                    # General stats widget
                    dstats.load_or_prepare_general_stats(load_only=True)
                except:
                    logs.warning("Issue with %s." % "general statistics")
                try:
                    # Labels widget
                    dstats.load_or_prepare_labels(load_only=True)
                except:
                    logs.warning("Issue with %s." % "label statistics")
                try:
                    # Text lengths widget
                    dstats.load_or_prepare_text_lengths(load_only=True)
                except:
                    logs.warning("Issue with %s." % "text length statistics")
                # TODO: If these are cached, can't we just show them by default?
                # It won't take up computation time.
                if show_embeddings:
                    try:
                        # Embeddings widget
                        dstats.load_or_prepare_embeddings(load_only=True)
                    except:
                        logs.warning("Issue with %s." % "embeddings")
                # TODO: If these are cached, can't we just show them by default?
                # It won't take up computation time.
                if show_perplexities:
                    try:
                        dstats.load_or_prepare_text_perplexities(load_only=True)
                    except:
                        logs.warning("Issue with %s." % "perplexities")
                try:
                    dstats.load_or_prepare_text_duplicates(load_only=True)
                except:
                    logs.warning("Issue with %s." % "text duplicates")
                try:
                    dstats.load_or_prepare_npmi(load_only=True)
                except:
                    logs.warning("Issue with %s." % "nPMI statistics")
                try:
                    dstats.load_or_prepare_zipf(load_only=True)
                except:
                    logs.warning("Issue with %s." % "Zipf statistics")
    # Calculates and creates cache_dir
    else:
        dstats.load_or_prepare_dset_peek()
        dstats.load_or_prepare_vocab()
        dstats.load_or_prepare_general_stats()
        dstats.load_or_prepare_labels()
        dstats.load_or_prepare_text_lengths()
        if show_embeddings:
            dstats.load_or_prepare_embeddings()
        if show_perplexities:
            dstats.load_or_prepare_text_perplexities()
        dstats.load_or_prepare_text_duplicates()
        dstats.load_or_prepare_npmi()
        dstats.load_or_prepare_zipf()
    return dstats


def show_column(dstats, ds_name_to_dict, show_embeddings, show_perplexities, column_id):
    """
    Function for displaying the elements in the right column of the streamlit app.
    Args:
        ds_name_to_dict (dict): the dataset name and options in dictionary form
        show_embeddings (Bool): whether embeddings should we loaded and displayed for this dataset
        show_perplexities (Bool): whether perplexities should be loaded and displayed for this dataset
        column_id (str): what column of the dataset the analysis is done on
    Returns:
        The function displays the information using the functions defined in the st_utils class.
    """
    # Note that at this point we assume we can use cache; default value is True.
    widget_dict = {1: ("general stats", st_utils.expander_general_stats), 2: (
    "label distribution", st_utils.expander_label_distribution),
                   3: ("text_lengths", st_utils.expander_text_lengths),
                   4: ("duplcates", st_utils.expander_text_duplicates),
                   5: ("npmi", st_utils.npmi_widget),
                   6: ("zipf", st_utils.expander_zipf)}

    # start showing stuff
    title_str = f"### Showing{column_id}: {dstats.dset_name} - {dstats.dset_config} - {dstats.split_name} - {'-'.join(dstats.text_field)}"
    st.markdown(title_str)
    logs.info("showing header")
    st_utils.expander_header(dstats, ds_name_to_dict, column_id)
    for widget_num, widget_call in sorted(widget_dict.items()):
        widget_type = widget_call[0]
        widget_fn = widget_call[1]
        logs.info("showing %s." % widget_type)
        try:
            widget_fn(dstats, column_id)
        except Exception as e:
            logs.info("Jk jk jk. There was an issue.")
            logs.warning("Issue with %s" % widget_type)
            logs.warning(e)
    # TODO: Can this be incorporated into the dictionary?
    if show_perplexities:
        st_utils.expander_text_perplexities(dstats, column_id)
    # TODO: Deprecate
    if show_embeddings:
        st_utils.expander_text_embeddings(
            dstats.text_dset,
            dstats.fig_tree,
            dstats.node_list,
            dstats.embeddings,
            OUR_TEXT_FIELD,
            column_id,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--live", default=False, required=False, action="store_true", help="Flag to specify that this is not running live.")
    parser.add_argument(
        "--pull_cache_from_hub", default=False, required=False, action="store_true", help="Flag to specify whether to look in the hub for measurements caches. If you are using this option, you must have HUB_CACHE_ORGANIZATION=<the organization you've set up on the hub to store your cache> and HF_TOKEN=<your hf token> on separate lines in a file named .env at the root of this repo.")
    arguments = parser.parse_args()
    live = arguments.live
    pull_cache_from_hub = arguments.pull_cache_from_hub
    # Sidebar description and selection
    ds_name_to_dict = dataset_utils.get_dataset_info_dicts()
    st.title("Data Measurements Tool")
    # Get the sidebar details
    st_utils.sidebar_header()
    # Set up naming, configs, and cache path.
    compare_mode = st.sidebar.checkbox("Comparison mode")

    # When using the app, try to use cache by default.
    use_cache = True
    # TODO: deprecate
    show_embeddings = st.sidebar.checkbox("Show text clusters")
    show_perplexities = st.sidebar.checkbox("Show text perplexities")
    # List of datasets for which embeddings are hard to compute:

    if compare_mode:
        logs.warning("Using Comparison Mode")
        dataset_args_left = st_utils.sidebar_selection(ds_name_to_dict, " A")
        dataset_args_right = st_utils.sidebar_selection(ds_name_to_dict, " B")
        left_col, _, right_col = st.columns([10, 1, 10])
        dstats_left = load_or_prepare_widgets(
            dataset_args_left, show_embeddings, show_perplexities, pull_cache_from_hub=pull_cache_from_hub, use_cache=use_cache
        )
        with left_col:
            if isdir(dstats_left.cache_path):
                show_column(dstats_left, ds_name_to_dict, show_embeddings, show_perplexities," A")
            else:
                st.markdown("### Missing pre-computed data measures!")
                st.write(dataset_args_left)
        dstats_right = load_or_prepare_widgets(
            dataset_args_right, show_embeddings, show_perplexities, pull_cache_from_hub=pull_cache_from_hub, use_cache=use_cache
        )
        with right_col:
            if isdir(dstats_right.cache_path):
                show_column(dstats_right, ds_name_to_dict, show_embeddings, show_perplexities, " B")
            else:
                st.markdown("### Missing pre-computed data measures!")
                st.write(dataset_args_right)
    else:
        logs.warning("Using Single Dataset Mode")
        dataset_args = st_utils.sidebar_selection(ds_name_to_dict, "")
        dstats = load_or_prepare_widgets(dataset_args, show_embeddings, show_perplexities, live=live, pull_cache_from_hub=pull_cache_from_hub, use_cache=use_cache)
        if isdir(dstats.cache_path):
            logs.warning(dstats.cache_path)
            show_column(dstats, ds_name_to_dict, show_embeddings, show_perplexities, "")
        else:
            st.markdown("### Missing pre-computed data measures!")
            st.write(dataset_args)


if __name__ == "__main__":
    main()