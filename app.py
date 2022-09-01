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
from data_measurements.dataset_statistics import DatasetStatisticsCacheClass as dmt_cls
import utils
from utils import dataset_utils
from utils import streamlit_utils as st_utils

logs = utils.prepare_logging(__file__)

# Utility for sidebar description and selection of the dataset
DATASET_NAME_TO_DICT = dataset_utils.get_dataset_info_dicts()


# Set up the basic interface of the app.
st.set_page_config(
    page_title="Demo to showcase dataset metrics",
    page_icon="https://huggingface.co/front/assets/huggingface_logo.svg",
    layout="wide",
    initial_sidebar_state="auto",
)

@st.cache(hash_funcs={dmt_cls: lambda dstats: dstats.cache_path}, allow_output_mutation=True,)

def get_widgets(dstats):
    """
    # A measurement widget requires 2 things:
    # - A load or prepare function
    # - A display function
    # We define these here; any widget can be trivially added in this way
    # and the rest of the app logic will work.
    """
    # Measurement calculation:
    # Add any additional modules and their load-prepare function here.
    load_prepare_list = [("general stats", dstats.load_or_prepare_general_stats),
                         ("label distribution", dstats.load_or_prepare_labels),
                         ("text_lengths", dstats.load_or_prepare_text_lengths),
                         ("duplicates", dstats.load_or_prepare_text_duplicates),
                         ("npmi", dstats.load_or_prepare_npmi),
                         ("zipf", dstats.load_or_prepare_zipf)]
    # Measurement interface:
    # Add the graphic interfaces for any new measurements here.
    display_list = [("general stats", st_utils.expander_general_stats),
                    ("label distribution", st_utils.expander_label_distribution),
                    ("text_lengths", st_utils.expander_text_lengths),
                    ("duplicates", st_utils.expander_text_duplicates),
                    ("npmi", st_utils.npmi_widget),
                    ("zipf", st_utils.expander_zipf)]

    return load_prepare_list, display_list

def display_title(dstats):
    title_str = f"### Showing: {dstats.dset_name} - {dstats.dset_config} - {dstats.split_name} - {'-'.join(dstats.text_field)}"
    st.markdown(title_str)
    logs.info("showing header")

def display_measurements(dataset_args, display_list, loaded_dstats,
                         show_perplexities):
    """Displays the measurement results in the UI"""
    if isdir(loaded_dstats.cache_path):
        show_column(loaded_dstats, display_list, show_perplexities)
    else:
        st.markdown("### Missing pre-computed data measures!")
        st.write(dataset_args)

def display_initial_UI():
    """Displays the header in the UI"""
    st.title("Data Measurements Tool")
    # Write out the sidebar details
    st_utils.sidebar_header()
    # Extract the selected arguments
    dataset_args = st_utils.sidebar_selection(DATASET_NAME_TO_DICT)
    return dataset_args

def load_or_prepare_widgets(dstats, load_prepare_list, show_perplexities, live=True, pull_cache_from_hub=False):
    """
     Takes the dataset arguments from the GUI and uses them to load a dataset from the Hub or, if
     a cache for those arguments is available, to load it from the cache.
     Widget data is loaded only when the system is live (deployed for users).
     Otherwise, the data is prepared if it doesn't yet exist.
     Args:
         ds_args (dict): the dataset arguments defined via the streamlit app GUI
         load_prepare_list (list): List of (widget_name, widget_load_or_prepare_function)
         show_perplexities (Bool): whether perplexities should be loaded and displayed for this dataset
         live (Bool): Whether the system is deployed for live use by users.
         pull_cache_from_hub (Bool): Whether the cache should be pulled from the hub (vs locally)
     Returns:
         dstats: the computed dataset statistics (from the dataset_statistics class)
     """

    # When we're "live" (tool is being used by users on our servers),
    # cache is used and the f'ns are instructed to only try to load cache,
    # not to prepare/compute anything anew.
    if live:
        # Only use what's cached; don't prepare anything
        load_only = True
        logs.info("Only using cache.")
    else:
        # Prepare things anew and cache them if we're not live.
        load_only = False
        logs.info("Making new calculations if cache is not there.")
    if pull_cache_from_hub:
        logs.info("Pulling cache from hub:")
        # TODO: This doesn't seem to be being used ?
        logs.info(dataset_utils.pull_cache_from_hub(dstats.cache_path, dstats.dataset_cache_dir))

    # Data common across DMT: The first snippet of the dataset,
    # and the vocabulary
    dstats.load_or_prepare_dset_peek()
    dstats.load_or_prepare_vocab()
    # Custom widgets
    for widget_tuple in load_prepare_list:
        widget_name = widget_tuple[0]
        widget_fn = widget_tuple[1]
        try:
            widget_fn(load_only=load_only)
        except Exception as e:
            logs.warning("Issue with %s." % widget_name)
            logs.warning(e)
    # TODO: If these are cached, can't we just show them by default?
    # It won't take up computation time.
    if show_perplexities:
        try:
            dstats.load_or_prepare_text_perplexities(load_only=load_only)
        except Exception as e:
            logs.warning("Issue with %s." % "perplexities")
            logs.warning(e)
    return dstats


def show_column(dstats, display_list, show_perplexities, column_id=""):
    """
    Function for displaying the elements in the streamlit app.
    Args:
        dstats (class): The dataset_statistics.py DatasetStatisticsCacheClass
        display_list (list): List of tuples for (widget_name, widget_display_function)
        show_perplexities (Bool): Whether perplexities should be loaded and displayed for this dataset
        column_id (str): Which column of the dataset the analysis is done on [DEPRECATED for v1]
    """

    # start showing stuff
    st_utils.expander_header(dstats, DATASET_NAME_TO_DICT)
    for widget_tuple in display_list:
        widget_type = widget_tuple[0]
        widget_fn = widget_tuple[1]
        logs.info("showing %s." % widget_type)
        try:
            widget_fn(dstats, column_id)
        except Exception as e:
            logs.warning("Jk jk jk. There was an issue with %s:" % widget_type)
            logs.warning(e)
    # TODO: Fix how this is a weird outlier.
    if show_perplexities:
        st_utils.expander_text_perplexities(dstats, column_id)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--live", default=False, required=False, action="store_true", help="Flag to specify that this is not running live.")
    parser.add_argument(
        "--pull_cache_from_hub", default=False, required=False, action="store_true", help="Flag to specify whether to look in the hub for measurements caches. If you are using this option, you must have HUB_CACHE_ORGANIZATION=<the organization you've set up on the hub to store your cache> and HF_TOKEN=<your hf token> on separate lines in a file named .env at the root of this repo.")
    arguments = parser.parse_args()
    live = arguments.live
    pull_cache_from_hub = arguments.pull_cache_from_hub

    # Initialize the interface and grab the UI-provided arguments
    dataset_args = display_initial_UI()

    # TODO: Make this less of a weird outlier.
    show_perplexities = st.sidebar.checkbox("Show text perplexities")

    # Initialize the main DMT class with the UI-provided arguments
    # When using the app (this file), try to use cache by default.
    dstats = dmt_cls(dataset_utils.CACHE_DIR, **dataset_args, use_cache=True)
    display_title(dstats)
    # Get the widget functionality for the different measurements.
    load_prepare_list, display_list = get_widgets(dstats)
    # Load/Prepare the DMT widgets.
    loaded_dstats = load_or_prepare_widgets(dstats, load_prepare_list,
                                            show_perplexities, live=live,
                                            pull_cache_from_hub=pull_cache_from_hub)
    # After the load_or_prepare functions are run,
    # we should have a cache for each measurement widget --
    # either because it was there already,
    # or we computed them (which we do when not live).
    # Write out on the UI what we have.
    display_measurements(dataset_args, display_list, loaded_dstats,
                         show_perplexities)

if __name__ == "__main__":
    main()
