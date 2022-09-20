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

import streamlit.components.v1 as components

import streamlit as st
from data_measurements.dataset_statistics import DatasetStatisticsCacheClass as dmt_cls
import utils
from utils import dataset_utils
from utils import streamlit_utils as st_utils



#"""
#Examples:
## When not in deployment mode
#streamlit run app.py -- --live=False
#
## When deployed.
#streamlit run app.py
#"""

from data_measurements import dataset_statistics
from utils import dataset_utils
from utils import streamlit_utils as st_utils


logs = utils.prepare_logging(__file__)





st.set_page_config(
    page_title="Demo to showcase dataset metrics",
    page_icon="https://huggingface.co/front/assets/huggingface_logo.svg",
    layout="wide",
    initial_sidebar_state="auto",
)



 #with open('style.css') as f:
  #      st.markdown(f'<style>{f.read()}</style>',unsafe_allow_html=True)


# Utility for sidebar description and selection of the dataset
DATASET_NAME_TO_DICT = dataset_utils.get_dataset_info_dicts()






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



def load_or_prepare_widgets(ds_args, show_embeddings, show_perplexities, live=False, use_cache=False):
    """
    Loader specifically for the widgets used in the app.
    Args:
        ds_args:
        show_embeddings:
        show_perplexities:
        use_cache:
     """

def display_measurements(dataset_args, display_list, loaded_dstats,
                         show_perplexities):
    """Displays the measurement results in the UI"""
    if isdir(loaded_dstats.dataset_cache_dir):
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
        dataset_utils.pull_cache_from_hub(dstats.cache_path, dstats.dataset_cache_dir)

    # Data common across DMT:
    # Includes the dataset, the dataset tokenized, and the vocabulary
    dstats.load_or_prepare_dataset(load_only=load_only)
    dstats.load_or_prepare_dset_peek(load_only=load_only)
    # Tokenized dataset
    dstats.load_or_prepare_tokenized_df(load_only=load_only)
    # Vocabulary (uses tokenized dataset)
    dstats.load_or_prepare_vocab(load_only=load_only)
    # Custom widgets
    for widget_tuple in load_prepare_list:
        widget_name = widget_tuple[0]
        widget_fn = widget_tuple[1]
        try:
            widget_fn(load_only=load_only)
        except Exception as e:
            logs.warning("Issue with %s." % widget_name)
            logs.exception(e)
    # TODO: If these are cached, can't we just show them by default?
    # It won't take up computation time.
    if show_perplexities:
        try:
            dstats.load_or_prepare_text_perplexities(load_only=load_only)
        except Exception as e:
            logs.warning("Issue with %s." % "perplexities")
            logs.exception(e)
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


    #title_str = f"### Showing{column_id}: {dstats.dset_name} - {dstats.dset_config} - {dstats.split_name} - {'-'.join(dstats.text_field)}"
    #st.markdown(title_str)
    tab1, tab2, tab3, tab4, tab5, tab6, tab7,tab8 = st.tabs(["Dataset Description", "General Text Statistics", "Label Distribution", "Text Lengths","Text Duplicates","Text Perplexities","nPMI","Zipfs Law Fit"])#,"Text Embedding Clustering"])
    
    with tab1:
        logs.info("showing header")
        st_utils.expander_header(dstats, DATASET_NAME_TO_DICT,column_id)
    with tab2:
        logs.info("showing general stats")
        st_utils.expander_general_stats(dstats, column_id)
    with tab3:
        st_utils.expander_label_distribution(dstats.fig_labels, column_id)
    with tab4:
        st_utils.expander_text_lengths(dstats, column_id)
    with tab5:
        st_utils.expander_text_duplicates(dstats, column_id)
    with tab6:
        st_utils.expander_text_perplexities(dstats, column_id)
    # Uses an interaction; handled a bit differently than other widgets.
    with tab7:
        logs.info("showing npmi widget")
        #st_utils.npmi_widget(dstats.load_or_prepare_npmi, column_id)
        st_utils.npmi_widget
    with tab8:
        logs.info("showing zipf")
        #st_utils.expander_zipf(dstats.z, dstats.zipf_fig, column_id)
        st_utils.expander_zipf(dstats,column_id)
    #with tab9:
        #if show_embeddings:
        #    st_utils.expander_text_embeddings(
        #        dstats.text_dset,
        #        dstats.fig_tree,
        #        dstats.node_list,
        #        dstats.embeddings,
        #        OUR_TEXT_FIELD,
        #        column_id,
        #    )
    show_embeddings = False
    #title_str = f"### Showing{column_id}: {dstats.dset_name} - {dstats.dset_config} - {dstats.split_name} - {'-'.join(dstats.text_field)}"
    #st.markdown(title_str)
    #logs.info("showing header")
    #st_utils.expander_header(dstats, ds_name_to_dict, column_id)
    logs.info("showing general stats")
    #st_utils.expander_general_stats(dstats, column_id)
    #st_utils.expander_label_distribution(dstats.fig_labels, column_id)
    #st_utils.expander_text_lengths(dstats, column_id)
    #st_utils.expander_text_duplicates(dstats, column_id)
    # Uses an interaction; handled a bit differently than other widgets.
    logs.info("showing npmi widget")
    #st_utils.npmi_widget(dstats.npmi_stats, _MIN_VOCAB_COUNT, column_id)
    logs.info("showing zipf")
   # st_utils.expander_zipf(dstats.z, dstats.zipf_fig, column_id)


   #if show_embeddings:
        #st_utils.expander_text_embeddings(
       #     dstats.text_dset,
        #    dstats.fig_tree,
        #    dstats.node_list,
        #    dstats.embeddings,
        #    OUR_TEXT_FIELD,
       #     column_id,
        #)
        
    ##TO:DO @Ezi --> incorperate 
    #st_utils.expander_header(dstats, DATASET_NAME_TO_DICT)
    for widget_tuple in display_list:
        widget_type = widget_tuple[0]
        widget_fn = widget_tuple[1]
        logs.info("showing %s." % widget_type)
        try:
            widget_fn(dstats, column_id)
        except Exception as e:
            logs.warning("Jk jk jk. There was an issue with %s:" % widget_type)
            logs.exception(e)
    # TODO: Fix how this is a weird outlier.

   #if show_perplexities:
    #    st_utils.expander_text_perplexities(dstats, column_id)
    #logs.info("Have finished displaying the widgets.")


def main():
    with open("style.css") as f:                                                # Reading style.css file and opening it 
        st.markdown(f'<style>{f.read()}</style>',unsafe_allow_html= True)       # Storing all styles in the streamlit markdown  &   unsafe_allow_html to true so that we can use html tags with the code  
    #----read and run html file

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--live", default=False, required=False, action="store_true", help="Flag to specify that this is not running live.")
    parser.add_argument(
        "--pull_cache_from_hub", default=False, required=False, action="store_true", help="Flag to specify whether to look in the hub for measurements caches. If you are using this option, you must have HUB_CACHE_ORGANIZATION=<the organization you've set up on the hub to store your cache> and HF_TOKEN=<your hf token> on separate lines in a file named .env at the root of this repo.")
    arguments = parser.parse_args()
    live = arguments.live



    # Sidebar description and selection

    ds_name_to_dict = dataset_utils.get_dataset_info_dicts()
    st.title("Data Measurements Tool")
    st.markdown("""
    This demo showcases the [dataset metrics as we develop them](https://huggingface.co/blog/data-measurements-tool).
    
    With this tool, you can:
    
    ✓ Dynamic loading of datasets in the lib

    ✓ Fetching config and info without downloading the dataset

    ✓ Propose the list of candidate text and label features to select
    We are still working on:

    ✓ Implementing all the current tools
    """,
        unsafe_allow_html=True,)
    # Get the sidebar details
    st_utils.sidebar_header()
    # Set up naming, configs, and cache path.
    #compare_mode = st.sidebar.checkbox("Comparison mode")

    # When not doing new development, use the cache.
    use_cache = True

    #show_embeddings = 0 #st.sidebar.checkbox("Show text clusters")
    # List of datasets for which embeddings are hard to compute:

    
    #show_embeddings =0 #st.sidebar.checkbox("Show text clusters")
    #show_perplexities =0# st.sidebar.checkbox("Show text perplexities")
  
    logs.warning("Using Single Dataset Mode")
    #dataset_args = st_utils.sidebar_selection(ds_name_to_dict, "")
    #dstats, cache_exists = load_or_prepare_widgets(dataset_args, show_embeddings,show_perplexities, live=live, use_cache=use_cache)
    #if cache_exists:
       # show_column(dstats, ds_name_to_dict, show_embeddings, show_perplexities, "")
   # else:

    #    st.markdown("### Missing pre-computed data measures!")
     #   st.write(dataset_args)
    
    st.sidebar.write('\n')
    st.sidebar.write('\n')
    st.sidebar.write('\n')
    #st.sidebar.write('\n')

    ### this is in streamlit_utils.py
    #st.sidebar.markdown(
   #     """
    
   # The tool is in development, and will keep growing in utility and functionality 🤗🚀
   # """,
   #     unsafe_allow_html=True,
   # )

   

    pull_cache_from_hub = arguments.pull_cache_from_hub

    dataset_args = st_utils.sidebar_selection(DATASET_NAME_TO_DICT)

    # Initialize the interface and grab the UI-provided arguments
    #dataset_args = display_initial_UI()

    # TODO: Make this less of a weird outlier.
    #show_perplexities = st.sidebar.checkbox("Show text perplexities")
    show_perplexities = False


    # Initialize the main DMT class with the UI-provided arguments
    # When using the app (this file), try to use cache by default.
    dstats = dmt_cls(**dataset_args, use_cache=True)
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
