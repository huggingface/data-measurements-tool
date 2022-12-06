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
import ast
import gradio as gr
from os.path import isdir
from data_measurements.dataset_statistics import DatasetStatisticsCacheClass as dmt_cls
import utils
from utils import dataset_utils
from utils import gradio_utils as gr_utils
import widgets

logs = utils.prepare_logging(__file__)

# Utility for sidebar description and selection of the dataset
DATASET_NAME_TO_DICT = dataset_utils.get_dataset_info_dicts()


def get_load_prepare_list(dstats):
    """
    # Get load_or_prepare functions for the measurements we will display
    """
    # Measurement calculation:
    # Add any additional modules and their load-prepare function here.
    load_prepare_list = [("general stats", dstats.load_or_prepare_general_stats),
                         ("label distribution", dstats.load_or_prepare_labels),
                         ("text_lengths", dstats.load_or_prepare_text_lengths),
                         ("duplicates", dstats.load_or_prepare_text_duplicates),
                         ("npmi", dstats.load_or_prepare_npmi),
                         ("zipf", dstats.load_or_prepare_zipf)]

    return load_prepare_list


def get_ui_widgets():
    """Get the widgets that will be displayed in the UI."""
    return [widgets.DatasetDescription(DATASET_NAME_TO_DICT),
            widgets.GeneralStats(),
            widgets.LabelDistribution(),
            widgets.TextLengths(),
            widgets.Npmi(),
            widgets.Zipf()]


def get_widgets():
    """
    # A measurement widget requires 2 things:
    # - A load or prepare function
    # - A display function
    # We define these in two separate functions get_load_prepare_list and get_ui_widgets;
    # any widget can be added by modifying both functions and the rest of the app logic will work.
    # get_load_prepare_list is a function since it requires a DatasetStatisticsCacheClass which will
    # not be created until dataset and config values are selected in the ui
    """
    return get_load_prepare_list, get_ui_widgets()


def get_title(dstats):
    title_str = f"### Showing: {dstats.dset_name} - {dstats.dset_config} - {dstats.split_name} - {'-'.join(dstats.text_field)}"
    logs.info("showing header")
    return title_str


def display_initial_UI():
    """Displays the header in the UI"""
    # Extract the selected arguments
    dataset_args = gr_utils.sidebar_selection(DATASET_NAME_TO_DICT)
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
    # Includes the dataset text/requested feature column,
    # the dataset tokenized, and the vocabulary
    dstats.load_or_prepare_text_dataset(load_only=load_only)
    # Just a snippet of the dataset
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
    gr_utils.expander_header(dstats, DATASET_NAME_TO_DICT)
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
    if show_perplexities:
        gr_utils.expander_text_perplexities(dstats, column_id)
    logs.info("Have finished displaying the widgets.")


def create_demo(live: bool, pull_cache_from_hub: bool):
    with gr.Blocks() as demo:
        state = gr.State()
        with gr.Row():
            with gr.Column(scale=1):
                dataset_args = display_initial_UI()
                get_load_prepare_list_fn, widget_list = get_widgets()
                # # TODO: Make this less of a weird outlier.
                # Doesn't do anything right now
                show_perplexities = gr.Checkbox(label="Show text perplexities")
            with gr.Column(scale=4):
                gr.Markdown("# Data Measurements Tool")
                title = gr.Markdown()
                for widget in widget_list:
                    widget.render()

            def update_ui(dataset: str, config: str, split: str, feature: str):
                feature = ast.literal_eval(feature)
                label_field, label_names = gr_utils.get_label_names(dataset, config, DATASET_NAME_TO_DICT)
                dstats = dmt_cls(dset_name=dataset, dset_config=config, split_name=split, text_field=feature,
                                 label_field=label_field, label_names=label_names, use_cache=True)
                load_prepare_list = get_load_prepare_list_fn(dstats)
                dstats = load_or_prepare_widgets(dstats, load_prepare_list, show_perplexities=False,
                                                 live=live, pull_cache_from_hub=pull_cache_from_hub)
                output = {title: get_title(dstats), state: dstats}
                for widget in widget_list:
                    output.update(widget.update(dstats))
                return output

            def update_dataset(dataset: str):
                new_values = gr_utils.update_dataset(dataset, DATASET_NAME_TO_DICT)
                config = new_values[0][1]
                feature = new_values[1][1]
                split = new_values[2][1]
                new_dropdown = {
                    dataset_args["dset_config"]: gr.Dropdown.update(choices=new_values[0][0], value=config),
                    dataset_args["text_field"]: gr.Dropdown.update(choices=new_values[1][0], value=feature),
                    dataset_args["split_name"]: gr.Dropdown.update(choices=new_values[2][0], value=split),
                }
                return new_dropdown

            def update_config(dataset: str, config: str):
                new_values = gr_utils.update_config(dataset, config, DATASET_NAME_TO_DICT)

                feature = new_values[0][1]
                split = new_values[1][1]
                new_dropdown = {
                    dataset_args["text_field"]: gr.Dropdown.update(choices=new_values[0][0], value=feature),
                    dataset_args["split_name"]: gr.Dropdown.update(choices=new_values[1][0], value=split)
                }
                return new_dropdown

            measurements = [comp for output in widget_list for comp in output.output_components]
            demo.load(update_ui,
                      inputs=[dataset_args["dset_name"], dataset_args["dset_config"], dataset_args["split_name"], dataset_args["text_field"]],
                      outputs=[title, state] + measurements)

            for widget in widget_list:
                widget.add_events(state)

            dataset_args["dset_name"].change(update_dataset,
                                             inputs=[dataset_args["dset_name"]],
                                             outputs=[dataset_args["dset_config"],
                                              dataset_args["split_name"], dataset_args["text_field"],
                                             title, state] + measurements)

            dataset_args["dset_config"].change(update_config,
                                               inputs=[dataset_args["dset_name"], dataset_args["dset_config"]],
                                               outputs=[dataset_args["split_name"], dataset_args["text_field"],
                                                        title, state] + measurements)

            dataset_args["calculate_btn"].click(update_ui,
                                                inputs=[dataset_args["dset_name"], dataset_args["dset_config"],
                                                        dataset_args["split_name"], dataset_args["text_field"]],
                                                outputs=[title, state] + measurements)
    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--live", default=False, required=False, action="store_true", help="Flag to specify that this is not running live.")
    parser.add_argument(
        "--pull_cache_from_hub", default=False, required=False, action="store_true", help="Flag to specify whether to look in the hub for measurements caches. If you are using this option, you must have HUB_CACHE_ORGANIZATION=<the organization you've set up on the hub to store your cache> and HF_TOKEN=<your hf token> on separate lines in a file named .env at the root of this repo.")
    arguments = parser.parse_args()
    live = arguments.live
    pull_cache_from_hub = arguments.pull_cache_from_hub

    # Create and initialize the demo
    demo = create_demo(live, pull_cache_from_hub)

    demo.launch()

if __name__ == "__main__":
    main()
