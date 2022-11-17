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
import logging
import seaborn as sns
import gradio as gr
import pandas as pd
from os import mkdir
from os.path import isdir
from pathlib import Path
from data_measurements.dataset_statistics import DatasetStatisticsCacheClass as dmt_cls
import utils
from utils import dataset_utils
from utils import gradio_utils as gr_utils

logs = utils.prepare_logging(__file__)

# Utility for sidebar description and selection of the dataset
DATASET_NAME_TO_DICT = dataset_utils.get_dataset_info_dicts()


# Set up the basic interface of the app.
# st.set_page_config(
#     page_title="Demo to showcase dataset metrics",
#     page_icon="https://huggingface.co/front/assets/huggingface_logo.svg",
#     layout="wide",
#     initial_sidebar_state="auto",
# )
from abc import ABC, abstractmethod


class Widget(ABC):

    @abstractmethod
    def ui(self):
        pass

    @abstractmethod
    def update(self, dstats: dmt_cls):
        pass

    @property
    @abstractmethod
    def output_components(self):
        pass


class DatasetDescription(Widget):

    def __init__(self):
        self.description_markdown = gr.Markdown(render=False)
        self.description_df = gr.DataFrame(render=False, wrap=True)

    def ui(self):
        with gr.Accordion("Dataset Description", open=False):
            self.description_markdown.render()
            self.description_df.render()

    def update(self, dstats: dmt_cls):
        return {self.description_markdown: DATASET_NAME_TO_DICT[dstats.dset_name][dstats.dset_config][gr_utils.HF_DESC_FIELD],
                self.description_df: pd.DataFrame(dstats.dset_peek)}

    @property
    def output_components(self):
        return [self.description_markdown, self.description_df]


class LabelDistribution(Widget):

    def __init__(self):
        self.label_dist_plot = gr.Plot(render=False, visible=False)
        self.label_dist_no_label_text = gr.Markdown(value="No labels were found in the dataset",
                                                    render=False, visible=False)
        self.label_dist_accordion = gr.Accordion(render=False, label="", open=False)

    def ui(self):
        with gr.Accordion(label="Label Distribution", open=False):
            gr.Markdown("Use this widget to see how balanced the labels in your dataset are.")
            self.label_dist_plot.render()
            self.label_dist_no_label_text.render()

    def update(self, dstats: dmt_cls):
        if dstats.fig_labels:
            output = {#self.label_dist_accordion: gr.Accordion.update(label=f"Label Distribution{dstats.text_field}"),
                      self.label_dist_plot: gr.Plot.update(value=dstats.fig_labels, visible=True),
                      self.label_dist_no_label_text: gr.Markdown.update(visible=False)}
        else:
            output = {#self.label_dist_accordion: gr.Accordion.update(label=f"Label Distribution{dstats.text_field}"),
                      self.label_dist_plot: gr.Plot.update(visible=False),
                      self.label_dist_no_label_text: gr.Markdown.update(visible=True)}
        return output

    @property
    def output_components(self):
        return [self.label_dist_plot, self.label_dist_no_label_text]


class GeneralStats(Widget):

    def __init__(self):
        self.general_stats = gr.Markdown(render=False)
        self.general_stats_top_vocab = gr.DataFrame(render=False)
        self.general_stats_missing = gr.Markdown(render=False)
        self.general_stats_duplicates = gr.Markdown(render=False)

    def ui(self):
        with gr.Accordion(f"General Text Statistics", open=False):
            self.general_stats.render()
            self.general_stats_top_vocab.render()
            self.general_stats_missing.render()
            self.general_stats_duplicates.render()

    def update(self, dstats: dmt_cls):
        general_stats_text = f"""
        Use this widget to check whether the terms you see most represented in the dataset make sense for the goals of the dataset.
        
        There are {str(dstats.total_words)} total words.
        
        There are {dstats.total_open_words} after removing closed class words.
        
        The most common [open class words](https://dictionary.apa.org/open-class-words) and their counts are: 
        """
        top_vocab = pd.DataFrame(dstats.sorted_top_vocab_df)
        missing_text = f"There are {dstats.text_nan_count} missing values in the dataset"
        if dstats.dups_frac > 0:
            dupes_text = f"The dataset is {round(dstats.dups_frac * 100, 2)}% duplicates, For more information about the duplicates, click the 'Duplicates' tab below."
        else:
            dupes_text = "There are 0 duplicate items in the dataset"
        return {self.general_stats: general_stats_text, self.general_stats_top_vocab: top_vocab,
                self.general_stats_missing: missing_text, self.general_stats_duplicates: dupes_text}

    @property
    def output_components(self):
        return [self.general_stats, self.general_stats_top_vocab,
                self.general_stats_missing, self.general_stats_duplicates]


class TextLengths(Widget):

    def __init__(self):
        self.text_length_distribution_plot = gr.Image(render=False)
        self.text_length_explainer = gr.Markdown(render=False)
        self.text_length_drop_down = gr.Dropdown(render=False)
        self.text_length_df = gr.DataFrame(render=False)

    def update_text_length_df(self, length, dstats):
        return dstats.length_obj.lengths_df[
                        dstats.length_obj.lengths_df["length"] == length
                        ].set_index("length")

    def ui(self):
        with gr.Accordion("Text Lengths", open=False):
            gr.Markdown("Use this widget to identify outliers, particularly suspiciously long outliers.")
            gr.Markdown(
                "Below, you can see how the lengths of the text instances in your "
                "dataset are distributed."
            )
            gr.Markdown(
                "Any unexpected peaks or valleys in the distribution may help to "
                "identify instances you want to remove or augment."
            )
            gr.Markdown(
                "### Here is the count of different text lengths in "
                "your dataset:"
            )
            # When matplotlib first creates this, it's a Figure.
            # Once it's saved, then read back in,
            # it's an ndarray that must be displayed using st.image
            # (I know, lame).
            self.text_length_distribution_plot.render()
            self.text_length_explainer.render()
            self.text_length_drop_down.render()
            self.text_length_df.render()

    def update(self, dstats: dmt_cls):
        explainer_text = (
                "The average length of text instances is **"
                + str(round(dstats.length_obj.avg_length, 2))
                + " words**, with a standard deviation of **"
                + str(round(dstats.length_obj.std_length, 2))
                + "**."
            )
        output = {self.text_length_distribution_plot: dstats.length_obj.fig_lengths,
                  self.text_length_explainer: explainer_text,
                  }
        if dstats.length_obj.lengths_df is not None:
            import numpy as np
            choices = np.sort(dstats.length_obj.lengths_df["length"].unique())[::-1].tolist()
            output[self.text_length_drop_down] = gr.Dropdown.update(choices=choices,
                                                                    value=choices[0])
            output[self.text_length_df] = self.update_text_length_df(choices[0], dstats)
        else:
            output[self.text_length_df] = gr.update(visible=False)
            output[self.text_length_drop_down] = gr.update(visible=False)
        return output

    @property
    def output_components(self):
        return [self.text_length_distribution_plot,
                self.text_length_explainer,
                self.text_length_drop_down, self.text_length_df]


class Zipf(Widget):

    def __init__(self):
        self.zipf_table = gr.DataFrame(render=False)
        self.alpha_warning = gr.Markdown(value="Your alpha value is a bit on the high side, which means that the distribution over words in this dataset is a bit unnatural. This could be due to non-language items throughout the dataset.",
                                         render=False, visible=False)
        self.xmin_warning = gr.Markdown(value="The minimum rank for this fit is a bit on the high side, which means that the frequencies of your most common words aren't distributed as would be expected by Zipf's law.",
                                        render=False, visible=False)
        self.zipf_summary = gr.Markdown(render=False)
        self.zipf_plot = gr.Plot(render=False)

    def ui(self):
        with gr.Accordion("Vocabulary Distribution: Zipf's Law Fit", open=False):
            gr.Markdown(
                "Use this widget for the counts of different words in your dataset, measuring the difference between the observed count and the expected count under Zipf's law."
            )
            gr.Markdown("""This shows how close the observed language is to an ideal
                        natural language distribution following [Zipf's law](https://en.wikipedia.org/wiki/Zipf%27s_law),
                        calculated by minimizing the [Kolmogorov-Smirnov (KS) statistic](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test).""")
            gr.Markdown(
            """
            A Zipfian distribution follows the power law: $p(x) \propto x^{-α}$ with an ideal α value of 1.
            
            In general, an alpha greater than 2 or a minimum rank greater than 10 (take with a grain of salt) means that your distribution is relativaly _unnatural_ for natural language. This can be a sign of mixed artefacts in the dataset, such as HTML markup.
            
            Below, you can see the counts of each word in your dataset vs. the expected number of counts following a Zipfian distribution.
            
            -----
            
            ### Here is your dataset's Zipf results:
            """
            )
            self.zipf_table.render()
            self.zipf_summary.render()
            self.zipf_plot.render()
            self.alpha_warning.render()
            self.xmin_warning.render()

    def update(self, dstats: dmt_cls):
        z = dstats.z
        zipf_fig = dstats.zipf_fig

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

        fit_results_table = pd.DataFrame.from_dict(
            {
                r"Alpha:": [str("%.2f" % z.alpha)],
                "KS distance:": [str("%.2f" % z.ks_distance)],
                "Min rank:": [str("%s" % int(z.xmin))],
            },
            columns=["Results"],
            orient="index",
        )
        fit_results_table.index.name = ""

        output = {self.zipf_table: fit_results_table,
                  self.zipf_summary: zipf_summary,
                  self.zipf_plot: zipf_fig,
                  self.alpha_warning: gr.Markdown.update(visible=False),
                  self.xmin_warning: gr.Markdown.update(visible=False)}
        if z.alpha > 2:
            output[self.alpha_warning] = gr.Markdown.update(visible=True)
        if z.xmin > 5:
           output[self.xmin_warning] = gr.Markdown.update(visible=True)
        return output

    @property
    def output_components(self):
        return [self.zipf_table, self.zipf_plot, self.zipf_summary, self.alpha_warning, self.xmin_warning]


class Npmi(Widget):

    def __init__(self):
        self.npmi_first_word = gr.Dropdown(render=False,
                                           label="What is the first word you want to select?")
        self.npmi_second_word = gr.Dropdown(render=False,
                                            label="What is the second word you want to select?")
        self.npmi_error_text = gr.Markdown(render=False)
        self.npmi_df = gr.HTML(render=False)
        self.npmi_empty_text = gr.Markdown(render=False)
        self.npmi_description = gr.Markdown(render=False)

    @property
    def output_components(self):
        return [self.npmi_first_word, self.npmi_second_word,
                self.npmi_error_text, self.npmi_df, self.npmi_description, self.npmi_empty_text]

    def ui(self):
        with gr.Accordion("Word Association: nPMI", open=False):
            self.npmi_description.render()
            self.npmi_first_word.render()
            self.npmi_second_word.render()
            self.npmi_df.render()
            self.npmi_empty_text.render()
            self.npmi_error_text.render()

    def update(self, dstats: dmt_cls):
        min_vocab = dstats.min_vocab_count
        npmi_stats = dstats.npmi_obj
        available_terms = npmi_stats.avail_identity_terms
        output = {comp: gr.update(visible=False) for comp in self.output_components}
        if npmi_stats and len(available_terms) > 0:
            output[self.npmi_description] = gr.Markdown.update(value=self.expander_npmi_description(min_vocab), visible=True)
            output[self.npmi_first_word] = gr.Dropdown.update(choices=available_terms, value=available_terms[0], visible=True)
            output[self.npmi_second_word] = gr.Dropdown.update(choices=available_terms[::-1], value=available_terms[-1], visible=True)
            output.update(self.npmi_show(available_terms[0], available_terms[-1], dstats))
        else:
            output[self.npmi_error_text] = gr.Markdown.update(visible=True,
                                                              value="No words found co-occurring with both of the selected identity terms.")
        return output

    def npmi_show(self, term1, term2, dstats):
        npmi_stats = dstats.npmi_obj
        paired_results = npmi_stats.get_display(term1, term2)
        output = {}
        if paired_results.empty:
            output[self.npmi_empty_text] = gr.Markdown.update(
                value="""No words that co-occur enough times for results! Or there's a 🐛. 
                        Or we're still computing this one. 🤷""",
                visible=True)
            output[self.npmi_df] = gr.HTML.update(visible=False)
        else:
            output[self.npmi_empty_text] = gr.Markdown.update(visible=False)
            logs.debug("Results to be shown in streamlit are")
            logs.debug(paired_results)
            s = pd.DataFrame(
                paired_results.sort_values(paired_results.columns[0], ascending=True))
            s.index.name = "word"
            s = s.reset_index()
            bias_col = [col for col in s.columns if col != "word"]
            # count_cols = s.filter(like="count").columns
            # Keep the dataframe from being crazy big.
            if s.shape[0] > 10000:
                bias_thres = max(abs(s[s[0]][5000]),
                                 abs(s[s[0]][-5000]))
                logs.info(f"filtering with bias threshold: {bias_thres}")
                s_filtered = s[s[0].abs() > bias_thres]
            else:
                s_filtered = s
            #cm = sns.palplot(sns.diverging_palette(270, 36, s=99, l=48, n=16))
            out_df = s_filtered.style.background_gradient(subset=bias_col).format(
                formatter="{:,.3f}", subset=bias_col)\
                .set_properties(**{"align": "center", "width": "100em"})\
                .set_caption(
                "nPMI scores between the selected identity terms and the words they both co-occur with")
            # set_properties(subset=count_cols, **{"width": "10em", "text-align": "center"}).
            # .format(subset=count_cols, formatter=int).
            # .format(subset=bias_col, formatter="{:,.3f}")
            output[self.npmi_df] = out_df.to_html()
        return output


    @staticmethod
    def expander_npmi_description(min_vocab):
        return f"""
        Use this widget to identify problematic biases and stereotypes in 
        your data.
        
        nPMI scores for a word help to identify potentially
        problematic associations, ranked by how close the association is.
        
        nPMI bias scores for paired words help to identify how word
        associations are skewed between the selected selected words
        ([Aka et al., 2021](https://arxiv.org/abs/2103.03417)).
       
        You can select from gender and sexual orientation
        identity terms that appear in the dataset at least {min_vocab} times.
        
        The resulting ranked words are those that co-occur with both identity terms.
        
        The more *positive* the score, the more associated the word is with 
        the first identity term.
        The more *negative* the score, the more associated the word is with 
        the second identity term.
        
        -----
        """

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
    # display_list = [("general stats", gr_utils.expander_general_stats),
    #                 ("label distribution", gr_utils.expander_label_distribution),
    #                 ("text_lengths", gr_utils.expander_text_lengths),
    #                 ("duplicates", gr_utils.expander_text_duplicates),
    #                 ("npmi", gr_utils.npmi_widget),
    #                 ("zipf", gr_utils.expander_zipf)]

    return load_prepare_list, display_list

def display_title(dstats):
    title_str = f"### Showing: {dstats.dset_name} - {dstats.dset_config} - {dstats.split_name} - {'-'.join(dstats.text_field)}"
    logs.info("showing header")
    return title_str

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
    with gr.Blocks() as demo:
        widget_list = [DatasetDescription(), GeneralStats(), LabelDistribution(), TextLengths(), Npmi(), Zipf()]
        state = gr.State()
        with gr.Row():
            with gr.Column():
                dataset_args = display_initial_UI()
                # # TODO: Make this less of a weird outlier.
                show_perplexities = gr.Checkbox(label="Show text perplexities")
            with gr.Column():
                gr.Markdown("# Data Measurements Tool")
                # Initialize the main DMT class with the UI-provided arguments
                # When using the app (this file), try to use cache by default.
                title = gr.Markdown()
                for widget in widget_list:
                    widget.ui()

            def update_ui(dataset: str, config: str, split: str, feature: str):
                feature = ast.literal_eval(feature) #if isinstance(feature, str) else feature
                label_field, label_names = gr_utils.get_label_names(dataset, config, DATASET_NAME_TO_DICT)
                dstats = dmt_cls(dset_name=dataset, dset_config=config, split_name=split, text_field=feature,
                                 label_field=label_field, label_names=label_names, use_cache=True)
                load_prepare_list = [("general stats", dstats.load_or_prepare_general_stats),
                                     ("label distribution", dstats.load_or_prepare_labels),
                                     ("text_lengths", dstats.load_or_prepare_text_lengths),
                                     ("duplicates", dstats.load_or_prepare_text_duplicates),
                                     ("npmi", dstats.load_or_prepare_npmi),
                                     ("zipf", dstats.load_or_prepare_zipf)]
                dstats = load_or_prepare_widgets(dstats, load_prepare_list, show_perplexities=False,
                                                 live=True, pull_cache_from_hub=False)
                output = {title: display_title(dstats), state: dstats}
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
                new_measurements = update_ui(dataset, config, split, feature)
                new_dropdown.update(new_measurements)
                return new_dropdown

            def update_config(dataset: str, config: str):
                new_values = gr_utils.update_config(dataset, config, DATASET_NAME_TO_DICT)

                feature = new_values[0][1]
                split = new_values[1][1]
                new_dropdown = {
                    dataset_args["text_field"]: gr.Dropdown.update(choices=new_values[0][0], value=feature),
                    dataset_args["split_name"]: gr.Dropdown.update(choices=new_values[1][0], value=split)
                }
                new_measurements = update_ui(dataset, config, split, feature)
                new_dropdown.update(new_measurements)
                return new_dropdown

            measurements = [comp for output in widget_list for comp in output.output_components]
            demo.load(update_ui,
                      inputs=[dataset_args["dset_name"], dataset_args["dset_config"], dataset_args["split_name"], dataset_args["text_field"]],
                      outputs=[title, state] + measurements)

            dataset_args["dset_name"].change(update_dataset,
                                             inputs=[dataset_args["dset_name"]],
                                             outputs=[dataset_args["dset_config"],
                                              dataset_args["split_name"], dataset_args["text_field"],
                                             title, state] + measurements)

            dataset_args["dset_config"].change(update_config,
                                             inputs=[dataset_args["dset_name"], dataset_args["dset_config"]],
                                             outputs=[dataset_args["split_name"], dataset_args["text_field"],
                                             title, state] + measurements)
            widget_list[3].text_length_drop_down.change(
                widget_list[3].update_text_length_df,
                inputs=[widget_list[3].text_length_drop_down, state],
                outputs=[widget_list[3].text_length_df]
            )
            widget_list[4].npmi_first_word.change(
                widget_list[4].npmi_show,
                inputs=[widget_list[4].npmi_first_word, widget_list[4].npmi_second_word, state],
                outputs=[widget_list[4].npmi_df, widget_list[4].npmi_empty_text]
            )
            widget_list[4].npmi_second_word.change(
                widget_list[4].npmi_show,
                inputs=[widget_list[4].npmi_first_word, widget_list[4].npmi_second_word, state],
                outputs=[widget_list[4].npmi_df, widget_list[4].npmi_empty_text]
            )

            dataset_args["text_field"].change(update_ui,
                                              inputs=[dataset_args["dset_name"], dataset_args["dset_config"],
                                                      dataset_args["split_name"], dataset_args["text_field"]],
                                              outputs=[title, state] + measurements)

            dataset_args["split_name"].change(update_ui,
                                              inputs=[dataset_args["dset_name"], dataset_args["dset_config"],
                                                      dataset_args["split_name"], dataset_args["text_field"]],
                                              outputs=[title, state] + measurements)


    demo.launch()

if __name__ == "__main__":
    main()
