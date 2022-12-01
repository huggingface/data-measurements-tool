import gradio as gr
import pandas as pd

from widgets.widget_base import Widget
from data_measurements.dataset_statistics import DatasetStatisticsCacheClass as dmt_cls
import utils

logs = utils.prepare_logging(__file__)


class GeneralStats(Widget):
    def __init__(self):
        self.general_stats = gr.Markdown(render=False)
        self.general_stats_top_vocab = gr.DataFrame(render=False)
        self.general_stats_missing = gr.Markdown(render=False)
        self.general_stats_duplicates = gr.Markdown(render=False)

    def render(self):
        with gr.TabItem(f"General Text Statistics"):
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
        missing_text = (
            f"There are {dstats.text_nan_count} missing values in the dataset"
        )
        if dstats.dups_frac > 0:
            dupes_text = f"The dataset is {round(dstats.dups_frac * 100, 2)}% duplicates, For more information about the duplicates, click the 'Duplicates' tab below."
        else:
            dupes_text = "There are 0 duplicate items in the dataset"
        return {
            self.general_stats: general_stats_text,
            self.general_stats_top_vocab: top_vocab,
            self.general_stats_missing: missing_text,
            self.general_stats_duplicates: dupes_text,
        }

    @property
    def output_components(self):
        return [
            self.general_stats,
            self.general_stats_top_vocab,
            self.general_stats_missing,
            self.general_stats_duplicates,
        ]

    def add_events(self, state: gr.State):
        pass
