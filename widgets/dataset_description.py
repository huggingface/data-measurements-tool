import gradio as gr
import pandas as pd

from widgets.widget_base import Widget
from data_measurements.dataset_statistics import DatasetStatisticsCacheClass as dmt_cls
from utils.dataset_utils import HF_DESC_FIELD
import utils

logs = utils.prepare_logging(__file__)


class DatasetDescription(Widget):
    def __init__(self, dataset_name_to_dict):
        self.dataset_name_to_dict = dataset_name_to_dict
        self.description_markdown = gr.Markdown(render=False)
        self.description_df = gr.DataFrame(render=False, wrap=True)

    def render(self):
        with gr.TabItem("Dataset Description",):
            self.description_markdown.render()
            self.description_df.render()

    def update(self, dstats: dmt_cls):
        return {
            self.description_markdown: self.dataset_name_to_dict[dstats.dset_name][
                dstats.dset_config
            ][HF_DESC_FIELD],
            self.description_df: pd.DataFrame(dstats.dset_peek),
        }

    def add_events(self, state: gr.State):
        pass

    @property
    def output_components(self):
        return [self.description_markdown, self.description_df]
