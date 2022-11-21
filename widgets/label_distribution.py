import gradio as gr

from widgets.widget_base import Widget
from data_measurements.dataset_statistics import DatasetStatisticsCacheClass as dmt_cls
import utils

logs = utils.prepare_logging(__file__)


class LabelDistribution(Widget):
    def __init__(self):
        self.label_dist_plot = gr.Plot(render=False, visible=False)
        self.label_dist_no_label_text = gr.Markdown(
            value="No labels were found in the dataset", render=False, visible=False
        )
        self.label_dist_accordion = gr.Accordion(render=False, label="", open=False)

    def render(self):
        with gr.Accordion(label="Label Distribution", open=False):
            gr.Markdown(
                "Use this widget to see how balanced the labels in your dataset are."
            )
            self.label_dist_plot.render()
            self.label_dist_no_label_text.render()

    def update(self, dstats: dmt_cls):
        logs.info(f"FIGS labels: {bool(dstats.fig_labels)}")
        if dstats.fig_labels:
            output = {
                self.label_dist_plot: gr.Plot.update(
                    value=dstats.fig_labels, visible=True
                ),
                self.label_dist_no_label_text: gr.Markdown.update(visible=False),
            }
        else:
            output = {
                self.label_dist_plot: gr.Plot.update(visible=False),
                self.label_dist_no_label_text: gr.Markdown.update(visible=True),
            }
        return output

    @property
    def output_components(self):
        return [self.label_dist_plot, self.label_dist_no_label_text]

    def add_events(self, state: gr.State):
        pass
