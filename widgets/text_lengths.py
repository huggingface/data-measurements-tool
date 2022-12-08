import gradio as gr

from widgets.widget_base import Widget
from data_measurements.dataset_statistics import DatasetStatisticsCacheClass as dmt_cls
import utils

logs = utils.prepare_logging(__file__)


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

    def render(self):
        with gr.TabItem("Text Lengths"):
            gr.Markdown(
                "Use this widget to identify outliers, particularly suspiciously long outliers."
            )
            gr.Markdown(
                "Below, you can see how the lengths of the text instances in your "
                "dataset are distributed."
            )
            gr.Markdown(
                "Any unexpected peaks or valleys in the distribution may help to "
                "identify instances you want to remove or augment."
            )
            gr.Markdown(
                "### Here is the count of different text lengths in " "your dataset:"
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
        # TODO: Add text on choosing the length you want to the dropdown.
        output = {
            self.text_length_distribution_plot: dstats.length_obj.fig_lengths,
            self.text_length_explainer: explainer_text,
        }
        if dstats.length_obj.lengths_df is not None:
            import numpy as np

            choices = np.sort(dstats.length_obj.lengths_df["length"].unique())[
                ::-1
            ].tolist()
            output[self.text_length_drop_down] = gr.Dropdown.update(
                choices=choices, value=choices[0]
            )
            output[self.text_length_df] = self.update_text_length_df(choices[0], dstats)
        else:
            output[self.text_length_df] = gr.update(visible=False)
            output[self.text_length_drop_down] = gr.update(visible=False)
        return output

    @property
    def output_components(self):
        return [
            self.text_length_distribution_plot,
            self.text_length_explainer,
            self.text_length_drop_down,
            self.text_length_df,
        ]

    def add_events(self, state: gr.State):
        self.text_length_drop_down.change(
            self.update_text_length_df,
            inputs=[self.text_length_drop_down, state],
            outputs=[self.text_length_df],
        )
