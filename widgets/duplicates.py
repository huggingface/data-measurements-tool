import gradio as gr

from widgets.widget_base import Widget
from data_measurements.dataset_statistics import DatasetStatisticsCacheClass as dmt_cls
import utils
import utils.dataset_utils as ds_utils

logs = utils.prepare_logging(__file__)


class Duplicates(Widget):
    def __init__(self):
        duplicates_text = f"""
        Use this widget to identify text strings that appear more than once. 

        A model's training and testing may be negatively affected by unwarranted duplicates ([Lee et al., 2021](https://arxiv.org/abs/2107.06499))

        ------

        ### Here is the list of all the duplicated items and their counts in the dataset. 
        """
        self.duplicates_intro = gr.Markdown(render=False, value=duplicates_text)
        self.duplicates_df = gr.DataFrame(render=False)
        self.duplicates_text = gr.Markdown(render=False)

    def render(self):
        with gr.TabItem(f"General Text Statistics"):
            self.duplicates_intro.render()
            self.duplicates_text.render()
            self.duplicates_df.render()

    def update(self, dstats: dmt_cls):
        output = {}

        if not dstats.duplicates_results:
            output[self.duplicates_df] = gr.DataFrame.update(visible=False)
            output[self.duplicates_text] = gr.Markdown.update(visible=True,
                                                              value="There are no duplicates in this dataset! 🥳")
        else:
            dupes_df = ds_utils.counter_dict_to_df(dstats.dups_dict)
            output[self.duplicates_df] = gr.DataFrame.update(visible=True, value=dupes_df)

            duplicates_text = f"The fraction of data that is duplicate is {str(round(dstats.dups_frac, 4))}"
            output[self.duplicates_text] = gr.Markdown.update(value=duplicates_text, visible=True)

        return output


    @property
    def output_components(self):
        return [
            self.duplicates_text,
            self.duplicates_df,
        ]

    def add_events(self, state: gr.State):
        pass