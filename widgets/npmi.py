import gradio as gr
import pandas as pd

from widgets.widget_base import Widget
from data_measurements.dataset_statistics import DatasetStatisticsCacheClass as dmt_cls
import utils

logs = utils.prepare_logging(__file__)


class Npmi(Widget):
    def __init__(self):
        self.npmi_first_word = gr.Dropdown(
            render=False, label="What is the first word you want to select?"
        )
        self.npmi_second_word = gr.Dropdown(
            render=False, label="What is the second word you want to select?"
        )
        self.npmi_error_text = gr.Markdown(render=False)
        self.npmi_df = gr.DataFrame(render=False)
        self.npmi_empty_text = gr.Markdown(render=False)
        self.npmi_description = gr.Markdown(render=False)

    @property
    def output_components(self):
        return [
            self.npmi_first_word,
            self.npmi_second_word,
            self.npmi_error_text,
            self.npmi_df,
            self.npmi_description,
            self.npmi_empty_text,
        ]

    def render(self):
        with gr.TabItem("Word Association: nPMI"):
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
            output[self.npmi_description] = gr.Markdown.update(
                value=self.expander_npmi_description(min_vocab), visible=True
            )
            output[self.npmi_first_word] = gr.Dropdown.update(
                choices=available_terms, value=available_terms[0], visible=True
            )
            output[self.npmi_second_word] = gr.Dropdown.update(
                choices=available_terms[::-1], value=available_terms[-1], visible=True
            )
            output.update(
                self.npmi_show(available_terms[0], available_terms[-1], dstats)
            )
        else:
            output[self.npmi_error_text] = gr.Markdown.update(
                visible=True,
                value="No words found co-occurring with both of the selected identity terms.",
            )
        return output

    def npmi_show(self, term1, term2, dstats):
        npmi_stats = dstats.npmi_obj
        paired_results = npmi_stats.get_display(term1, term2)
        output = {}
        if paired_results.empty:
            output[self.npmi_empty_text] = gr.Markdown.update(
                value="""No words that co-occur enough times for results! Or there's a 🐛. 
                        Or we're still computing this one. 🤷""",
                visible=True,
            )
            output[self.npmi_df] = gr.DataFrame.update(visible=False)
        else:
            output[self.npmi_empty_text] = gr.Markdown.update(visible=False)
            logs.debug("Results to be shown in streamlit are")
            logs.debug(paired_results)
            s = pd.DataFrame(
                paired_results.sort_values(paired_results.columns[0], ascending=True)
            )
            s.index.name = "word"
            s = s.reset_index().round(4)
            bias_col = [col for col in s.columns if col != "word"]
            # Keep the dataframe from being crazy big.
            if s.shape[0] > 10000:
                bias_thres = max(abs(s[s[0]][5000]), abs(s[s[0]][-5000]))
                logs.info(f"filtering with bias threshold: {bias_thres}")
                s_filtered = s[s[0].abs() > bias_thres]
            else:
                s_filtered = s
            output[self.npmi_df] = s_filtered
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

    def add_events(self, state: gr.State):
        self.npmi_first_word.change(
            self.npmi_show,
            inputs=[self.npmi_first_word, self.npmi_second_word, state],
            outputs=[self.npmi_df, self.npmi_empty_text],
        )
        self.npmi_second_word.change(
            self.npmi_show,
            inputs=[self.npmi_first_word, self.npmi_second_word, state],
            outputs=[self.npmi_df, self.npmi_empty_text],
        )
