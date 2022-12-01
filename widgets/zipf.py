import gradio as gr
import pandas as pd

from widgets.widget_base import Widget
from data_measurements.dataset_statistics import DatasetStatisticsCacheClass as dmt_cls
import utils

logs = utils.prepare_logging(__file__)


class Zipf(Widget):
    def __init__(self):
        self.zipf_table = gr.DataFrame(render=False)
        self.alpha_warning = gr.Markdown(
            value="Your alpha value is a bit on the high side, which means that the distribution over words in this dataset is a bit unnatural. This could be due to non-language items throughout the dataset.",
            render=False,
            visible=False,
        )
        self.xmin_warning = gr.Markdown(
            value="The minimum rank for this fit is a bit on the high side, which means that the frequencies of your most common words aren't distributed as would be expected by Zipf's law.",
            render=False,
            visible=False,
        )
        self.zipf_summary = gr.Markdown(render=False)
        self.zipf_plot = gr.Plot(render=False)

    def render(self):
        with gr.TabItem("Vocabulary Distribution: Zipf's Law Fit"):
            gr.Markdown(
                "Use this widget for the counts of different words in your dataset, measuring the difference between the observed count and the expected count under Zipf's law."
            )
            gr.Markdown(
                """This shows how close the observed language is to an ideal
                        natural language distribution following [Zipf's law](https://en.wikipedia.org/wiki/Zipf%27s_law),
                        calculated by minimizing the [Kolmogorov-Smirnov (KS) statistic](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test)."""
            )
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

        output = {
            self.zipf_table: fit_results_table,
            self.zipf_summary: zipf_summary,
            self.zipf_plot: zipf_fig,
            self.alpha_warning: gr.Markdown.update(visible=False),
            self.xmin_warning: gr.Markdown.update(visible=False),
        }
        if z.alpha > 2:
            output[self.alpha_warning] = gr.Markdown.update(visible=True)
        if z.xmin > 5:
            output[self.xmin_warning] = gr.Markdown.update(visible=True)
        return output

    @property
    def output_components(self):
        return [
            self.zipf_table,
            self.zipf_plot,
            self.zipf_summary,
            self.alpha_warning,
            self.xmin_warning,
        ]

    def add_events(self, state: gr.State):
        pass
