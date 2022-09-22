import logging
import pandas as pd
from datasets import load_metric
from os.path import exists
from os.path import join as pjoin
import utils
from utils import dataset_utils as ds_utils

logs = utils.prepare_logging(__file__)

TOK_MODEL = "gpt2"
PERPLEXITY = load_metric("perplexity")
PERPLEXITY_FIELD = "perplexity"


class DMTHelper:
    def __init__(self, dstats, load_only=False):
        self.dstats = dstats
        self.load_only = load_only
        self.results_dict = {}
        # Where in the Dataset object to find the text for the calculation
        self.text_field = ds_utils.OUR_TEXT_FIELD
        # Results in dataframe form
        self.df = None
        # Cache file
        self.perplexities_df_fid = pjoin(self.dstats.dataset_cache_dir,
                                         "perplexities_df.json")

    def run_DMT_processing(self):
        if self.dstats.use_cache and exists(self.perplexities_df_fid):
            self.df = ds_utils.read_df(self.perplexities_df_fid)
        elif not self.load_only:
            self.prepare_text_perplexities()
            if self.dstats.save:
                ds_utils.write_df(self.df, self.perplexities_df_fid)

    def prepare_text_perplexities(self):
        texts = self.dstats.text_dset[self.text_field]
        eval_results = PERPLEXITY.compute(input_texts=texts, model_id=TOK_MODEL)
        # TODO: What other stuff might be useful to grab?
        self.results_dict = {PERPLEXITY_FIELD: eval_results["perplexities"],
                             self.text_field: self.dstats.text_dset[self.text_field]}
        self.df = pd.DataFrame(self.results_dict).sort_values(
            by=PERPLEXITY_FIELD, ascending=False)

    def get_df(self):
        return self.df
