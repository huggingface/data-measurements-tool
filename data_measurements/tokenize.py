import pandas as pd
import utils
from sklearn.feature_extraction.text import CountVectorizer

logs = utils.prepare_logging(__file__)

TEXT = "text"
TOKENIZED_TEXT = "tokenized_text"


class Tokenize:

    def __init__(self, text_dset, feature=TEXT, tok_feature=TOKENIZED_TEXT,
                 lowercase=True):
        self.text_dset = text_dset
        self.feature = feature
        self.tok_feature = tok_feature
        self.lowercase = lowercase
        # Pattern for tokenization
        self.cvec = CountVectorizer(token_pattern="(?u)\\b\\w+\\b",
                                    lowercase=lowercase)
        self.tokenized_dset = self.do_tokenization()

    def do_tokenization(self):
        """
        Tokenizes a Hugging Face dataset in the self.feature field.
        :return: Hugging Face Dataset with tokenized text in self.tok_feature.
        """
        sent_tokenizer = self.cvec.build_tokenizer()

        def tokenize_batch(examples):
            if self.lowercase:
                tok_sent = {
                    self.tok_feature: [tuple(sent_tokenizer(text.lower())) for
                                       text in examples[self.feature]]}
            else:
                tok_sent = {
                    self.tok_feature: [tuple(sent_tokenizer(text)) for text in
                                       examples[self.feature]]}
            return tok_sent

        tokenized_dset = self.text_dset.map(
            tokenize_batch,
            batched=True
        )
        logs.info("Tokenized the dataset.")
        return tokenized_dset

    def get(self):
        return self.tokenized_dset

    def get_df(self):
        return pd.DataFrame(self.tokenized_dset)