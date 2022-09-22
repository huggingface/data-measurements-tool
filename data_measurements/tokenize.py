from sklearn.feature_extraction.text import CountVectorizer

class Tokenize:

    def __init__(self, text_series, lowercase=True):
        self.text_series = text_series
        self.lowercase = lowercase
        # Pattern for tokenization
        self.cvec = CountVectorizer(token_pattern="(?u)\\b\\w+\\b", lowercase=lowercase)
        self.tokenized_series = self.do_tokenization()

    def do_tokenization(self):
        """
        Tokenizes a Hugging Face dataset in the TEXT_FIELD.
        :return: tokenized_dset, a Hugging Face Dataset with a TOKENIZED_FIELD
        """
        sent_tokenizer = self.cvec.build_tokenizer()

        def tokenize_batch(examples):
            tok_sent = [tuple(sent_tokenizer(text)) for text in examples]
            return tok_sent

        tokenized_series = self.text_series.map(
            tokenize_batch,
            batched=True
        )
        logs.info("tokenized series is")
        logs.info(tokenized_series)
        return tokenized_series

    def get(self):
        return self.tokenized_series