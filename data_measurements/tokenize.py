from sklearn.feature_extraction.text import CountVectorizer

TEXT_FIELD = "text"
TOKENIZED_FIELD = "tokenized_text"

class Tokenize:

    def __init__(self, text_dset, df=None, dset=None, lowercase=True):
        self.df = df
        self.dset = dset
        self.text_dset = text_dset
        self.lowercase = lowercase
        # Pattern for tokenization
        self.cvec = CountVectorizer(token_pattern="(?u)\\b\\w+\\b", lowercase=lowercase)
        self.do_tokenization()

    def do_tokenization(self):
        """
        Tokenizes a Hugging Face dataset in the TEXT_FIELD.
        :return: tokenized_dset, a Hugging Face Dataset with a TOKENIZED_FIELD
        """
        sent_tokenizer = self.cvec.build_tokenizer()

        def tokenize_batch(examples):
            tok_sent = {
                TOKENIZED_FIELD: [
                    tuple(sent_tokenizer(text))
                    for text in examples[TEXT_FIELD]
                ]
            }
            return tok_sent

        tokenized_dset = self.text_dset.map(
            tokenize_batch,
            batched=True
        )
        return tokenized_dset