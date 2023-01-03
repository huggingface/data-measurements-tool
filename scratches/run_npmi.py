import datasets
from datasets import Dataset, load_dataset
from utils.dataset_utils import TEXT_FIELD, TOKENIZED_FIELD
from data_measurements.tokenize import Tokenize
from data_measurements.dataset_statistics import count_vocab_frequencies, calc_p_word, filter_vocab, IDENTITY_TERMS, MIN_VOCAB_COUNT, CNT
from data_measurements.npmi.npmi import nPMI, PMI

# ds = Dataset.from_dict({"text": [
#     "The man went to the park.",
#     "How many parks did that man go to?"
# ]})

ds = datasets.load_dataset("hate_speech18", "default", split="train")

# Small
ds = ds.shuffle().select(range(1000))

tokenized = Tokenize(ds, feature=TEXT_FIELD, tok_feature=TOKENIZED_FIELD).get_df()

word_count_df = count_vocab_frequencies(tokenized)
vocab_counts_df = calc_p_word(word_count_df)

def prepare_identity_terms(vocab_counts_df, identity_terms, min_count):
    """Pulled from npmi.py"""
    # Mask to get the identity terms
    true_false = [term in vocab_counts_df.index for term in
                  identity_terms]
    # List of identity terms
    word_list_tmp = [x for x, y in zip(identity_terms, true_false) if
                     y]
    # Whether said identity terms have a count > min_count
    true_false_counts = [
        vocab_counts_df.loc[word, CNT] >= min_count for word in
        word_list_tmp]
    # List of identity terms with a count higher than min_count
    avail_identity_terms = [word for word, y in
                            zip(word_list_tmp, true_false_counts) if y]
    return avail_identity_terms

identity_terms = prepare_identity_terms(vocab_counts_df, IDENTITY_TERMS, MIN_VOCAB_COUNT)

npmi_result = nPMI(vocab_counts_df, tokenized["tokenized_text"], identity_terms)

print(npmi_result.bias_results_dict)
