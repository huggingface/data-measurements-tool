import sys
import re

import nltk
import numpy as np
import pandas as pd
# If you don't have this installed, see https://huggingface.co/docs/datasets/installation.html
from datasets import load_dataset, DatasetDict, Dataset, IterableDatasetDict, IterableDataset
from nltk.lm import MLE
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.stem import WordNetLemmatizer

# Used later in vocab statistics.
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
from nltk.tokenize import RegexpTokenizer, sent_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
import sentencepiece, statistics

from typing import Dict, Tuple, Sequence, List, Union
InputDataType = Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]
import yaml

# Make sure to give us data peeks, etc.
VERBOSE = True

tokenizer = RegexpTokenizer(r"\w+")
wnl = WordNetLemmatizer()


# A 'Preprocessing' step -- Preprocessing should be in its own module
def cleanhtml(raw_html: str) -> str:
    cleanr = re.compile("<.*?>")
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext


# Dataset Characteristics
def get_data_basics(input_data: InputDataType, label_column: str, json_column=False, label_type='discrete') -> yaml:
    df = pd.DataFrame.from_dict(input_data)
    yaml_data = ""
    if json_column:
        df = pd.json_normalize(df[json_column])
    if VERBOSE:
        print("\n* Peek at data:")
        print(df.head())
    data_shape = df.shape
    num_rows = "\nNumber of rows: %s" % data_shape[0]
    num_cols = "\nNumber of columns: %s" % data_shape[1]
    yaml_data += num_rows
    yaml_data += num_cols
    if label_type == "discrete":
        label_value_counts = "\nLabel counts:\n" + str(df[label_column].value_counts())
        yaml_data += label_value_counts
    elif label_type == "real":
        np_array = np.array(df[label_column])
        min_val = "\nLabel min: " + str(np_array.min())
        yaml_data += min_val
        max_val = "\nLabel Mmx: " + str(np_array.max())
        yaml_data += max_val
        mean_val = "\nLabel mean: " + str(np_array.mean())
        yaml_data += mean_val
        var_val = "\nLabel variance: " + str(np_array.var())
        yaml_data += var_val
    print(yaml_data)
    return yaml_data


# Vocabulary Size
def get_count_vocab(input_data, lower=True, language="english"):
    # Counts the number of tokens, with or without lowercase normalization.
    tokenized_text = tokenizer.tokenize(input_data)
    language_stopwords = stopwords.words(language)
    if lower:
        vocab = FreqDist(word.lower() for word in tokenized_text)
        # Are all the stopwords in lowercase?
        filtered_vocab = FreqDist(word.lower() for word in tokenized_text if word.lower() not in language_stopwords)
        lem_vocab = FreqDist(
            wnl.lemmatize(word.lower()) for word in tokenized_text if word.lower() not in language_stopwords)
    else:
        vocab = FreqDist(word for word in tokenized_text)
        filtered_vocab = FreqDist(word for word in tokenized_text if word not in language_stopwords)
        lem_vocab = FreqDist(wnl.lemmatize(word for word in tokenized_text if word not in language_stopwords))
    print("There are " + str(len(vocab)) + " words including stop words")
    print("There are " + str(len(filtered_vocab)) + " words after removing stop words")
    print("There are " + str(len(lem_vocab)) + " words after removing stop words and lemmatizing")


# Instance Characteristics
def get_text_stats(text_list):
    # Calculates sufficient statistics for text-based instances: average, mean, medium
    total_lens = 0
    alllengths = []
    # TODO(meg): Turn into yaml output
    for i, sent in enumerate(text_list):
        lent = len(tokenizer.tokenize(sent))
        alllengths.append(lent)
        total_lens += lent
    avg_sent_len = total_lens / i
    print("The average sentence length is: " + str(round(avg_sent_len, 4)) + " words.")
    print("The mean sentence length is: " + str(statistics.mean(alllengths)) + " words.")
    print("The mean sentence length is: " + str(statistics.median(alllengths)) + " words.")


# Per-label characteristics
# TBD. Sasha had focused on imdb: most frequent words for each label,
# and words only present in the top 10,000 most common positive/negative words

# Dataset: glue-ax
""" A manually-curated evaluation dataset for fine-grained analysis 
of system performance on a broad range of linguistic phenomena. 
This dataset evaluates sentence understanding through Natural Language Inference (NLI) problems. 
Use a model trained on MulitNLI to produce predictions for this dataset."""
dataset: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset] = load_dataset("glue", "ax")

yaml_data = get_data_basics(dataset, label_column="label", json_column="test")

asset = load_dataset("asset", "ratings")

yaml_data = get_data_basics(asset, label_column="rating", json_column="full", label_type="real")

imdb = load_dataset("imdb")
imdb_train = imdb['train']
yaml_data = get_data_basics(imdb_train, label_column="label")

# Preprocessing for IMDB
alllist = [cleanhtml(sent) for sent in imdb_train["text"]]
imdb_text = ' '.join(s for s in alllist)
get_count_vocab(imdb_text, True)


# TODO: Redo in a way that doesn't require kenlm.
def score_ppl(test: str):
    # Perplexity
    ## Based on Wikipedia using the pretrained model from CCNet https://github.com/facebookresearch/cc_net/
    test = alllist[1]
    sp_model = sentencepiece.SentencePieceProcessor('en.sp.model')
    # TBD. Issue with accessing kenlm.
    # model = kenlm.Model('/home/sasha/Documents/MilaPostDoc/Python/cc_net/data/lm_sp/en.arpa.bin')
    score = 0
    doc_length = 0
    for sentence in sent_tokenize(test):
        sentence = sp_model.encode_as_pieces(sentence)
        # score += model.score(" ".join(sentence))
        doc_length += len(sentence) + 1
    print("Final score: " + str(score))


# score_ppl()

# from https://stackoverflow.com/questions/54941966/how-can-i-calculate-perplexity-using-nltk/55043954

train_sentences = ['an apple', 'an orange']
tokenized_text = [list(map(str.lower, nltk.tokenize.word_tokenize(sent)))
                  for sent in train_sentences]
n = 1
train_data, padded_vocab = padded_everygram_pipeline(n, tokenized_text)
model = MLE(n)
model.fit(train_data, padded_vocab)

test_sentences = ['an apple', 'an ant']
tokenized_text = [list(map(str.lower, nltk.tokenize.word_tokenize(sent)))
                  for sent in test_sentences]

test_data, _ = padded_everygram_pipeline(n, tokenized_text)
for test in test_data:
    print("MLE Estimates:", [((ngram[-1], ngram[:-1]), model.score(ngram[-1], ngram[:-1])) for ngram in test])

test_data, _ = padded_everygram_pipeline(n, tokenized_text)

for i, test in enumerate(test_data):
    print("PP({0}):{1}".format(test_sentences[i], model.perplexity(test)))
