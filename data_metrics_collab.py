import itertools
import re
import statistics
from collections import ChainMap, Counter
from typing import Dict

import nltk
import numpy as np
import pandas as pd
import torch
import yaml
# If you don't have this installed, see https://huggingface.co/docs/datasets/installation.html
from datasets import load_dataset, Dataset
from nltk.corpus import stopwords
from nltk.lm import MLE
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
# See https://huggingface.co/transformers/installation.html
# Used from loading pretrained models and tokenizers
from transformers import AutoTokenizer, AutoModelWithLMHead, AutoModelForMaskedLM

# Used later in vocab statistics.
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

"""
The DatasetDict structure is like this:

DatasetDict({
    USER_PROVIDED_SPLIT_KEY: Dataset({
        features: [USER_PROVIDED_SOURCE_COLUMN_NAME, USER_PROVIDED_TARGET_COLUMN_NAME, USER_PROVIDED_SCORE_COLUMN_NAME],
        num_rows: <automatic>
    })
})

# USER_PROVIDED_SPLIT_KEY: this could be 'train', 'test', 'full', etc.
"""

# Make sure to give us data peeks, etc.
VERBOSE = True

tokenizer = RegexpTokenizer(r"\w+")
wnl = WordNetLemmatizer()

# Using GPU/CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def perplex_model_data(**kwargs):
    '''
    arguments:
    m_name : name of the model, e.g. "bert-base-uncased"
    d_name : name of the dataset, e.g. "oscar"
    d_option : dataset options, e.g. "unshuffled_deduplicated_en"
    d_split : train or test
    d_streaming : True or False
    d_size : how many instances to load if streaming
    d_col : which column to analyze in the dataset

    Calculates the perplexity of a model, dataset pair
    Probably will need to run several times for a given dataset, based on models that have been trained on other datasets (from the same task?)
    E.g. I'm uploading a new summarization dataset, and I'll calculate its perplexity based on models trained on existing summarization datasets.
    '''
    modlist = []
    # TODO: We may need multiple lists if we want to test a bunch of different models.
    maskedLM = ["bert-base-uncased"]
    maskedHead = ["t5-small"]
    tok = AutoTokenizer.from_pretrained(kwargs['m_name'])
    # TODO: fix the redundant loading of models!
    if kwargs['m_name'] in modlist:
        pass
    elif kwargs['m_name'] in maskedHead:
        model = AutoModelWithLMHead.from_pretrained(kwargs['m_name'])
        modlist.append(kwargs['m_name'])
    else:
        model = AutoModelForMaskedLM.from_pretrained(kwargs['m_name'])
        print(str(kwargs['m_name']) + " model loaded!")
        modlist.append(kwargs['m_name'])
    data = load_dataset(kwargs['d_name'], kwargs['d_option'], split=kwargs['d_split'], streaming=kwargs['d_streaming'])
    print(str(kwargs['d_name']) + " dataset loaded!")
    if kwargs['d_streaming'] == True:
        data_head = data.take(kwargs['d_size'])
        # try:
        #    text = [l['text'] for l in list(data_head)]
        # except:
        #    # TODO: figure out a better way to do this
        feature = kwargs['d_col']
        text = [l[feature] for l in list(data_head)]

    else:
        feature = next(iter(data.features))
        text = data[feature][:kwargs['d_size']]

    encodings = tok('\n\n'.join(text), return_tensors='pt')
    try:
        max_length = model.config.max_position_embeddings
    except AttributeError as error:
        max_length = model.config.n_positions
    except:
        max_length = 512

    # From https://huggingface.co/transformers/perplexity.html
    stride = 512
    lls = []
    for i in range(0, encodings.input_ids.size(1), stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        trg_len = end_loc - i  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            log_likelihood = outputs[0] * trg_len

        lls.append(log_likelihood)

    ppl = torch.exp(torch.stack(lls).sum() / end_loc)
    print("The perplexity of the " + str(kwargs['m_name']) + " model with the " + str(
        kwargs['d_name']) + " dataset is " + str(ppl.item()))


def get_combo_pp(models, datasets):
    '''
    Takes a list of [models] and [datasets] and their respective options (same as above) and calculates
    all of the perplexity scores of the *combinations* of models and datasets
    e.g.
    models = ["bert-base-uncased", "t5-small"]

    datasets= [
        ["asset", "ratings", "full", False, 5],
        ["oscar", "unshuffled_deduplicated_en", "train", True,5],
        ["imdb","plain_text", "test", False ,5],
        ["poem_sentiment","plain_text", "test", True ,5],
        ["c4", "en", "train", True, 5]
       ]
    '''
    combos = list(itertools.product(models, datasets))
    print("There are " + str(len(list(combos))) + " combinations of models and datasets.")
    for m, d in combos:
        print('Analyzing the ' + m + ' model and the ' + d[0] + ' dataset.')
        print(d)
        perplex_model_data(m_name=m, d_name=d[0], d_option=d[1], d_split=d[2], d_streaming=d[3], d_size=d[4],
                           d_col=d[5])


def get_perplexity(dataset_name, config_name, dataset_split_name, langa_column_name, streaming=True):
    # TODO: What's a better way to do this? Can we check whether streaming is possible for the dataset before calling?
    # ALSO, it seems that '.orig' and similar are simply text files, which are streamable; they just don't have
    # the right name.
    try:
        data = load_dataset(dataset_name, config_name, split=dataset_split_name, streaming=streaming)
    except NotImplementedError:
        data = load_dataset(dataset_name, config_name, split=dataset_split_name, streaming=False)
    mods = ["bert-base-uncased", "t5-small"]
    # These are organized as dataset name, config name, data split name, where the langa stuff is written
    # TODO: Pulls these out more automatically.
    datas = [
        ["asset", "ratings", "full", False, 5, "simplification"],
        ["oscar", "unshuffled_deduplicated_en", "train", True, 5, "text"],
        ["imdb", "plain_text", "test", False, 5, "text"],
        ["poem_sentiment", "plain_text", "test", True, 5, "verse_text"],
        ["c4", "en", "train", True, 5, "text"]
    ]
    perplex_model_data(m_name="bert-base-uncased", d_name="oscar", d_option="unshuffled_deduplicated_en",
                       d_split="train", d_streaming=True, d_size=5, d_col="text")
    get_combo_pp(mods, datas)
    base_url = 'https://storage.googleapis.com/huggingface-nlp/cache/datasets/wikipedia/20200501.en/1.0.0/'
    data_files = {"train": base_url + "wikipedia-train.parquet"}
    wiki = load_dataset("parquet", data_files=data_files, split="train", streaming=True)
    print(next(iter(wiki)))
    # {'title': 'Yangliuqing', 'text': 'Yangliuqing () is a market town in Xiqing District...'}


def write_yaml(yaml_data, fid):
    stream = open(fid, 'w+')
    # default_flow_style allows us to use dicts to dump, rather than strings.
    yaml.dump(yaml_data, stream, default_flow_style=False)


# A 'Preprocessing' step -- Preprocessing should be in its own module
# TODO: Replace with a standard HTML-stripping utility.
def clean_html(raw_html: str) -> str:
    cleanr = re.compile("<.*?>")
    clean_text = re.sub(cleanr, '', raw_html)
    return clean_text


# Dataset Characteristics
def get_data_basics(input_dataset: Dataset, label_column_name: str, label_type='discrete') -> Dict:
    # Should we ask about deduping?!
    """
    # Takes a DatasetDict & isolates the Dataset of interest as a dataframe using json_normalize
    # on the value of the relevant Dataset key (dataset_column_name).
    # We will need to know this Dataset key name from a config file.
    :type input_dataset: Dataset
    :type label_column_name: str
    :type label_type: str
    :rtype: Dict
    """
    basics_dict = {}
    num_rows = input_dataset.num_rows
    # Turn the Dataset into a data frame.
    df = pd.DataFrame.from_dict(input_dataset)
    # Grab the Dataset itself from the data frame, using json_normalize so that the Dataset, too, will be a data frame.
    # TODO: .info() as the summarization mechanism, as well as .isnull() to get a view of the NaN
    if VERBOSE:
        print('\n* Step 1: Peek at data. Calculating dataset characteristics.')
        print(df.head())
    # Will need to grab 'shape' when we're streaming;
    # Or will we?!  Depends how we're streaming in.
    # Is each chunk of the stream a 'DatasetDict'?
    # If so, then 'num_rows' specifies all of the dataset rows.
    # If not, then we will need to use shape,
    # And it will be a smaller number of rows when streaming into here.
    data_shape = df.shape
    assert (data_shape[0] == num_rows)
    basics_dict['num_rows'] = data_shape[0]
    basics_dict['num_cols'] = data_shape[1]
    if label_type == "discrete":
        label_value_counts = str(df[label_column_name].value_counts()).replace('\n', ', ')
        basics_dict['label_counts'] = label_value_counts
    elif label_type == "real":
        np_array = np.array(df[label_column_name])
        basics_dict["label_min"] = round(np_array.min(), 4)
        basics_dict["label_max"] = round(np_array.max(), 4)
        basics_dict["label_mean"] = round(np_array.mean(), 4)
        basics_dict["label_var"] = round(np_array.var(), 4)
    if VERBOSE:
        print('\n* Step 1 summary.')
        print(basics_dict)
    return basics_dict


# Vocabulary Size
def get_count_vocab(input_dataset: Dataset, langa_column_name: str, lower=True, language="english",
                    do_clean_html=False) -> Dict:
    vocab_dict = {}
    vocab = Counter()
    filtered_vocab = Counter()
    lem_vocab = Counter()
    # TODO: A single ID will have multiple duplicated sources if there are multiple translations.
    # So yeah. We have to deal with that.
    # Turn the Dataset into a data frame.
    df = pd.DataFrame.from_dict(input_dataset)
    # Counts the number of tokens, with or without lowercase normalization.
    if VERBOSE:
        print('\n* Step 2: Calculating statistics on text looking like this.')
        print(df[langa_column_name].head())
    # TODO: Do this the fast way.
    for sent in df[langa_column_name]:
        if do_clean_html:
            sent = clean_html(sent)
        tokenized_text = tokenizer.tokenize(sent)
        language_stopwords = stopwords.words(language)
        if lower:
            vocab_tmp = FreqDist(word.lower() for word in tokenized_text)
            # Are all the stopwords in lowercase?
            filtered_vocab_tmp = FreqDist(
                word.lower() for word in tokenized_text if word.lower() not in language_stopwords)
            lem_vocab_tmp = FreqDist(
                wnl.lemmatize(word.lower()) for word in tokenized_text if word.lower() not in language_stopwords)
        else:
            vocab_tmp = FreqDist(word for word in tokenized_text)
            filtered_vocab_tmp = FreqDist(word for word in tokenized_text if word not in language_stopwords)
            lem_vocab_tmp = FreqDist(wnl.lemmatize(word for word in tokenized_text if word not in language_stopwords))
        vocab.update(vocab_tmp)
        filtered_vocab.update(filtered_vocab_tmp)
        lem_vocab.update(lem_vocab_tmp)
    if VERBOSE:
        print("\n* Step 2 summary.")
        print("There are {0} words including stop words".format(str(len(vocab))))
        print("There are " + str(len(filtered_vocab)) + " words after removing stop words")
        print("There are " + str(len(lem_vocab)) + " words after removing stop words and lemmatizing")
    vocab_dict['num_words'] = len(vocab)
    vocab_dict['num_filtered_words'] = len(filtered_vocab)
    vocab_dict['num_lemmatized_words'] = len(lem_vocab)
    return vocab_dict


# def get_label_stats()
# TODO: Show the top tokens by count for each label.
# Proportion within dataset
# Closed and open pi chart?
# Find the closed word lists for the different languages in order to calculate distributional statistics.


# def do_zipf():
# Then show Zipf plot for top 400 tokens.
# Real vs. Projected value according to Zipf; flag those that are a difference higher than x
# # for what's predicted by the law.
# For closed class words, what could we say about that means semantically?

# Instance Characteristics
def get_text_stats(input_dataset: Dataset, langa_column_name: str) -> Dict:
    # Calculates sufficient statistics for text-based instances: average, mean, median
    total_lens = 0
    all_lengths = []
    text_dict = {}
    i = 1
    # Turn the Dataset into a data frame.
    df = pd.DataFrame.from_dict(input_dataset)
    if VERBOSE:
        print("\n* Step 3: Get text stats. Text is looking like this.")
        print(df.head())
    for sent in df[langa_column_name]:  # enumerate(source_text):
        lent = len(tokenizer.tokenize(sent))
        all_lengths.append(lent)
        total_lens += lent
        i += 1.0
    avg_sent_len = total_lens / i
    if VERBOSE:
        print("\n* Step 3 summary.")
        # Hm, weird that average and mean are different numbers. Must be rounding?
        print("The average sentence length is: " + str(avg_sent_len) + " words.")
        print("The mean sentence length is: " + str(statistics.mean(all_lengths)) + " words.")
        print("The median sentence length is: " + str(statistics.median(all_lengths)) + " words.")
    text_dict['mean_sent_len'] = round(statistics.mean(all_lengths), 4)
    text_dict['median_sent_len'] = round(statistics.median(all_lengths), 4)
    return text_dict


def do_dataset(dataset_name: str, config_name: str, dataset_split_name: str, label_column_name: str,
               label_type="discrete", langa_column_name="text", lower=True, language="english",
               do_clean_html=False) -> ChainMap:
    data_dict = load_dataset(dataset_name, config_name)
    desired_dataset = data_dict[dataset_split_name]
    data_basics_dict = get_data_basics(desired_dataset, label_column_name=label_column_name, label_type=label_type)
    # Want to do this for both *source* and *target*
    data_vocab_dict = get_count_vocab(input_dataset=desired_dataset, langa_column_name=langa_column_name, lower=lower,
                                      language=language, do_clean_html=do_clean_html)
    data_text_dict = get_text_stats(desired_dataset, langa_column_name=langa_column_name)
    get_perplexity(dataset_name, config_name, dataset_split_name, langa_column_name)
    # TODO: Run all the rest of the metrics
    output_yaml_data = ChainMap(data_basics_dict, data_vocab_dict, data_text_dict)
    return output_yaml_data


def do_glue_ax_dataset() -> ChainMap:
    # Dataset: glue-ax
    """ A manually-curated evaluation dataset for fine-grained analysis
    of system performance on a broad range of linguistic phenomena.
    This dataset evaluates sentence understanding through Natural Language Inference (NLI) problems.
    Use a model trained on MulitNLI to produce predictions for this dataset."""
    glue_ax_yaml = do_dataset(dataset_name="glue", config_name="ax", dataset_split_name="test",
                              label_column_name="label", label_type="discrete", langa_column_name="premise")
    return glue_ax_yaml


def do_asset_ratings_dataset() -> ChainMap:
    # Dataset: Asset-ratings
    asset_ratings_yaml = do_dataset(dataset_name="asset", config_name="ratings", dataset_split_name="full",
                                    label_column_name="rating", label_type="real", langa_column_name="original")
    return asset_ratings_yaml


def do_imdb_train_dataset() -> ChainMap:
    imdb_yaml = do_dataset(dataset_name="imdb", config_name="plain_text", dataset_split_name="train",
                           label_column_name="label", label_type="discrete", langa_column_name="text",
                           do_clean_html=True)
    return imdb_yaml


# Large datasets we stream; this requires different handling,
# specifically different read-in functions for a streamed vs fully-read
# TODO: Get dataset size to help solve whether to stream of read in full.
# (could be config, could be automatically grabbed)

# Lists of datasets and their deets are available at https://huggingface.co/datasets
print("\n\n=== Processing Glue, ax...===")
glue_yaml = do_glue_ax_dataset()
write_yaml(glue_yaml, 'glue-ax.yaml')

print("\n\n=== Processing Asset, ratings...===")
asset_yaml = do_asset_ratings_dataset()
write_yaml(asset_yaml, 'asset-ratings.yaml')

print("\n\n=== Processing IMDB ===")
imdb_train_yaml = do_imdb_train_dataset()
write_yaml(imdb_train_yaml, 'imdb-train.yaml')

# TODO: Clean up everything below this.
# TODO: Redo in a way that doesn't require kenlm.
"""
def score_ppl_kenlm(test: str):
    # Perplexity
    ## Based on Wikipedia using the pretrained model from CCNet https://github.com/facebookresearch/cc_net/
    test = alllist[1]
    # sp_model = sentencepiece.SentencePieceProcessor('en.sp.model')
    # TBD. Issue with accessing kenlm.
    # model = kenlm.Model('/home/sasha/Documents/MilaPostDoc/Python/cc_net/data/lm_sp/en.arpa.bin')
    score = 0
    doc_length = 0
    for sentence in sent_tokenize(test):
        sentence = sp_model.encode_as_pieces(sentence)
        # score += model.score(" ".join(sentence))
        doc_length += len(sentence) + 1
    print("Final score: " + str(score))
"""


def score_ppl(test: str, modelname: str):
    tokenizer = AutoTokenizer.from_pretrained(modelname)
    model = AutoModelForMaskedLM.from_pretrained(modelname)
    encodings = tokenizer(test, return_tensors='pt')
    max_length = model.config.max_position_embeddings
    # From https://huggingface.co/transformers/perplexity.html
    stride = 512
    lls = []
    end_loc = 1
    for i in range(0, encodings.input_ids.size(1), stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        trg_len = end_loc - i  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            log_likelihood = outputs[0] * trg_len

        lls.append(log_likelihood)

    ppl = torch.exp(torch.stack(lls).sum() / end_loc)
    print("The perplexity of the " + str(modelname) + " model for the test string is " + str(ppl.item()))


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
