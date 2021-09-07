import argparse
import re
import statistics
import sys
import textwrap
from collections import Counter
from typing import Dict

import nltk
import numpy as np
import pandas as pd
import torch
import yaml
# If you don't have this installed, see https://huggingface.co/docs/datasets/installation.html
from datasets import load_dataset, Dataset
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
# See https://huggingface.co/transformers/installation.html
# Used from loading pretrained models and tokenizers
from transformers import AutoTokenizer, AutoModelWithLMHead, AutoModelForMaskedLM

parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description=textwrap.dedent('''
                                 
                                 Example for Glue dataset:
                                 python3 data_metrics_collab.py --dataset="glue" --config="ax" --split="test" --label-column="label" --label-type="discrete" --language-column="premise"
                                 
                                 Example for Asset dataset:
                                 python3 data_metrics_collab.py --dataset="asset" --config="ratings" --split="full" --label-column="rating" --label-type="real" --language-column="original"
                                 
                                 Example for IMDB dataset:
                                 python3 data_metrics_collab.py --dataset="imdb" --config="plain_text" --split="train" --label-column="label" --label-type="discrete" --language-column="text" --clean-html
                        
                                 '''))

parser.add_argument('--dataset', type=str,
                    help='Name of the dataset (Required)', required=True)
parser.add_argument('--config', type=str, required=True, help='Dataset configuration to use (Required)')
parser.add_argument('--split', type=str, required=True, help='Name of the dataset split to use (Required)')
# TODO: Handle situations that are not just straightforward single-cell labels.
parser.add_argument('--label-column', type=str, required=False, default='', help='Name of the column where the labels are (Required)')
parser.add_argument('--label-type', type=str, required=False, default='', choices=["discrete", "real"],
                    help='Type of label: discrete or real-valued (Required)')
parser.add_argument('--language-column', type=str, required=True,
                    help='Name of the column with the natural language is (Required)')
parser.add_argument('--clean-html', default=False, required=False, action="store_true",
                    help='Whether to clear out HTML in the text before processing (Optional)')

args = parser.parse_args()

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
STREAM_BATCH_SIZE = 5
STRIDE = 512

tokenizer = RegexpTokenizer(r"\w+")
wnl = WordNetLemmatizer()

# Using GPU/CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def perplex_model_data(**kwargs):
    """
    arguments:
    m_name : name of the model, e.g. "bert-base-uncased"
    d_name : name of the dataset, e.g. "oscar"
    d_option : dataset configuration, e.g. "unshuffled_deduplicated_en"
    d_split : e.g., train or test or full or validation
    d_streaming : True or False
    d_col : which column to analyze in the dataset (column with natural language in it)

    Calculates the perplexity of a model, dataset pair
    Probably will need to run several times for a given dataset, based on models that have been trained on other
    datasets (from the same task?)
    E.g. I'm uploading a new summarization dataset, and I'll calculate its perplexity based on models trained on
    existing summarization datasets.
    """
    # TODO: Ask What is the purpose of "modList"?
    # modlist = []
    # TODO: We may need multiple lists if we want to test a bunch of different models.
    # Masked Language Models are models trained with cloze-test-type masking over words during training.
    # maskedLM = ["bert-base-uncased"]
    # TODO: "Masked Head" models are.... tktktkt
    masked_head = ["t5-small"]
    tok = AutoTokenizer.from_pretrained(kwargs['m_name'])
    # TODO: fix the redundant loading of models!
    # if kwargs['m_name'] in modlist:
    #    pass
    if kwargs['m_name'] in masked_head:
        model = AutoModelWithLMHead.from_pretrained(kwargs['m_name'])
        # modlist.append(kwargs['m_name'])
    else:
        model = AutoModelForMaskedLM.from_pretrained(kwargs['m_name'])
        # modlist.append(kwargs['m_name'])
    if VERBOSE:
        print(str(kwargs['m_name']) + " model loaded!")
    try:
        data = load_dataset(kwargs['d_name'], kwargs['d_option'], split=kwargs['d_split'],
                            streaming=kwargs['d_streaming'])
    # TODO: NotImplemented wrt Extraction Protocol specifically
    except NotImplementedError:
        data = load_dataset(kwargs['d_name'], kwargs['d_option'], split=kwargs['d_split'], streaming=False)
        kwargs['d_streaming'] = False
    if VERBOSE:
        print(str(kwargs['d_name']) + " dataset loaded!")
    if kwargs['d_streaming']:
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
    except AttributeError:
        max_length = model.config.n_positions
    except:
        max_length = STRIDE

    # From https://huggingface.co/transformers/perplexity.html
    lls = []
    end_loc = 0
    for i in range(0, encodings.input_ids.size(1), STRIDE):
        begin_loc = max(i + STRIDE - max_length, 0)
        end_loc = min(i + STRIDE, encodings.input_ids.size(1))
        trg_len = end_loc - i  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        # ? Why 100?
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            log_likelihood = outputs[0] * trg_len

        lls.append(log_likelihood)
    # Note that this will produce a Zero Division error if the STRIDE is too large for the given range above.
    ppl = torch.exp(torch.stack(lls).sum() / end_loc)
    if VERBOSE:
        print("The perplexity of the " + str(kwargs['m_name']) + " model with the " + str(
            kwargs['d_name']) + " dataset is " + str(ppl.item()))
    return ppl.item()


def get_perplexity(dataset_name, config_name, dataset_split_name, langa_column_name, streaming=True):
    # TODO: What's a better way to do this? Can we check whether streaming is possible for the dataset before calling?
    # ALSO, it seems that '.orig' and similar are simply text files, which are streamable; they just don't have
    # the right name.
    lm_models = ["bert-base-uncased", "t5-small"]
    ppl_dict = {}
    for lm_model_name in lm_models:
        # TODO: Make STREAM_BATCH_SIZE an option that the user can specify (I guess...?)
        ppl = perplex_model_data(m_name=lm_model_name, d_name=dataset_name, d_option=config_name,
                                 d_split=dataset_split_name, d_streaming=streaming, d_size=STREAM_BATCH_SIZE,
                                 d_col=langa_column_name)
        ppl_dict[lm_model_name] = ppl
    # TODO: Figure out relevance of this from the jupyter notebook.
    # base_url = 'https://storage.googleapis.com/huggingface-nlp/cache/datasets/wikipedia/20200501.en/1.0.0/'
    # data_files = {"train": base_url + "wikipedia-train.parquet"}
    # wiki = load_dataset("parquet", data_files=data_files, split="train", streaming=True)
    # print(next(iter(wiki)))
    # {'title': 'Yangliuqing', 'text': 'Yangliuqing () is a market town in Xiqing District...'}
    return ppl_dict


def write_yaml(yaml_data, fid):
    stream = open(fid, 'w+')
    # default_flow_style allows us to use dicts to dump, rather than strings.
    yaml.dump(yaml_data, stream, default_flow_style=False)


# A 'Preprocessing' step -- Preprocessing should be in its own module
# TODO: Replace with a standard HTML-stripping utility.
def do_clean_html(raw_html: str) -> str:
    cleanr = re.compile("<.*?>")
    clean_text = re.sub(cleanr, '', raw_html)
    return clean_text


# Dataset Characteristics
def get_data_basics(input_dataset: Dataset, label_column_name: str, label_type: str) -> Dict:
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
    # Turn the Dataset into a data frame, using json_normalize so that the Dataset, too, will be a data frame.
    # Note that json_normalize is preferable to from_dict for handling of nested dicts.
    df = pd.json_normalize(input_dataset)
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
    if label_column_name:
        if label_type == "discrete":
            label_value_counts = str(df[label_column_name].value_counts()).replace('\n', ', ')
            basics_dict['label_counts'] = label_value_counts
        elif label_type == "real":
            np_array = np.array(df[label_column_name])
            basics_dict["label_min"] = float(round(np_array.min(), 4))
            basics_dict["label_max"] = float(round(np_array.max(), 4))
            basics_dict["label_mean"] = float(round(np_array.mean(), 4))
            basics_dict["label_var"] = float(round(np_array.var(), 4))
        else:
            sys.stderr.write("No label type specified; not calculating label statistics.\n")
    if VERBOSE:
        print('\n* Step 1 summary.')
        print(basics_dict)
    return basics_dict


# Vocabulary Size
def get_count_vocab(input_dataset: Dataset, langa_column_name: str, lower=True, language="english", clean_html=False) \
        -> Dict:
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
        if clean_html:
            sent = do_clean_html(sent)
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
               clean_html=False) -> Dict:
    data_dict = load_dataset(dataset_name, config_name)
    desired_dataset = data_dict[dataset_split_name]
    data_basics_dict = get_data_basics(desired_dataset, label_column_name=label_column_name, label_type=label_type)
    # Want to do this for both *source* and *target*
    data_vocab_dict = get_count_vocab(input_dataset=desired_dataset, langa_column_name=langa_column_name, lower=lower,
                                      language=language, clean_html=clean_html)
    data_text_dict = get_text_stats(desired_dataset, langa_column_name=langa_column_name)
    ppl_dict = get_perplexity(dataset_name, config_name, dataset_split_name, langa_column_name)
    # TODO: Run all the rest of the metrics
    output_yaml_data = {"Basic Data Characteristics": data_basics_dict, "Vocab Characteristics": data_vocab_dict,
                        "Text Characteristics": data_text_dict, "Perplexity": ppl_dict}
    return output_yaml_data


def do_glue_ax_dataset() -> Dict:
    # Dataset: glue-ax
    """ A manually-curated evaluation dataset for fine-grained analysis
    of system performance on a broad range of linguistic phenomena.
    This dataset evaluates sentence understanding through Natural Language Inference (NLI) problems.
    Use a model trained on MulitNLI to produce predictions for this dataset."""
    glue_ax_yaml = do_dataset(dataset_name="glue", config_name="ax", dataset_split_name="test",
                              label_column_name="label", label_type="discrete", langa_column_name="premise")
    return glue_ax_yaml


def do_asset_ratings_dataset() -> Dict:
    # Dataset: Asset-ratings
    asset_ratings_yaml = do_dataset(dataset_name="asset", config_name="ratings", dataset_split_name="full",
                                    label_column_name="rating", label_type="real", langa_column_name="original")
    return asset_ratings_yaml


def do_imdb_train_dataset() -> Dict:
    imdb_yaml = do_dataset(dataset_name="imdb", config_name="plain_text", dataset_split_name="train",
                           label_column_name="label", label_type="discrete", langa_column_name="text",
                           clean_html=True)
    return imdb_yaml


# Large datasets we stream; this requires different handling,
# specifically different read-in functions for a streamed vs fully-read
# TODO: Get dataset size to help solve whether to stream of read in full.
# (could be config, could be automatically grabbed)

def main(args) -> Dict:
    dataset_name = args.dataset
    config_name = args.config
    dataset_split_name = args.split
    label_column_name = args.label_column
    label_type = args.label_type
    langa_column_name = args.language_column
    clean_html = args.clean_html
    # This is going to need to expand as the other options will be necessary to distinguish between, e.g,
    # different splits of the dataset being analyzed.
    yaml_name = dataset_name + "-" + config_name + ".yaml"
    # TODO: Decide whether to handle these as kwargs, or attributes of a Data class, or something else.
    output_yaml = do_dataset(dataset_name=dataset_name, config_name=config_name, dataset_split_name=dataset_split_name,
                             label_column_name=label_column_name, label_type=label_type,
                             langa_column_name=langa_column_name, clean_html=clean_html)
    write_yaml(output_yaml, yaml_name)
    # TODO: Dump a json for the lists of ranked words, etc.; this can be used for visualization work.
    return [yaml_name]


if __name__ == "__main__":
    # execute only if run as a script
    if len(sys.argv) == 1:
        output_filenames = []
        print("No arguments specified; running through 3 datasets as an example: Glue, Asset, IMDB")
        # Lists of datasets and their deets are available at https://huggingface.co/datasets
        print("\n\n=== Processing Glue, ax...===")
        glue_yaml = do_glue_ax_dataset()
        write_yaml(glue_yaml, 'glue-ax.yaml')
        output_filenames += ['glue-ax.yaml']

        print("\n\n=== Processing Asset, ratings...===")
        asset_yaml = do_asset_ratings_dataset()
        write_yaml(asset_yaml, 'asset-ratings.yaml')
        output_filenames += ['asset-ratings.yaml']

        print("\n\n=== Processing IMDB ===")
        imdb_train_yaml = do_imdb_train_dataset()
        write_yaml(imdb_train_yaml, 'imdb-train.yaml')
        output_filenames += ['imdb-train.yaml']
    else:
        output_filenames = main(args)
    print("\n\nDone!  Output to yaml file(s):")
    print(' '.join(output_filenames))
    print("\n")







# ================= Scratch =================
# TODO: Are we still using this? Figure out how it fits in.
"""
def what_is_this_used_for_now:
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
"""
