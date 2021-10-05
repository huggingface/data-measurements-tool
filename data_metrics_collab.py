import argparse
import re
import statistics
import sys
import textwrap
from collections import Counter
from typing import Dict, List

import matplotlib.pyplot as plt
import nltk
import torch
import yaml
# If you don't have this installed, see https://huggingface.co/docs/datasets/installation.html
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
# See https://huggingface.co/transformers/installation.html
# Used from loading pretrained models and tokenizers
from transformers import AutoTokenizer, AutoModelWithLMHead, \
    AutoModelForMaskedLM

plt.style.use('fivethirtyeight')
import powerlaw
import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import CountVectorizer
from scipy.stats import ks_2samp, zipf

# See http://docs.allennlp.org/main/api/fairness/bias_metrics/#associationwithoutgroundtruth

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description=textwrap.dedent('''
                                 
                                 Example for Glue dataset:
                                 python3 data_metrics_collab.py --dataset="glue" --config="ax" --split="test" --label-column="label" --label-type="discrete" --language-column="premise"
                                 
                                 Example for Asset dataset:
                                 python3 data_metrics_collab.py --dataset="asset" --config="ratings" --split="full" --label-column="rating" --label-type="real" --language-column="original"
                                 
                                 Example for IMDB dataset:
                                 python3 data_metrics_collab.py --dataset="imdb" --config="plain_text" --split="train" --label-column="label" --label-type="discrete" --language-column="text" --clean-html
                                 
                                 Example for summarization datasets:
                                 python3 data_metrics_collab.py --dataset="xsum" --config="default" --split="train" --language-column="summary"
                                 python3 data_metrics_collab.py --dataset="csebuetnlp/xlsum" --config="english" --split="train" --language-column="summary"         
                        
                                 '''))

parser.add_argument('--dataset', type=str,
                    help='Name of the dataset (Required)', required=True)
parser.add_argument('--config', type=str, required=True,
                    help='Dataset configuration to use (Required)')
parser.add_argument('--split', type=str, required=True,
                    help='Name of the dataset split to use (Required)')
# TODO: Handle situations that are not just straightforward single-cell labels.
parser.add_argument('--label-column', type=str, required=False, default='',
                    help='Name of the column where the labels are (Required)')
parser.add_argument('--label-type', type=str, required=False, default='',
                    choices=["discrete", "real"],
                    help='Type of label: discrete or real-valued (Required)')
parser.add_argument('--language-column', type=str, required=True,
                    help='Name of the column with the natural language is (Required)')
parser.add_argument('--clean-html', default=False, required=False,
                    action="store_true",
                    help='Whether to clear out HTML in the text before processing (Optional)')
parser.add_argument('--streaming', default=False, required=False,
                    action="store_true",
                    help='Whether to stream data in (Optional)')

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
        data = load_dataset(kwargs['d_name'], kwargs['d_option'],
                            split=kwargs['d_split'],
                            streaming=kwargs['d_streaming'])
    # TODO: NotImplemented wrt Extraction Protocol specifically
    except NotImplementedError:
        data = load_dataset(kwargs['d_name'], kwargs['d_option'],
                            split=kwargs['d_split'], streaming=False)
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
        print("The perplexity of the " + str(
            kwargs['m_name']) + " model with the " + str(
            kwargs['d_name']) + " dataset is " + str(ppl.item()))
    return ppl.item()


def get_perplexity(dataset_name, config_name, dataset_split_name,
                   langa_column_name, streaming=True):
    # TODO: What's a better way to do this? Can we check whether streaming is possible for the dataset before calling?
    # ALSO, it seems that '.orig' and similar are simply text files, which are streamable; they just don't have
    # the right name.
    lm_models = ["bert-base-uncased", "t5-small"]
    ppl_dict = {}
    for lm_model_name in lm_models:
        # TODO: Make STREAM_BATCH_SIZE an option that the user can specify (I guess...?)
        ppl = perplex_model_data(m_name=lm_model_name, d_name=dataset_name,
                                 d_option=config_name,
                                 d_split=dataset_split_name,
                                 d_streaming=streaming,
                                 d_size=STREAM_BATCH_SIZE,
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
def get_data_basics(input_dataset, label_column_name: str,
                    label_type: str) -> Dict:
    # Should we ask about deduping?!
    """
    # Takes a DatasetDict & isolates the Dataset of interest as a dataframe using json_normalize
    # on the value of the relevant Dataset key (dataset_column_name).
    # We will need to know this Dataset key name from a config file.
    :type input_dataset: DataFrame
    :type label_column_name: str
    :type label_type: str
    :rtype: Dict
    """
    basics_dict = {}
    # Turn the Dataset into a data frame, using json_normalize so that the Dataset, too, will be a data frame.
    # Note that json_normalize is preferable to from_dict for handling of nested dicts.
    # df = pd.json_normalize(input_dataset)
    df = input_dataset
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
    basics_dict['num_rows'] = data_shape[0]
    basics_dict['num_cols'] = data_shape[1]
    if label_column_name:
        if label_type == "discrete":
            label_value_counts = str(
                df[label_column_name].value_counts()).replace('\n', ', ')
            basics_dict['label_counts'] = label_value_counts
        elif label_type == "real":
            np_array = np.array(df[label_column_name])
            basics_dict["label_min"] = float(round(np_array.min(), 4))
            basics_dict["label_max"] = float(round(np_array.max(), 4))
            basics_dict["label_mean"] = float(round(np_array.mean(), 4))
            basics_dict["label_var"] = float(round(np_array.var(), 4))
        else:
            sys.stderr.write(
                "No label type specified; not calculating label statistics.\n")
    if VERBOSE:
        print('\n* Step 1 summary.')
        print(basics_dict)
    return basics_dict


def rank_words(term_df):
    # Put the right rank for each word. Words with the same count are in the same rank.
    # Save the words in each rank in a dict, used later when plotting it out.
    ranked_words = {}
    unique_counts = pd.unique(term_df['count'])
    unique_ranks = np.arange(1, len(unique_counts) + 1)

    for count, rank in zip(unique_counts, unique_ranks):
        term_df.loc[term_df['count'] == count, 'rank'] = rank
        ranked_words[rank] = ','.join(term_df[term_df['count'] == count].index)
    return term_df, ranked_words


def test_fit(fit):
    print(
        "Checking log likelihood ratio to see if the data is better explained")
    print("by other well-behaved distributions...")
    # The first value returned from distribution_compare is the log likelihood ratio
    better_distro = False
    trunc = fit.distribution_compare('power_law', 'truncated_power_law')
    if trunc[0] < 0:
        print("Seems a truncated power law is a better fit.")
        better_distro = True

    lognormal = fit.distribution_compare('power_law', 'lognormal')
    if lognormal[0] < 0:
        print("Seems a lognormal distribution is a better fit.")
        print("But don't panic -- that happens sometimes with language.")
        better_distro = True

    exponential = fit.distribution_compare('power_law', 'exponential')
    if exponential[0] < 0:
        print("Seems an exponential distribution is a better fit. Panic.")
        better_distro = True

    if not better_distro:
        print("\nSeems your data is best fit by a power law. Celebrate!!")


def do_zipf_pmf(term_df, xmin, alpha):
    # The fit is based on an optimal xmin (minimum rank)
    # Let's use this to make count estimates for the zipf pmf
    pmf_mass = float(sum(pd.unique(term_df[term_df['rank'] > xmin]['count'])))
    # Note that the pmf without the xmin accounting should sum to 1.
    zipf_pmf = np.array([int(round(zipf.pmf(p, alpha) * pmf_mass)) for p in
                         pd.unique(term_df['rank'])])


def fit_data(term_df):
    # Uses the powerlaw package to fit the observed frequencies to a zipfian distribution
    observed_counts = term_df['count'].values
    # 'fit_method' is MLE by default; doesn't seem to change the results in my initial pokings.
    # Also tried discrete_approximation="xmax"
    # Note another method for determining alpha
    # might be defined by (Newman, 2005 for details): alpha = 1 + n * sum(ln( xi / xmin )) ^ -1
    fit = powerlaw.Fit(observed_counts, fit_method="KS", discrete=True)
    # Returns:
    #     pdf_bin_edges = The portion of the data that is within the bin.
    #     observed_pdf = The probability density function (normalized histogram) of the data.
    # This should probably be a pmf, not a pdf. But perhaps using discrete=True above helps.
    # Setting original_data to False uses only the data used for the fit (within xmin and xmax).
    pdf_bin_edges, observed_pdf = fit.pdf(original_data=False)
    # Descending, not ascending
    observed_pdf = np.flip(observed_pdf)
    pdf_bin_edges = np.flip(pdf_bin_edges)
    # This seems to basically be the 'Distribution' class described here: https://pythonhosted.org/powerlaw/#powerlaw.Fit.pdf
    theoretical_distribution = fit.power_law
    # The likelihoods of the observed data from the theoretical distribution.
    predicted_likelihoods = theoretical_distribution.likelihoods
    # The logarithm of the likelihoods of the observed data from the theoretical distribution.
    predicted_log_likelihoods = theoretical_distribution.loglikelihoods
    # The probability density function (normalized histogram) of the theoretical distribution
    predicted_pdf = theoretical_distribution.pdf()
    # Descending, not ascending
    predicted_pdf = np.flip(predicted_pdf)
    # !!!! CRITICAL VALUE FOR ZIPF !!!!
    alpha = theoretical_distribution.alpha
    # The optimal xmin *beyond which* the scaling regime of the power law fits best. (This means exclusive xmin, right?)
    xmin = theoretical_distribution.xmin
    xmax = theoretical_distribution.xmax
    distance = theoretical_distribution.KS()
    print("------------------------------------------------")
    print("Optimal alpha:\t\t%.4f" % alpha)
    print("Optimal first rank:\t%s" % xmin)
    print("Optimal last rank:\t%s" % xmax)
    print("Distance:\t\t%.4f" % distance)
    return alpha, xmin, xmax, distance, observed_pdf, predicted_pdf


# Vocabulary Size
def get_count_vocab(input_dataset, langa_column_name: str,
                    lower=True,
                    language="english", clean_html=False) \
        -> Dict:
    vocab_dict = {}
    vocab = Counter()
    filtered_vocab = Counter()
    lem_vocab = Counter()
    # TODO: A single ID will have multiple duplicated sources if there are multiple translations.
    # So yeah. We have to deal with that.
    # Turn the Dataset into a data frame.
    df = input_dataset  # pd.DataFrame.from_dict(input_dataset)
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
                word.lower() for word in tokenized_text if
                word.lower() not in language_stopwords)
            lem_vocab_tmp = FreqDist(
                wnl.lemmatize(word.lower()) for word in tokenized_text if
                word.lower() not in language_stopwords)
        else:
            vocab_tmp = FreqDist(word for word in tokenized_text)
            filtered_vocab_tmp = FreqDist(word for word in tokenized_text if
                                          word not in language_stopwords)
            lem_vocab_tmp = FreqDist(wnl.lemmatize(
                word for word in tokenized_text if
                word not in language_stopwords))
        vocab.update(vocab_tmp)
        filtered_vocab.update(filtered_vocab_tmp)
        lem_vocab.update(lem_vocab_tmp)
    if VERBOSE:
        print("\n* Step 2 summary.")
        print(
            "There are {0} words including stop words".format(str(len(vocab))))
        print("There are " + str(
            len(filtered_vocab)) + " words after removing stop words")
        print("There are " + str(
            len(lem_vocab)) + " words after removing stop words and lemmatizing")
    vocab_dict['num_words'] = len(vocab)
    vocab_dict['num_filtered_words'] = len(filtered_vocab)
    vocab_dict['num_lemmatized_words'] = len(lem_vocab)
    return vocab_dict


# TODO: Association metrics
# def forward(self, *args, **kwargs):
# Accumulate metric over batches
#    self._npmixy(predicted_labels, protected_variable_labels)

# def do_association_metrics():
#    """ AssociationWithoutGroundTruth measures model biases in the absence of ground truth. It does so by computing
#    one-vs-all or pairwise association gaps using statistical measures like nPMIxy, nPMIy, PMI^2, and PMI, which are
#    capable of capturing labels across a range of marginal frequencies. A gap of nearly 0 implies less bias on the
#    basis of Association the Absence of Ground Truth.
#    """
#    self._npmixy = AssociationWithoutGroundTruth()
#    model = YourModel(...)
#    # Get final values of metric after all batches have been processed
#    print(model._npmixy.get_metric())

# def get_label_stats()
# TODO: Show the top tokens by count for each label.
# Proportion within dataset
# Closed and open pi chart?
# Find the closed word lists for the different languages in order to calculate distributional statistics.

def do_zipf(input_dataset, langa_column_name):
    term_df = count_vocab_frequencies(input_dataset,
                                      langa_column_name=langa_column_name)
    term_df['proportion'] = term_df['count'] / float(sum(term_df['count']))
    rank_column = term_df['count'].rank(method='dense', numeric_only=True,
                                        ascending=False)
    term_df['rank'] = rank_column.astype('int64')
    print("Info on the observed frequencies:")
    print(term_df.info())
    print("------------------------------------")
    print("Vocab size (types):\t%s" % len(term_df))
    print("Vocab size (tokens):\t%s" % sum(term_df['count']))
    print("Observations look like this:")
    print(term_df.head())
    print('...')
    print(term_df.tail())
    alpha, xmin, xmax, distance, observed_pdf, predicted_pdf = fit_data(term_df)
    # Significance testing
    # Note: We may want to use bootstrapping (instead of the standard KS test p-value tables) to determine statistical significance
    # See: https://stats.stackexchange.com/questions/264431/how-to-determine-if-zipfs-law-can-be-applied Answer #4
    print("Checking the goodness of fit of our observed distribution")
    print(" to the hypothesized power law distribution")
    print(" using a Kolmogorov–Smirnov (KS) test.")
    ks_test = ks_2samp(observed_pdf, predicted_pdf)
    # print("KS test:", end='\t\t')
    print(ks_test)
    print("\nThe KS test p-value is: %.4f" % ks_test.pvalue)
    if ks_test.pvalue < .01:
        print(
            "\nYour data fits a powerlaw with a minimum KS distance of %.4f" % distance)
        print("\nWoohoo!")
    else:
        print("\nYour data does not fits a powerlaw. =(")
        print("\nDO BETTER.")
    term_df, ranked_words = rank_words(term_df)
    if not xmax:
        xmax = len(ranked_words)
    zipf_dict = {'alpha': float(alpha), 'xmin': float(xmin), 'xmax': float(
        xmax),
                 'distance': float(distance), 'pvalue': float(ks_test.pvalue)}
    print(zipf_dict)
    return zipf_dict, term_df, ranked_words


def count_vocab_frequencies(term_df, langa_column_name):
    """
    Based on an input pandas DataFrame with a 'text' column,
    this function will count the occurrences of ALL words
    (no stop word removal) and will return another DataFrame
    with the rows corresponding to the different vocabulary words
    and the column to the total count of that word.
    """
    term_df = term_df[langa_column_name]
    cvec = CountVectorizer(token_pattern=u"(?u)\\b\\w+\\b")
    # Needed to modify the minimum token length :
    # https://stackoverflow.com/questions/33260505/countvectorizer-ignoring-i
    cvec.fit(term_df)
    document_matrix = cvec.transform(term_df)
    batches = np.linspace(0, term_df.shape[0], 100).astype(int)
    i = 0
    tf = []
    while i < len(batches) - 1:
        batch_result = np.sum(
            document_matrix[batches[i]:batches[i + 1]].toarray(), axis=0)
        tf.append(batch_result)
        i += 1
    term_freq_df = pd.DataFrame([np.sum(tf, axis=0)],
                                columns=cvec.get_feature_names()).transpose()
    term_freq_df.columns = ['count']
    term_freq_df.index.name = 'word'
    sorted_term_freq_df = pd.DataFrame(
        term_freq_df.sort_values(by='count', ascending=False)['count'])
    return sorted_term_freq_df


"""

def do_zipf(input_dataset, langa_column_name):
    # Turn the Dataset into a data frame.
    df = input_dataset #pd.DataFrame.from_dict(input_dataset)
    term_df = count_vocab_frequencies(df[langa_column_name])
    term_freq_df = count_vocab_frequencies(term_df)
    # Real vs. Projected value according to Zipf; flag those that are a difference higher than x
    # # for what's predicted by the law.
    # For closed class words, what could we say about that means semantically?
    return term_freq_df.to_dict()
"""


# Instance Characteristics
def get_text_stats(input_dataset, langa_column_name: str) -> Dict:
    # Calculates sufficient statistics for text-based instances: average, mean, median
    total_lens = 0
    all_lengths = []
    text_dict = {}
    i = 1
    # Turn the Dataset into a data frame.
    df = input_dataset
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
        print(
            "The average sentence length is: " + str(avg_sent_len) + " words.")
        print("The mean sentence length is: " + str(
            statistics.mean(all_lengths)) + " words.")
        print("The median sentence length is: " + str(
            statistics.median(all_lengths)) + " words.")
    text_dict['mean_sent_len'] = round(statistics.mean(all_lengths), 4)
    text_dict['median_sent_len'] = round(statistics.median(all_lengths), 4)
    return text_dict


def do_dataset(dataset_name: str, config_name: str, dataset_split_name: str,
               label_column_name: str,
               label_type="discrete", langa_column_name="text", lower=True,
               language="english",
               clean_html=False, streaming=False) -> Dict:
    # Dataset object
    loaded_dataset = load_dataset(dataset_name, config_name,
                                  split=dataset_split_name, streaming=streaming)
    # For now, 'streaming' just means we're taking some first 5000 chunks.
    n = 5000
    # TODO: Json normalize for both?
    if streaming:
        data_head = loaded_dataset.take(n)
        df = pd.DataFrame(data_head)
    else:
        df = pd.json_normalize(loaded_dataset)
    # desired_dataset = data_dict[dataset_split_name]
    data_basics_dict = get_data_basics(input_dataset=df,
                                       label_column_name=label_column_name,
                                       label_type=label_type)
    # Want to do this for both *source* and *target*
    data_vocab_dict = get_count_vocab(input_dataset=df,
                                      langa_column_name=langa_column_name,
                                      lower=lower,
                                      language=language, clean_html=clean_html)
    data_text_dict = get_text_stats(input_dataset=df,
                                    langa_column_name=langa_column_name)
    ppl_dict = get_perplexity(dataset_name, config_name, dataset_split_name,
                              langa_column_name)
    zipf_dict, term_df, ranked_words = do_zipf(input_dataset=df,
                                               langa_column_name=langa_column_name)
    # TODO: Run all the rest of the metrics
    output_yaml_data = {"Basic Data Characteristics": data_basics_dict,
                        "Vocab Characteristics": data_vocab_dict,
                        "Text Characteristics": data_text_dict,
                        "Perplexity": ppl_dict, "Zipf distribution": zipf_dict}
    return output_yaml_data


def do_c4_dataset() -> Dict:
    # data= load_dataset("oscar", "unshuffled_deduplicated_en", split = "train", streaming= True)
    #data = load_dataset("c4", "en", split="train", streaming=True)
    #grab_n = 5000
    # For streaming data
    #print('Note: Just taking the first %s instances.' % grab_n)
    #data_head = data.take(grab_n)
    #df = pd.DataFrame(data_head)
    c4_yaml = do_dataset(dataset_name="c4", config_name="en",
                         dataset_split_name="train",
                         langa_column_name="text", streaming=True)
    return c4_yaml


def do_glue_ax_dataset() -> Dict:
    # Dataset: glue-ax
    """ A manually-curated evaluation dataset for fine-grained analysis
    of system performance on a broad range of linguistic phenomena.
    This dataset evaluates sentence understanding through Natural Language Inference (NLI) problems.
    Use a model trained on MulitNLI to produce predictions for this dataset."""
    glue_ax_yaml = do_dataset(dataset_name="glue", config_name="ax",
                              dataset_split_name="test",
                              label_column_name="label", label_type="discrete",
                              langa_column_name="premise")
    return glue_ax_yaml


def do_asset_ratings_dataset() -> Dict:
    # Dataset: Asset-ratings
    asset_ratings_yaml = do_dataset(dataset_name="asset", config_name="ratings",
                                    dataset_split_name="full",
                                    label_column_name="rating",
                                    label_type="real",
                                    langa_column_name="original")
    return asset_ratings_yaml


def do_imdb_train_dataset() -> Dict:
    imdb_yaml = do_dataset(dataset_name="imdb", config_name="plain_text",
                           dataset_split_name="train",
                           label_column_name="label", label_type="discrete",
                           langa_column_name="text",
                           clean_html=True)
    return imdb_yaml


# Large datasets we stream; this requires different handling,
# specifically different read-in functions for a streamed vs fully-read
# TODO: Get dataset size to help solve whether to stream of read in full.
# (could be config, could be automatically grabbed)

def main(args) -> List:
    dataset_name = args.dataset
    config_name = args.config
    dataset_split_name = args.split
    label_column_name = args.label_column
    label_type = args.label_type
    langa_column_name = args.language_column
    clean_html = args.clean_html
    streaming = args.streaming
    # This is going to need to expand as the other options will be necessary to distinguish between, e.g,
    # different splits of the dataset being analyzed.
    yaml_name = dataset_name + "-" + config_name + ".yaml"
    # TODO: Decide whether to handle these as kwargs, or attributes of a Data class, or something else.
    output_yaml = do_dataset(dataset_name=dataset_name, config_name=config_name,
                             dataset_split_name=dataset_split_name,
                             label_column_name=label_column_name,
                             label_type=label_type,
                             langa_column_name=langa_column_name,
                             clean_html=clean_html,
                             streaming=streaming)
    write_yaml(output_yaml, yaml_name)
    # TODO: Dump a json for the lists of ranked words, etc.; this can be used for visualization work.
    return [yaml_name]


if __name__ == "__main__":
    # execute only if run as a script
    if len(sys.argv) == 1:
        output_filenames = []
        print(
            "No arguments specified; running through 3 datasets as an example: Glue, Asset, IMDB")
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
        
# Just stashing it here so I can use a different version
def count_vocab_frequencies(df, cutoff=0):
    \"\"\"
    Based on an input pandas DataFrame with a 'text' column,
    this function will count the occurrences of all words
    with a frequency higher than 'cutoff' and will return another DataFrame
    with the rows corresponding to the different vocabulary words
    and the column to the count count of that word.
    \"\"\"
    # Move this up as a constant in larger code.
    batch_size = 10
    cvec = CountVectorizer(token_pattern=u"(?u)\\b\\w+\\b")
    # Needed to modify the minimum token length:
    # https://stackoverflow.com/questions/33260505/countvectorizer-ignoring-i
    cvec.fit(df.text)
    document_matrix = cvec.transform(df.text)
    batches = np.linspace(0, df.shape[0], batch_size).astype(int)
    i = 0
    tf = []
    while i < len(batches) - 1:
        batch_result = np.sum(document_matrix[batches[i]:batches[i+1]].toarray(), axis=0)
        tf.append(batch_result)
        i += 1
    term_freq_df = pd.DataFrame([np.sum(tf, axis=0)], columns=cvec.get_feature_names()).transpose()
    term_freq_df.columns = ['count']
    term_freq_df.index.name = 'word'
    term_freq_df = term_freq_df[term_freq_df['count'] > cutoff]
    sorted_term_freq_df = pd.DataFrame(term_freq_df.sort_values(by='count', ascending=False)['count'])
    return sorted_term_freq_df
"""
