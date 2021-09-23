import igraph
import math
import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import tokenizers
import transformers
import pandas as pd
import powerlaw
import torch
from sklearn.feature_extraction.text import CountVectorizer
from scipy.stats import hmean, norm, ks_2samp
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

from dataclasses import asdict
from datasets import (
    Dataset,
    get_dataset_infos,
    list_datasets,
    load_dataset,
    load_dataset_builder,
)
from datasets.utils import metadata
from igraph import Graph, EdgeSeq
from os.path import join as pjoin
from sklearn.cluster import AgglomerativeClustering
import nltk
from nltk.corpus import stopwords
from nltk.probability import FreqDist
nltk.download('stopwords')

#Using this simple tokenizer for now, but we can use the HF one everywhere, if that's preferred
from nltk.tokenize import RegexpTokenizer
simple_tokenizer = RegexpTokenizer(r"\w+")

#For language detection (since datasets don't automatically come with a language)
#TODO: get language from the dataset card
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0
from iso639 import languages

st.set_page_config(
    page_title="Demo to showcase dataset metrics",
    page_icon="https://huggingface.co/front/assets/huggingface_logo.svg",
    layout="wide",
    initial_sidebar_state="auto",
)

# colorblind-friendly colors
colors = [
    "#332288",
    "#117733",
    "#882255",
    "#AA4499",
    "#CC6677",
    "#44AA99",
    "#DDCC77",
    "#88CCEE",
]

########## preparation functions

_SAMPLE_SIZE = 5000
_TREE_DEPTH = 10
_TREE_MIN_NODES = 250

@st.cache()
def all_datasets():
    ds_list = list_datasets(with_community_datasets=False, with_details=True)
    ds_name_to_dict = {ds.id: ds.description for ds in ds_list}
    ds_names = list(ds_name_to_dict.keys())
    return (ds_names, ds_name_to_dict)


# list of fiels we want to keep around from the DatasetInfo object
keep_info_fields = [
    "features",
    "config_name",
    "splits",
]



# Recursively get a list of all features of a certain dtype
# the output is a list of tuples > e.g. ('A', 'B', 'C') for feature example['A']['B']['C']
@st.cache(allow_output_mutation=True)
def get_typed_features(features, ftype='string', parents=None):
    if parents is None:
        parents = []
    typed_features = []
    for name, feat in features.items():
        if feat.get("dtype", None) == ftype:
            typed_features += [tuple(parents + [name])]
        elif "feature" in feat:
            if feat["feature"].get("dtype", None) == ftype:
                typed_features += [tuple(parents + [name])]
            elif isinstance(feat["feature"], dict):
                typed_features += get_typed_features(feat["feature"], ftype, parents + [name])
    return typed_features

# Recursively get a list of all features that are ClassLabels
# the outputs are pairs of tuples as above and the list of class names
# TODO: fix for DICT
@st.cache(allow_output_mutation=True)
def get_label_features(features, parents=None):
    if parents is None:
        parents = []
    text_features = []
    for name, feat in features.items():
        if 'num_classes' in feat:
            text_features += [(tuple(parents + [name]), feat["names"])]
        elif "feature" in feat:
            if 'num_classes' in feat:
                text_features += [(tuple(parents + [name]), feat["feature"]["names"])]
            elif isinstance(feat["feature"], dict):
                text_features += get_label_features(feat["feature"], parents + [name])
    return text_features

# Cast info to pure dictionary by casting SplitInfo and pre-selecting features
@st.cache(allow_output_mutation=True)
def dictionarize_info(dset_info):
    info_dict = asdict(dset_info)
    res = {}
    res["config_name"] = info_dict["config_name"]
    res["splits"] = {
        spl: spl_info["num_examples"]
        for spl, spl_info in info_dict["splits"].items()
    }
    res["features"] = {
        "string": get_typed_features(info_dict["features"], 'string'),
        "int32": get_typed_features(info_dict["features"], 'int32'),
        "float32": get_typed_features(info_dict["features"], 'float32'),
        "label": get_label_features(info_dict["features"]),
    }
    return res

# All together now!
@st.cache(allow_output_mutation=True)
def get_config_infos_dict(name):
    return {
        conf_name: dictionarize_info(conf_info)
        for conf_name, conf_info in get_dataset_infos(name).items()
    }

####### prepare text and scores
# extract text to analyze:
# list of all examples of a given feature in a given split of a given config of a given dataset
# returns a dataset of strings
@st.cache(allow_output_mutation=True)
def extract_text(examples, text_path):
    text_list = []
    example_ct = 0
    item_list = examples[text_path[0]]
    for field_name in text_path[1:]:
        item_list = [
            next_item
            for item in item_list
            for next_item in (item[field_name] if isinstance(item[field_name], list) else [item[field_name]])
        ]
    text_list += [
        text
        for item in item_list
        for text in (item if isinstance(item, list) else [item])
    ]
    return {"text": text_list}

@st.cache(allow_output_mutation=True)
def get_text_to_analyze(name, text_path, config, split=None, max_items=20000, streaming=False):
    ### default arguments
    if split is None:
        split = 'train' if 'train' in config["splits"] else list(config["splits"])[0]
        print(f"using default split: {split}")
    ### get text from dataset
    print(f"running -- load_dataset({name}, {config['config_name']}, streaming={streaming})")
    dataset = load_dataset(name, config["config_name"], streaming=streaming)
    data_split = dataset[split].select(range(max_items))
    dataset_text = data_split.map(
        lambda examples: extract_text(examples, text_path),
        batched=True,
        remove_columns=data_split.column_names,
    )
    return dataset_text

def get_labels(name, text_path, config, split=None, max_items=20000, streaming=False):
    ### default arguments
    if split is None:
        split = 'train' if 'train' in config["splits"] else list(config["splits"])[0]
        print(f"using default split: {split}")
    ### get text from dataset
    dataset = load_dataset(name, config["config_name"], streaming=streaming)
    data_split = dataset[split].select(range(max_items))
    #TODO: find other ways of finding labels? other names?
    try:
        dataset_labels = [(data_split.info.features['label'].names[k],v) for k, v in Counter(data_split['label']).items()]
    except:
        dataset_labels = [(data_split.info.features['class'].names[k],v) for k, v in Counter(data_split['class']).items()]
    return dataset_labels

#Quick and dirty way of getting language from datacard if possible, otherwise using langdetect on the first sentence in the dataset
def get_language(name, text):
    try:
        metadata= utils.metadata.DatasetMetadata.from_readme("./datasets/"+name+"/README.md")
        langs = metadata['languages']
        if len(langs) > 1:
            return ("Languages detected: " + [languages.get(alpha2=l).name.lower() for l in langs])
        else:
            return(languages.get(alpha2=languages[0]).name.lower())
    except:
        lang= detect(text)
        return(languages.get(alpha2=lang).name.lower())

#Counting vocabulary from the text

def get_count_vocab(datatext, language):
    language_stopwords = stopwords.words(language)
    vocab_dict = {}
    vocab = Counter()
    for sent in datatext['text']:
        tokenized_text = simple_tokenizer.tokenize(sent)
        vocab_tmp = FreqDist(word for word in tokenized_text if word.lower() not in language_stopwords)
        vocab.update(vocab_tmp)
    return(vocab)

#Checking for NaNs

def get_nans(name, text_path, config, split=None, max_items=20000, streaming=False):
    if split is None:
        split = 'train' if 'train' in config["splits"] else list(config["splits"])[0]
        print(f"using default split: {split}")
    ### get text from dataset
    dataset = load_dataset(name, config["config_name"], streaming=streaming)
    data_split = dataset[split].select(range(max_items))
    # TODO: figure out how to do this without converting to DataFrame
    datadf = pd.DataFrame(data_split)
    nans= datadf.isnull().sum().sum()
    return nans


def dedup_count(name, text_path, config, split=None, max_items=20000, streaming=False):
    dataset = load_dataset(name, config["config_name"], streaming=streaming)
    data_split = dataset[split].select(range(max_items))
    # TODO: figure out how to do this without converting to DataFrame
    datadf = pd.DataFrame(data_split)
    dict_count = {}
    count=0
    total=0
    for t in datadf[text_path[0]].tolist():
        if len(t) > 0:
            count+=1
            try:
                dict_count[str(t)] += 1
                #print("Duplicate sentence %s at index %d " % (t, datadf.index[count]))
                total +=1
            except KeyError:
                dict_count[str(t)] = 1
        else:
            continue
    return(total)

def dedup_print(name, text_path, config, split=None, max_items=20000, streaming=False):
    dataset = load_dataset(name, config["config_name"], streaming=streaming)
    data_split = dataset[split].select(range(max_items))
    # TODO: figure out how to do this without converting to DataFrame
    datadf = pd.DataFrame(data_split)
    dict_count = {}
    count=0
    total=0
    duplicatelist=[]
    for t in datadf[text_path[0]].tolist():
        if len(t) > 0:
            count+=1
            try:
                dict_count[str(t)] += 1
                duplicatelist.append(t)
                total +=1
            except KeyError:
                dict_count[str(t)] = 1
        else:
            continue
    return(duplicatelist)


########## metrics code
@st.cache(allow_output_mutation=True, hash_funcs={Dataset: lambda _: None})
def run_tok_length_analysis(text_dset, cache_name):
    text_dset_lengths = text_dset.map(
        lambda exple: {"space_tok_length": len(exple["text"].split())},
        load_from_cache_file=True,
        cache_file_name=pjoin("cache_dir", f"{cache_name}_space_tok_length"),
    )
    hist_data_tok_length = [text_dset_lengths["space_tok_length"]]
    fig_tok_length = ff.create_distplot(hist_data_tok_length, group_labels=["text lengths"])
    sorted_sents_lengths = [
        (l, s) for s, l in sorted(
            [(sentence["text"], sentence["space_tok_length"]) for sentence in text_dset_lengths],
            key=lambda x:x[1], reverse=True,
        )
    ]
    return (sorted_sents_lengths, fig_tok_length)

device = "cuda:0"
@st.cache(allow_output_mutation=True, hash_funcs={
    Dataset: lambda _: None,
    transformers.models.xlnet.tokenization_xlnet_fast.XLNetTokenizerFast: lambda _: None,
    transformers.models.xlnet.modeling_xlnet.XLNetLMHeadModel: lambda _: None,
})
def run_perplexity_analysis(text_dset, cache_name):
    tokenizer = transformers.AutoTokenizer.from_pretrained("xlnet-base-cased")
    model = transformers.AutoModelForCausalLM.from_pretrained("xlnet-base-cased").to(device)
    def get_single_sent_loss(sent):
        batch = tokenizer(sent, return_tensors="pt", padding=True)
        batch['labels'] = batch['input_ids']
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            loss = model(**batch).loss.item()
            return loss
    text_dset_loss = text_dset.map(
        lambda exple: {"xlnet_loss": get_single_sent_loss(exple["text"])},
        load_from_cache_file=True,
        cache_file_name=pjoin("cache_dir", f"{cache_name}_xlnet_loss"),
    )
    hist_data_loss = [text_dset_loss["xlnet_loss"]]
    fig_tok_loss = ff.create_distplot(hist_data_loss, group_labels=["text perplexities"])
    sorted_sents_loss = [
        (l, s) for s, l in sorted(
            [(sentence["text"], sentence["xlnet_loss"]) for sentence in text_dset_loss],
            key=lambda x:x[1], reverse=True,
        )
    ]
    return (sorted_sents_loss, fig_tok_loss)

@st.cache(allow_output_mutation=True, hash_funcs={
    Dataset: lambda _: None,
    transformers.models.mpnet.tokenization_mpnet_fast.MPNetTokenizerFast: lambda _: None,
    transformers.models.mpnet.modeling_mpnet.MPNetModel: lambda _: None,
})
def run_hierarchical_clustering(text_dset, cache_name):
    # First step: pre-compute all embeddings
    s_tokenizer = transformers.AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
    s_model = transformers.AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2').to(device)
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    def embed_sentences(sentences):
        sents = sentences["text"]
        batch = s_tokenizer(sents, padding=True, truncation=True, return_tensors='pt')
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            model_output = s_model(**batch)
            sentence_embeds = mean_pooling(model_output, batch['attention_mask'])
            sentence_embeds /= sentence_embeds.norm(dim=-1, keepdim=True)
            return {"embed": [embed.tolist() for embed in sentence_embeds]}
    text_dset_embeds = text_dset.map(
        embed_sentences,
        batched=True,
        batch_size=32,
        load_from_cache_file=True,
        cache_file_name=pjoin("cache_dir", f"{cache_name}_space_embeds"),
    )
    # Second step: on to the clustering
    np_embeds = np.array(text_dset_embeds["embed"])
    clustering_model = AgglomerativeClustering(n_clusters=None, affinity='cosine', linkage='average', distance_threshold=0.)
    clustering_model.fit(np_embeds)
    merged = clustering_model.children_
    in_merge = [nid for mg in merged for nid in mg]
    assert len(in_merge) == len(set(in_merge))
    # make actual tree from merges
    nodes = [{
        "nid": i,
        "parent": -1,
        "child_left": -1,
        "child_right": -1,
        "sent_id_list": [i],
        "weight": 1,
        "depth": 0,
    } for i in range(text_dset.num_rows)] + [{
        "nid": text_dset.num_rows + i,
        "parent": -1,
        "child_left": -1,
        "child_right": -1,
        "sent_id_list": [],
        "weight": 0,
        "depth": 0,
    } for i in range(text_dset.num_rows-1)]
    for inid, (c_a, c_b) in enumerate(merged):
        nid = inid + text_dset.num_rows
        nodes[nid]["child_left"] = int(c_a)
        nodes[nid]["child_right"] = int(c_b)
        nodes[c_a]["parent"] = nid
        nodes[c_b]["parent"] = nid
        nodes[nid]["depth"] = max(nodes[nid]["depth"], nodes[c_a]["depth"] + 1, nodes[c_b]["depth"] + 1)
        nodes[nid]["weight"] = nodes[c_a]["weight"] + nodes[c_b]["weight"]
    # restrict the depth
    tree_depth = max([node["depth"] for node in nodes])
    root = nodes[[node["depth"] for node in nodes].index(tree_depth)]
    def compute_rec_depth(node, current_depth):
        node["depth"] = current_depth
        if node["child_left"] != -1:
            compute_rec_depth(nodes[node["child_left"]], current_depth+1)
        if node["child_right"] != -1:
            compute_rec_depth(nodes[node["child_right"]], current_depth+1)
    compute_rec_depth(root, 0)
    def aggregate_children_sentences(node):
        if node["child_left"] != -1 and node["child_right"] != -1:
            assert nodes[node["child_left"]]["parent"] == node["nid"], f"C {node} \n -- {nodes[node['child_left']]} \n -- {nodes[node['child_right']]}"
            assert nodes[node["child_right"]]["parent"] == node["nid"], f"D {node} \n -- {nodes[node['child_left']]} \n -- {nodes[node['child_right']]}"
            aggregate_children_sentences(nodes[node["child_left"]])
            aggregate_children_sentences(nodes[node["child_right"]])
            node["sent_id_list"] = nodes[node["child_left"]]["sent_id_list"][:] + nodes[node["child_right"]]["sent_id_list"][:]
            assert node["weight"] == len(node["sent_id_list"]), f"{node} \n -- {nodes[node['child_left']]} \n -- {nodes[node['child_right']]}"
    cutoff_depth = _TREE_DEPTH
    cutoff_nodes = _TREE_MIN_NODES
    for node in nodes:
        if node["depth"] == cutoff_depth or (
            (0 < node["depth"] and node["depth"] < cutoff_depth) and \
            (node["weight"] < cutoff_nodes and nodes[node["parent"]]["weight"] >= cutoff_nodes)
        ):
            aggregate_children_sentences(node)
    top_nodes = [
        node
        for nid, node in enumerate(nodes) if node["depth"] <= cutoff_depth and (
            node["weight"] >= cutoff_nodes or \
            nodes[node["parent"]]["weight"] >= cutoff_nodes
        )
    ]
    top_nodes.reverse()
    id_map = dict([(node["nid"], i) for i, node in enumerate(top_nodes)])
    id_map[-1] = -1
    for node in top_nodes:
        node["orig_id"] = node["nid"]
        for k in ["nid", "parent", "child_left", "child_right"]:
            node[k] = id_map.get(node[k], -1)
    # TODO node embeddings and leaf histograms
    leaves = [node for node in top_nodes if node["child_left"] == -1 and len(node["sent_id_list"]) > 2]
    for leaf in leaves:
        assert len(leaf["sent_id_list"]) > 0, f"{leaf}"
        embeddings = torch.Tensor([text_dset_embeds[sid]["embed"] for sid in leaf["sent_id_list"]])
        centroid = embeddings.mean(dim=0, keepdim=True).norm(dim=-1)
        distances = (embeddings * centroid).sum(dim=-1) # using dot product
        distance_list = distances.tolist()
        # For now: distance from centroid
        hist_data_excenter = [distance_list]
        fig_tok_excenter = ff.create_distplot(hist_data_excenter, group_labels=["embeding distance from centroid"])
        sorted_sents_excenter = [
            (d, text_dset_embeds["text"][sid]) for sid, d in sorted(
                zip(leaf["sent_id_list"], distance_list),
                key=lambda x:x[1], reverse=True,
            )
        ]
        leaf["figure"] = fig_tok_excenter
        leaf["sorted"] = sorted_sents_excenter
    return top_nodes, leaves

# copied code from https://plotly.com/python/tree-plots/
@st.cache(allow_output_mutation=True)
def make_tree_plot(node_list):
    # make plot nodes
    labels = [f"{nid:2d} - {node['weight']:5d} sents" for nid, node in enumerate(node_list)]
    root = node_list[0]
    root["X"] = 0
    root["Y"] = 0
    def rec_make_coordinates(node):
        if node["child_left"] != -1:
            child_l = node_list[node["child_left"]]
            child_r = node_list[node["child_right"]]
            child_l["X"] = node["X"]
            child_l["Y"] = node["Y"] - 1
            child_r["X"] = node["X"] + child_l["weight"] * 10 / root["weight"]
            child_r["Y"] = node["Y"] - 1
            rec_make_coordinates(child_l)
            rec_make_coordinates(child_r)
    rec_make_coordinates(root)
    E = [] # list of edges
    Xn = []
    Yn = []
    Xe = []
    Ye = []
    for nid, node in enumerate(node_list):
        Xn += [node["X"]]
        Yn += [node["Y"]]
        c_a, c_b = (node["child_left"], node["child_right"])
        if c_a != -1:
            E += [(nid, c_a)]
            Xe += [node["X"], node_list[c_a]["X"], None]
            Ye += [node["Y"], node_list[c_a]["Y"], None]
        if c_b != -1:
            E += [(nid, c_b)]
            Xe += [node["X"], node_list[c_b]["X"], None]
            Ye += [node["Y"], node_list[c_b]["Y"], None]
    # make figure
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=Xe, y=Ye, mode='lines', line=dict(color='rgb(210,210,210)', width=1), hoverinfo='none')
    )
    fig.add_trace(
        go.Scatter(x=Xn, y=Yn, mode='markers', name='nodes',
                marker=dict(
                symbol='circle-dot',
                size=18,
                color='#6175c1',    #'#DB4551',
                line=dict(color='rgb(50,50,50)', width=1)
            ),
            text=labels, hoverinfo='text', opacity=0.8
        )
    )
    return fig

# vocab counting code
def count_vocab_frequencies(df, cutoff=3):
    """
    Based on an input pandas DataFrame with a 'text' column,
    this function will count the occurrences of all words
    with a frequency higher than 'cutoff' and will return another DataFrame
    with the rows corresponding to the different vocabulary words
    and the column to the total count of that word.
    """
    # Move this up as a constant in larger code.
    batch_size = 10
    cvec = CountVectorizer(token_pattern=u"(?u)\\b\\w+\\b")
    # Needed to modify the minimum token length:
    # https://stackoverflow.com/questions/33260505/countvectorizer-ignoring-i
    cvec.fit(df['text'])
    document_matrix = cvec.transform(df['text'])
    batches = np.linspace(0, df.shape[0], batch_size).astype(int)
    i = 0
    tf = []
    while i < len(batches) - 1:
        batch_result = np.sum(document_matrix[batches[i]:batches[i+1]].toarray(), axis=0)
        tf.append(batch_result)
        i += 1
    term_freq_df = pd.DataFrame([np.sum(tf, axis=0)], columns=cvec.get_feature_names()).transpose()
    term_freq_df.columns = ['total']
    term_freq_df = term_freq_df[term_freq_df['total'] > cutoff]
    sorted_term_freq_df = pd.DataFrame(term_freq_df.sort_values(by='total')['total'])
    return sorted_term_freq_df


# Uses the powerlaw package to fit the observed frequencies to a zipfian distribution
def fit_Zipf(term_df):
    observed_counts = np.flip(term_df['total'].values)
    norm = float(sum(observed_counts))
    fit = powerlaw.Fit(observed_counts, fit_method="KS", discrete=True)
    bin_edges, log_observed_probabilities = fit.pdf(original_data=True)
    log_predicted_probabilities = fit.power_law.pdf()
    alpha = fit.power_law.alpha
    xmin = fit.power_law.xmin
    distance = fit.power_law.KS()
    observed_probabilities = np.flip(observed_counts/norm)
    predicted_probabilities = np.flip(bin_edges/norm)
    st.markdown("The optimal alpha is :\t\t%.4f" % alpha)
    #st.markdown("Optimal Frequency cut-off:\t%s" % xmin)
    #st.markdown("Distance:\t\t%.4f" % distance)
    # Significance testing
    # Note: We may want to use bootstrapping (instead of the standard KS test p-value tables) to determine statistical significance
    # See: https://stats.stackexchange.com/questions/264431/how-to-determine-if-zipfs-law-can-be-applied Answer #4
    ks_test = ks_2samp(observed_probabilities, predicted_probabilities)
    # print("KS test:", end='\t\t')
    #st.markdown(ks_test)
    #st.markdown("\nThe KS test p-value is: %.4f" % ks_test.pvalue)
    if ks_test.pvalue < .01:
        st.markdown("\n Great news! Your data fits a powerlaw with a minimum KS distance of %.4f" % distance)
    else:
        st.markdown("\n Sadly, your data does not fit a powerlaw. =\(")
    predicted_per_rank = defaultdict(list)
    j = 0
    # For each rank in the observed_counts
    for i in range(len(observed_counts)):
        observed_count = observed_counts[i]
        rank = i+1
        # while the predicted count is higher than the observed count,
        # set its rank to the observed rank
        if j < len(bin_edges):
            while np.flip(bin_edges)[j] >= observed_count:
                bin_rank = rank
                j +=1
                predicted_per_rank[i] += [np.flip(bin_edges)[j-1]]
                if (j>=len(bin_edges)):
                    break

    predicted_x_axis = []
    predicted_y_axis = []
    for i, j in sorted(predicted_per_rank.items()):
        predicted_x_axis += [i]
        predicted_y_axis += [sum(j)/len(j)]
    # Graph it out.
    fig = plt.figure(figsize=(20,15))
    # The pdf of the observed data.
    # The continuous line is superfluous and confusing I think (?) Hacky removing.
    fit.plot_pdf(color='r', linewidth=0, linestyle=':', marker='o')
    # The pdf of the best fit powerlaw
    fit.power_law.plot_pdf(color='b', linestyle='--', linewidth=2)
    fig.suptitle('Log-log plot of word frequency (y-axis) vs. rank (x-axis) \nObserved = red dots\n Power law = blue lines', fontsize=20)
    plt.ylabel('Log frequency', fontsize=18)
    plt.xlabel('Log rank (binned frequencies).\nGaps signify there weren\'t observed words in the fitted frequency bin.', fontsize=16)
    return(fig)

########## streamlit code

ds_names, ds_name_to_dict = all_datasets()

st.title("Data Analysis Tools")

description = """
This demo showcases the [dataset metrics as we develop them](https://github.com/huggingface/datasets-tool-metrics).
Right now this has:
- dynamic loading of datasets in the lib
- fetching config and info without downloading the dataset
- propose the list of candidate text and label features to select
Still working on:
- implementing all the current tools
"""
st.sidebar.markdown(description, unsafe_allow_html=True)

with st.sidebar.expander("Choose first dataset and field"):
    # choose a dataset to analyze
    ds_name_a = st.selectbox(
        "Choose a first dataset to explore:",
        ds_names,
        index=ds_names.index("hate_speech18"),
    )
    # choose a config to analyze
    ds_configs_a = get_config_infos_dict(ds_name_a)
    config_names_a = list(ds_configs_a.keys())
    config_name_a = st.selectbox(
        "Choose a first configuration:",
        config_names_a,
        index=0,
    )
    # choose a subset of num_examples
    ds_config_a = ds_configs_a[config_name_a]
    text_features_a = ds_config_a["features"]["string"]
    text_feature_a = st.selectbox(
        "Which text feature from the first dataset would you like to analyze?",
        text_features_a,
        index=max([i for i, tp in enumerate(text_features_a) if tp[0] != 'id']),
    )
    # choose a split and dataset size
    split_a = st.selectbox(
        "Which split from the first dataset would you like to analyze?",
        ds_config_a["splits"].keys(),
        index=0,
    )
    analyze_sample_a = st.checkbox(
        f"Only analyze the first {_SAMPLE_SIZE} for the first dataset",
        value=True,
    )
    num_examples_a = _SAMPLE_SIZE if analyze_sample_a else ds_config_a["splits"][split_a]
    streaming_a = st.checkbox(
        "Use streaming functionality for the first dataset",
        value=False,
    )
    analyses_a = st.multiselect(
        "which analyses do you want to run for the first dataset?",
        ["sentence lengths", "sentence perplexities", "distance to centroid"],
        ["sentence lengths"],
    )

with st.sidebar.expander("Choose second dataset and field"):
    # choose a dataset to analyze
    ds_name_b = st.selectbox(
        "Choose a second dataset to explore:",
        ds_names,
        index=ds_names.index("hate_speech_offensive"),
    )
    # choose a config to analyze
    ds_configs_b = get_config_infos_dict(ds_name_b)
    config_names_b = list(ds_configs_b.keys())
    config_name_b = st.selectbox(
        "Choose a second configuration:",
        config_names_b,
        index=0,
    )
    # choose a subset of num_examples
    ds_config_b = ds_configs_b[config_name_b]
    text_features_b = ds_config_b["features"]["string"]
    text_feature_b = st.selectbox(
        "Which text feature from the second dataset would you like to analyze?",
        text_features_b,
        index=max([i for i, tp in enumerate(text_features_b) if tp[0] != 'id']),
    )
    # choose a split and dataset size
    split_b = st.selectbox(
        "Which split from the second dataset would you like to analyze?",
        ds_config_b["splits"].keys(),
        index=0,
    )
    analyze_sample_b = st.checkbox(
        f"Only analyze the first {_SAMPLE_SIZE} for the second dataset",
        value=True,
    )
    num_examples_b = _SAMPLE_SIZE if analyze_sample_b else ds_config_b["splits"][split_b]
    streaming_b = st.checkbox(
        "Use streaming functionality for the second dataset",
        value=False,
    )
    analyses_b = st.multiselect(
        "which analyses do you want to run for the second dataset?",
        ["sentence lengths", "sentence perplexities", "distance to centroid"],
        ["sentence lengths"],
    )


# doing some of the caching manually
cache_name_a = f"{ds_name_a}_{config_name_a}_{split_a}_{'-'.join(text_feature_a)}_{num_examples_a}"
cache_name_b = f"{ds_name_b}_{config_name_b}_{split_b}_{'-'.join(text_feature_b)}_{num_examples_b}"

# Grab the text requested
text_dset_a = get_text_to_analyze(
    ds_name_a, text_feature_a, ds_config_a,
    split=split_a, max_items=num_examples_a, streaming=streaming_a
)
text_dset_b = get_text_to_analyze(
    ds_name_b, text_feature_b, ds_config_b,
    split=split_b, max_items=num_examples_b, streaming=streaming_b
)

# Grab the label Distribution


######## Main window

left_col, right_col = st.columns(2)

left_col.markdown(f"### Showing {ds_name_a} - {config_name_a} - {text_feature_a}")
with left_col.expander("Dataset Description A"):
    st.markdown(ds_name_to_dict[ds_name_a])

right_col.markdown(f"### Showing {ds_name_b} - {config_name_b} - {text_feature_b}")
with right_col.expander("Dataset Description B"):
    st.markdown(ds_name_to_dict[ds_name_b])

### Calculate the vocab size
with left_col.expander("Dataset A - General Text Statistics"):
    language_a = get_language(ds_name_a,text_dset_a[0]['text'])
    vocab_a = get_count_vocab(text_dset_a,language_a)
    common_a = vocab_a.most_common(10)
    nancount_a= get_nans(
                ds_name_a, text_feature_a, ds_config_a,
                split=split_a, max_items=num_examples_a, streaming=streaming_a
            )
    dedup_a= dedup_count(
                ds_name_a, text_feature_a, ds_config_a,
                split=split_a, max_items=num_examples_a, streaming=streaming_a
            )
    st.markdown("The language detected is: " + language_a.capitalize())
    st.markdown("There are {0} words after removing stop words".format(str(len(vocab_a))))
    st.markdown("The most common words and their counts are: "+ ', '.join((map(str, common_a))))
    st.markdown("There are {0} missing values in the dataset.".format(str(nancount_a)))
    st.markdown("There are {0} duplicate items in the dataset. For more information about the duplicates, click the 'Duplicates' tab below.".format(str(dedup_a)))

with right_col.expander("Dataset B - General Text Statistics"):
    language_b = get_language(ds_name_b,text_dset_b[0]['text'])
    vocab_b = get_count_vocab(text_dset_b,language_b)
    common_b = vocab_b.most_common(10)
    nancount_b= get_nans(
                ds_name_b, text_feature_b, ds_config_b,
                split=split_b, max_items=num_examples_b, streaming=streaming_b
            )
    dedup_b= dedup_count(
                ds_name_b, text_feature_b, ds_config_b,
                split=split_b, max_items=num_examples_b, streaming=streaming_b
            )

    st.markdown("The language detected is: " + language_b.capitalize())
    st.markdown("There are {0} words after removing stop words".format(str(len(vocab_b))))
    st.markdown("The most common words and their counts are: "+ ', '.join((map(str, common_b))))
    st.markdown("There are {0} missing values in the dataset.".format(str(nancount_b)))
    st.markdown("There are {0} duplicate items in the dataset. For more information about the duplicates, click the 'Duplicates' tab below.".format(str(dedup_b)))

### Show the label distribution from the datasets
with left_col.expander("Dataset A - Label Distribution"):
    try:
        labs_a=get_labels(
            ds_name_a, text_feature_a, ds_config_a,
            split=split_a, max_items=num_examples_a, streaming=streaming_a
        )
        labnames_a = [l[0] for l in labs_a]
        labcounts_a = [l[1] for l in labs_a]
        fig_label_a= px.pie(labcounts_a, values=labcounts_a, names=labnames_a)
        #fig_label_a.update_layout(margin=dict(l=10, r=10, b=10, t=10))
        st.markdown("There are {0} labels in this dataset, with the following distribution: ".format(str(len(labnames_a))))
        fig_label_a.update_traces(hoverinfo='label+percent', textinfo='percent')
        st.plotly_chart(fig_label_a, use_container_width=True)
        #st.markdown("The distribution of labels is the following: " + str(labs_a))
    except KeyError as e:
        st.markdown("No labels were found in the dataset")

### Show the label distribution from the dataset
with right_col.expander("Dataset B - Label Distribution"):
    try:
        labs_b= get_labels(
            ds_name_b, text_feature_b, ds_config_b,
            split=split_b, max_items=num_examples_b, streaming=streaming_b
        )
        labnames_b = [l[0] for l in labs_b]
        labcounts_b = [l[1] for l in labs_b]
        fig_label_b= px.pie(labcounts_b, values=labcounts_b, names=labnames_b)
        fig_label_b.update_traces(hoverinfo='label+percent', textinfo='percent')
        st.markdown("There are {0} labels in this dataset, with the following distribution: ".format(str(len(labnames_b))))
        st.plotly_chart(fig_label_b, use_container_width=True)
    except KeyError as e:
        st.markdown("No labels were found in the dataset")

### First, show the distribution of text lengths
with left_col.expander("Show text lengths A", expanded=False):
    if "sentence lengths" in analyses_a:
        st.markdown("### Text lengths A")
        sorted_sents_lengths_a, fig_tok_length_a = run_tok_length_analysis(text_dset_a, cache_name_a)
        st.plotly_chart(fig_tok_length_a, use_container_width=True)
        start_id_show_lengths_a = st.slider(
            'Show longest sentences in A starting at index:',
            0, text_dset_a.num_rows - 5, value=0, step=5
        )
        for ln, sent in sorted_sents_lengths_a[start_id_show_lengths_a:start_id_show_lengths_a+5]:
            st.text(f"{ln} | {sent}")
    else:
        st.write("To show the lengths of examples, select `sentence lengths` in the list of analyses box left")

with right_col.expander("Show text lengths B", expanded=False):
    if "sentence lengths" in analyses_b:
        st.markdown("### Text lengths B")
        sorted_sents_lengths_b, fig_tok_length_b = run_tok_length_analysis(text_dset_b, cache_name_b)
        st.plotly_chart(fig_tok_length_b, use_container_width=True)
        start_id_show_lengths_b = st.slider(
            'Show longest sentences in B starting at index:',
            0, text_dset_b.num_rows - 5, value=0, step=5
        )
        for ln, sent in sorted_sents_lengths_b[start_id_show_lengths_b:start_id_show_lengths_b+5]:
            st.text(f"{ln} | {sent}")
    else:
        st.write("To show the lengths of examples, select `sentence lengths` in the list of analyses box left")

### Second, show the distribution of text perplexities
with left_col.expander("Show text perplexities A", expanded=True):
    if "sentence perplexities" in analyses_a:
        st.markdown("### Text perplexities A")
        sorted_sents_loss_a, fig_loss_a = run_perplexity_analysis(text_dset_a, cache_name_a)
        st.plotly_chart(fig_loss_a, use_container_width=True)
        start_id_show_loss_a = st.slider(
            'Show highest perplexity sentences in A starting at index:',
            0, text_dset_a.num_rows - 5, value=0, step=5
        )
        for lss, sent in sorted_sents_loss_a[start_id_show_loss_a:start_id_show_loss_a+5]:
            st.text(f"{lss:.3f} {sent}")
    else:
        st.write("To show perplexity of examples, select `sentence perplexities` in the list of analyses box left")

with right_col.expander("Show text perplexities B", expanded=True):
    if "sentence perplexities" in analyses_b:
        st.markdown("### Text perplexities B")
        sorted_sents_loss_b, fig_loss_b = run_perplexity_analysis(text_dset_b, cache_name_b)
        st.plotly_chart(fig_loss_b, use_container_width=True)
        start_id_show_loss_b = st.slider(
            'Show highest perplexity sentences in B starting at index:',
            0, text_dset_b.num_rows - 5, value=0, step=5
        )
        for lss, sent in sorted_sents_loss_b[start_id_show_loss_b:start_id_show_loss_b+5]:
            st.text(f"{lss:.3f} {sent}")
    else:
        st.write("To show perplexity of examples, select `sentence perplexities` in the list of analyses box left")

### Third, use a sentence embedding model
with left_col.expander("Show text embedding outliers A", expanded=True):
    if "distance to centroid" in analyses_a:
        st.markdown("### Text embedding A")
        node_list_a, leaf_list_a = run_hierarchical_clustering(text_dset_a, cache_name_a)
        leaf_ids_a = [leaf["nid"] for leaf in leaf_list_a]
        fig_tree_a = make_tree_plot(node_list_a)
        st.plotly_chart(fig_tree_a, use_container_width=True)
        show_leaf_a = st.selectbox(
            "Choose a leaf node to explore in the first dataset:",
            leaf_ids_a,
            index=0,
        )
        figure_leaf_a = leaf_list_a[leaf_ids_a.index(show_leaf_a)]["figure"]
        sorted_leaf_a = leaf_list_a[leaf_ids_a.index(show_leaf_a)]["sorted"]
        st.plotly_chart(figure_leaf_a, use_container_width=True)
        start_id_show_leaf_a = st.slider(
            'Show closest sentences in leaf to the centroid in A starting at index:',
            0, len(sorted_leaf_a) - 5, value=0, step=5
        )
        for lss, sent in sorted_leaf_a[start_id_show_leaf_a:start_id_show_leaf_a+5]:
            st.text(f"{lss:.3f} {sent}")
    else:
        st.write("To show example distances from the centroid, select `distance to centroid` in the list of analyses box left")

with right_col.expander("Show text embedding outliers B", expanded=True):
    if "distance to centroid" in analyses_b:
        st.markdown("### Text embedding B")
        node_list_b, leaf_list_b = run_hierarchical_clustering(text_dset_b, cache_name_b)
        leaf_ids_b = [leaf["nid"] for leaf in leaf_list_b]
        fig_tree_b = make_tree_plot(node_list_b)
        st.plotly_chart(fig_tree_b, use_container_width=True)
        show_leaf_b = st.selectbox(
            "Choose a leaf node to explore in the second dataset:",
            leaf_ids_b,
            index=0,
        )
        figure_leaf_b = leaf_list_b[leaf_ids_b.index(show_leaf_b)]["figure"]
        sorted_leaf_b = leaf_list_b[leaf_ids_b.index(show_leaf_b)]["sorted"]
        st.plotly_chart(figure_leaf_b, use_container_width=True)
        start_id_show_leaf_b = st.slider(
            'Show closest sentences in leaf to the centroid in B starting at index:',
            0, len(sorted_leaf_b) - 5, value=0, step=5
        )
        for lss, sent in sorted_leaf_b[start_id_show_leaf_b:start_id_show_leaf_b+5]:
            st.text(f"{lss:.3f} {sent}")
    else:
        st.write("To show example distances from the centroid, select `distance to centroid` in the list of analyses box left")

### Fourth, show Zipf stuff
with left_col.expander("Show Zipf's Law fit for Dataset A", expanded=False):
    term_freq_df_a= count_vocab_frequencies(text_dset_a)
    st.markdown("Checking the goodness of fit of our observed distribution to the hypothesized power law distribution using a Kolmogorov–Smirnov (KS) test.")
    st.pyplot(fit_Zipf(term_freq_df_a), use_container_width=True)

with right_col.expander("Show Zipf's Law Fit for Dataset B", expanded=False):
    term_freq_df_b = count_vocab_frequencies(text_dset_b)
    st.markdown("Checking the goodness of fit of our observed distribution to the hypothesized power law distribution using a Kolmogorov–Smirnov (KS) test.")
    st.pyplot(fit_Zipf(term_freq_df_b), use_container_width=True)


### Then, show duplicates
with left_col.expander("Show Duplicates from Dataset A", expanded=False):
    st.write("### Here is a list of all the duplicated items:")
    dedup_list_a= dedup_print(
                ds_name_a, text_feature_a, ds_config_a,
                split=split_a, max_items=num_examples_a, streaming=streaming_a
            )
    if dedup_list_a == 0:
        st.write("There are no duplicates in this dataset!")
    else:
        for l in dedup_list_a:
            st.write(l)

with right_col.expander("Show Duplicates from Dataset B", expanded=False):
    st.write("### Here is a list of all the duplicated items:")
    dedup_list_b= dedup_print(
                ds_name_b, text_feature_b, ds_config_b,
                split=split_b, max_items=num_examples_b, streaming=streaming_b
            )
    if len(dedup_list_b) == 0:
        st.write("There are no duplicates in this dataset!")
    else:
        for q in dedup_list_b:
            st.write(q)
