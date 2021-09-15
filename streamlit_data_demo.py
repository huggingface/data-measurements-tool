import math
import numpy as np
import plotly.figure_factory as ff
import streamlit as st
import tokenizers
import transformers
import torch

from dataclasses import asdict
from datasets import (
    Dataset,
    get_dataset_infos,
    list_datasets,
    load_dataset,
    load_dataset_builder,
)
from os.path import join as pjoin

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
                typed_features += get_typed_features(feat.feature, ftype, parents + [name])
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
        if hasattr(feat, 'num_classes'):
            text_features += [(tuple(parents + [name]), feat.names)]
        elif hasattr(feat, 'feature'):
            if hasattr(feat.feature, 'num_classes'):
                text_features += [(tuple(parents + [name]), feat.feature.names)]
            elif isinstance(feat.feature, dict):
                text_features += get_label_features(feat.feature, parents + [name])
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
        index=ds_names.index("squad"),
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
    compute_perplexities_a = st.checkbox(
        "Compute perplexities for the first dataset",
        value=False,
    )

with st.sidebar.expander("Choose second dataset and field"):
    # choose a dataset to analyze
    ds_name_b = st.selectbox(
        "Choose a second dataset to explore:",
        ds_names,
        index=ds_names.index("squad_v2"),
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
    compute_perplexities_b = st.checkbox(
        "Compute perplexities for the second dataset",
        value=False,
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

######## Main window

left_col, right_col = st.columns(2)

### First, show some example texts from the dataset
left_col.markdown(f"### Showing {ds_name_a} - {config_name_a} - {text_feature_a}")
with left_col.expander("Dataset Description A"):
    st.markdown(ds_name_to_dict[ds_name_a])

right_col.markdown(f"### Showing {ds_name_b} - {config_name_b} - {text_feature_b}")
with right_col.expander("Dataset Description B"):
    st.markdown(ds_name_to_dict[ds_name_b])

### Second, show the distribution of text lengths
with left_col.expander("Show text lengths A", expanded=True):
    st.markdown("### Text lengths A")
    sorted_sents_lengths_a, fig_tok_length_a = run_tok_length_analysis(text_dset_a, cache_name_a)
    st.plotly_chart(fig_tok_length_a, use_container_width=True)
    start_id_show_lengths_a = st.slider(
        'Show longest sentences in A starting at index:',
        0, text_dset_a.num_rows - 5, value=0, step=5
    )
    for ln, sent in sorted_sents_lengths_a[start_id_show_lengths_a:start_id_show_lengths_a+5]:
        st.text(f"{ln} | {sent}")

with right_col.expander("Show text lengths B", expanded=True):
    st.markdown("### Text lengths B")
    sorted_sents_lengths_b, fig_tok_length_b = run_tok_length_analysis(text_dset_b, cache_name_b)
    st.plotly_chart(fig_tok_length_b, use_container_width=True)
    start_id_show_lengths_b = st.slider(
        'Show longest sentences in B starting at index:',
        0, text_dset_b.num_rows - 5, value=0, step=5
    )
    for ln, sent in sorted_sents_lengths_b[start_id_show_lengths_b:start_id_show_lengths_b+5]:
        st.text(f"{ln} | {sent}")

### Third, show the distribution of text perplexities
with left_col.expander("Show text perplexities A", expanded=True):
    if compute_perplexities_a:
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
        st.write("To show perplexity of examples, check the `compute perplexities for the first dataset` box left")

with right_col.expander("Show text perplexities B", expanded=True):
    if compute_perplexities_b:
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
        st.write("To show perplexity of examples, check the `compute perplexities for the first dataset` box left")
