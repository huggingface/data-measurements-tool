import math
import numpy as np
import plotly.figure_factory as ff
import streamlit as st
import tokenizers
import torch

from datasets import (
    import_main_class,
    list_datasets,
    load_dataset,
    load_dataset_builder,
    prepare_module,
)
from os.path import join as pjoin
from transformers import AutoTokenizer, AutoModelForCausalLM

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

# list of fiels we want to keep around from the DatasetInfo object
keep_info_fields = [
    "features",
    "config_name",
    "splits",
]

# get DatasetInfo object without downloading the dataset
def get_config_infos(name):
    module_path, *_ = prepare_module(name, dataset=True)
    builder_cls = import_main_class(module_path, dataset=True)
    configs = [c.name for c in builder_cls.BUILDER_CONFIGS] or [None]
    if len(configs) == 1:
        info_dict = load_dataset_builder(name).info.__dict__
        return [
            {k: info_dict[k] for k in keep_info_fields}
        ]
    else:
        config_list = []
        for config_name in configs:
            info_dict = load_dataset_builder(name, config_name).info.__dict__
            config_list += [
                {k: info_dict[k] for k in keep_info_fields}
            ]
        return config_list

# Recursively get a list of all features of a certain dtype
# the output is a list of tuples > e.g. ('A', 'B', 'C') for feature example['A']['B']['C']
def get_typed_features(features, ftype='string', parents=None):
    if parents is None:
        parents = []
    typed_features = []
    for name, feat in features.items():
        if hasattr(feat, 'dtype') and feat.dtype == ftype:
            typed_features += [tuple(parents + [name])]
        elif hasattr(feat, 'feature'):
            if hasattr(feat.feature, 'dtype') and feat.feature.dtype == ftype:
                typed_features += [tuple(parents + [name])]
            elif isinstance(feat.feature, dict):
                typed_features += get_typed_features(feat.feature, ftype, parents + [name])
    return typed_features

# Recursively get a list of all features that are ClassLabels
# the outputs are pairs of tuples as above and the list of class names
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
def dictionarize_info(info_dict):
    res = {}
    res["config_name"] = info_dict["config_name"]
    res["splits"] = {
        spl: spl_info.num_examples
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
@st.cache(persist=True, allow_output_mutation=True)
def get_config_infos_dict(name):
    return {
        config_info["config_name"]: dictionarize_info(config_info)
        for config_info in get_config_infos(name)
    }

####### prepare text and scores
# extract text to analyze:
# list of all examples of a given feature in a given split of a given config of a given dataset
# returns a dataset of strings
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

@st.cache(persist=True, allow_output_mutation=True)
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
tokenizer = AutoTokenizer.from_pretrained("xlnet-base-cased")
model = AutoModelForCausalLM.from_pretrained("xlnet-base-cased")
def get_single_sent_loss(sent):
    batch = tokenizer(sent, return_tensors="pt", padding=True)
    batch['labels'] = batch['input_ids']
    loss = model(**batch).loss.item()
    return loss

########## streamlit code

ds_list = list_datasets(with_community_datasets=False, with_details=True)
ds_name_to_dict = {ds.id: ds.description for ds in ds_list}
ds_names = list(ds_name_to_dict.keys())

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
    num_examples_a = st.slider(
        "Number of examples to analyze in the first dataset:",
        0,
        min(ds_config_a["splits"][split_a], 50000),
        value=min(1000, ds_config_a["splits"][split_a]),
        step=1000,
    )
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
    num_examples_b = st.slider(
        "Number of examples to analyze in the second dataset:",
        0,
        min(ds_config_b["splits"][split_b], 50000),
        value=min(1000, ds_config_b["splits"][split_b]),
        step=1000,
    )
    streaming_b = st.checkbox(
        "Use streaming functionality for the second dataset",
        value=False,
    )
    compute_perplexities_b = st.checkbox(
        "Compute perplexities for the second dataset",
        value=False,
    )


# Grab the text requested
text_dset_a = get_text_to_analyze(
    ds_name_a, text_feature_a, ds_config_a,
    split=split_a, max_items=num_examples_a, streaming=streaming_a
)
text_dset_b = get_text_to_analyze(
    ds_name_b, text_feature_b, ds_config_b,
    split=split_b, max_items=num_examples_b, streaming=streaming_b
)

cache_name_a = f"{ds_name_a}_{config_name_a}_{split_a}_{'-'.join(text_feature_a)}_{num_examples_a}"
cache_name_b = f"{ds_name_b}_{config_name_b}_{split_b}_{'-'.join(text_feature_b)}_{num_examples_b}"

text_dset_a_lengths = text_dset_a.map(
    lambda exple: {"space_tok_length": len(exple["text"].split())},
    load_from_cache_file=True,
    cache_file_name=pjoin("cache_dir", f"{cache_name_a}_space_tok_length"),
)
text_dset_b_lengths = text_dset_b.map(
    lambda exple: {"space_tok_length": len(exple["text"].split())},
    load_from_cache_file=True,
    cache_file_name=pjoin("cache_dir", f"{cache_name_b}_space_tok_length"),
)

if compute_perplexities_a:
    text_dset_a_loss = text_dset_a.map(
        lambda exple: {"xlnet_loss": get_single_sent_loss(exple["text"])},
        load_from_cache_file=True,
        cache_file_name=pjoin("cache_dir", f"{cache_name_a}_xlnet_loss"),
    )
if compute_perplexities_b:
    text_dset_b_loss = text_dset_b.map(
        lambda exple: {"xlnet_loss": get_single_sent_loss(exple["text"])},
        load_from_cache_file=True,
        cache_file_name=pjoin("cache_dir", f"{cache_name_b}_xlnet_loss"),
    )

######## Main window

left_col, right_col = st.columns(2)

### First, show some example texts from the dataset
left_col.markdown(f"### Showing {ds_name_a} - {config_name_a} - {text_feature_a}")
with left_col.expander("Dataset Description A"):
    st.markdown(ds_name_to_dict[ds_name_a])

with left_col.expander("Show some examples A"):
    st.markdown("### Example text fields A")
    start_id_show = st.slider('Starting index A:', 0, text_dset_a.num_rows - 5, value=0, step=5)
    st.dataframe(text_dset_a[start_id_show:start_id_show+10]["text"])

right_col.markdown(f"### Showing {ds_name_b} - {config_name_b} - {text_feature_b}")
with right_col.expander("Dataset Description B"):
    st.markdown(ds_name_to_dict[ds_name_b])

with right_col.expander("Show some examples B"):
    st.markdown("### Example text fields B")
    start_id_show = st.slider('Starting index B:', 0, text_dset_b.num_rows - 10, 5)
    st.dataframe(text_dset_b[start_id_show:start_id_show+10]["text"])

### Second, show the distribution of text lengths
with left_col.expander("Show text lengths A", expanded=True):
    st.markdown("### Text lengths A")
    hist_data_tok_length_a = [text_dset_a_lengths["space_tok_length"]]
    fig_tok_length_a = ff.create_distplot(hist_data_tok_length_a, group_labels=["text lengths"])
    st.plotly_chart(fig_tok_length_a, use_container_width=True)
    sorted_sents_lengths_a = [
        s for s, l in sorted(
            [(sentence["text"], sentence["space_tok_length"]) for sentence in text_dset_a_lengths],
            key=lambda x:x[1], reverse=True,
        )
    ]
    start_id_show_lengths_a = st.slider(
        'Show longest sentences in A starting at index:',
        0, text_dset_a.num_rows - 5, value=0, step=5
    )
    for sent in sorted_sents_lengths_a[start_id_show_lengths_a:start_id_show_lengths_a+5]:
        st.text(sent)

with right_col.expander("Show text lengths B", expanded=True):
    st.markdown("### Text lengths B")
    hist_data_tok_length_b = [text_dset_b_lengths["space_tok_length"]]
    fig_tok_length_b = ff.create_distplot(hist_data_tok_length_b, group_labels=["text lengths"])
    st.plotly_chart(fig_tok_length_b, use_container_width=True)
    sorted_sents_lengths_b = [
        s for s, l in sorted(
            [(sentence["text"], sentence["space_tok_length"]) for sentence in text_dset_b_lengths],
            key=lambda x:x[1], reverse=True,
        )
    ]
    start_id_show_lengths_b = st.slider(
        'Show longest sentences in B starting at index:',
        0, text_dset_a.num_rows - 5, value=0, step=5
    )
    for sent in sorted_sents_lengths_b[start_id_show_lengths_b:start_id_show_lengths_b+5]:
        st.text(sent)

### Third, show the shortest sentences
with left_col.expander("Here are the shortest sentences in the column", expanded=True):
    st.markdown("### Shortest sentences in A")
    hist_data_tok_length_a = [text_dset_a_lengths["space_tok_length"]]
    fig_tok_length_a = ff.create_distplot(hist_data_tok_length_a, group_labels=["text lengths"])
    st.plotly_chart(fig_tok_length_a, use_container_width=True)
    sorted_sents_lengths_a = [
        s for s, l in sorted(
            [(sentence["text"], sentence["space_tok_length"]) for sentence in text_dset_a_lengths],
            key=lambda x:x[1], reverse=False,
        )
    ]
    start_id_show_lengths_a = st.slider(
        'Show shortest sentences in A starting at index:',
        0, text_dset_a.num_rows - 5, value=0, step=5
    )
    for sent in sorted_sents_lengths_a[start_id_show_lengths_a:start_id_show_lengths_a+5]:
        st.text(sent)

with right_col.expander("Here are the shortest sentences in the column", expanded=True):
    st.markdown("### Shortest sentences in B")
    hist_data_tok_length_b = [text_dset_b_lengths["space_tok_length"]]
    fig_tok_length_b = ff.create_distplot(hist_data_tok_length_b, group_labels=["text lengths"])
    st.plotly_chart(fig_tok_length_b, use_container_width=True)
    sorted_sents_lengths_b = [
        s for s, l in sorted(
            [(sentence["text"], sentence["space_tok_length"]) for sentence in text_dset_b_lengths],
            key=lambda x:x[1], reverse=False,
        )
    ]
    start_id_show_lengths_b = st.slider(
        'Show shortest sentences in B starting at index:',
        0, text_dset_a.num_rows - 5, value=0, step=5
    )
    for sent in sorted_sents_lengths_b[start_id_show_lengths_b:start_id_show_lengths_b+5]:
        st.text(sent)

### Third, show the distribution of text perplexities
with left_col.expander("Show text perplexities A", expanded=True):
    if compute_perplexities_a:
        st.markdown("### Text perplexities A")
        hist_data_loss_a = [text_dset_a_loss["xlnet_loss"]]
        fig_loss_a = ff.create_distplot(hist_data_loss_a, group_labels=["text perplexity"])
        st.plotly_chart(fig_loss_a, use_container_width=True)
        sorted_sents_loss_a = [
            s for s, l in sorted(
                [(sentence["text"], sentence["xlnet_loss"]) for sentence in text_dset_a_loss],
                key=lambda x:x[1], reverse=True,
            )
        ]
        start_id_show_loss_a = st.slider(
            'Show highest perplexity sentences in A starting at index:',
            0, text_dset_a.num_rows - 5, value=0, step=5
        )
        for sent in sorted_sents_loss_a[start_id_show_loss_a:start_id_show_loss_a+5]:
            st.text(sent)
    else:
        st.write("To show perplexity of examples, check the `compute perplexities for the first dataset` box left")

with right_col.expander("Show text perplexities B", expanded=True):
    if compute_perplexities_b:
        st.markdown("### Text perplexities B")
        hist_data_loss_b = [text_dset_b_loss["xlnet_loss"]]
        fig_loss_b = ff.create_distplot(hist_data_loss_b, group_labels=["text perplexity"])
        st.plotly_chart(fig_loss_b, use_container_width=True)
        sorted_sents_loss_b = [
            s for s, l in sorted(
                [(sentence["text"], sentence["xlnet_loss"]) for sentence in text_dset_b_loss],
                key=lambda x:x[1], reverse=True,
            )
        ]
        start_id_show_loss_b = st.slider(
            'Show highest perplexity sentences in B starting at index:',
            0, text_dset_b.num_rows - 5, value=0, step=5
        )
        for sent in sorted_sents_loss_b[start_id_show_loss_b:start_id_show_loss_b+5]:
            st.text(sent)
    else:
        st.write("To show perplexity of examples, check the `compute perplexities for the second dataset` box left")
