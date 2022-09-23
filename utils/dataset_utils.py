# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import pandas as pd
import plotly
import pyarrow.feather as feather
import utils
from dataclasses import asdict
from datasets import Dataset, get_dataset_infos, load_dataset, load_from_disk, \
    NamedSplit
from dotenv import load_dotenv
from huggingface_hub import Repository, list_datasets
from json2html import *
from os import getenv
from os.path import exists, isdir, join as pjoin
from pathlib import Path

# treating inf values as NaN as well
pd.set_option("use_inf_as_na", True)

## String names used in Hugging Face dataset configs.
HF_FEATURE_FIELD = "features"
HF_LABEL_FIELD = "label"
HF_DESC_FIELD = "description"

CACHE_DIR = "cache_dir"
## String names we are using within this code.
# These are not coming from the stored dataset nor HF config,
# but rather used as identifiers in our dicts and dataframes.
TEXT_FIELD = "text"
PERPLEXITY_FIELD = "perplexity"
TOKENIZED_FIELD = "tokenized_text"
EMBEDDING_FIELD = "embedding"
LENGTH_FIELD = "length"
VOCAB = "vocab"
WORD = "word"
CNT = "count"
PROP = "proportion"
TEXT_NAN_CNT = "text_nan_count"
TXT_LEN = "text lengths"
TOT_WORDS = "total words"
TOT_OPEN_WORDS = "total open words"

_DATASET_LIST = [
    "c4",
    "squad",
    "squad_v2",
    "hate_speech18",
    "hate_speech_offensive",
    "glue",
    "super_glue",
    "wikitext",
    "imdb",
]

_STREAMABLE_DATASET_LIST = [
    "c4",
    "wikitext",
]

_MAX_ROWS = 200000

logs = utils.prepare_logging(__file__)

def _load_dotenv_for_cache_on_hub():
    """
    This function loads and returns the organization name that you've set up on the
    hub for storing your data measurements cache on the hub. It also loads the associated
    access token. It expects you to have HUB_CACHE_ORGANIZATION=<the organization you've set up on the hub to store your cache>
    and HF_TOKEN=<your hf token> on separate lines in a file named .env at the root of this repo.

    Returns:
        tuple of strings: hub_cache_organization, hf_token
    """
    if Path(".env").is_file():
        load_dotenv(".env")
    hf_token = getenv("HF_TOKEN")
    hub_cache_organization = getenv("HUB_CACHE_ORGANIZATION")
    return hub_cache_organization, hf_token

def get_cache_dir_naming(out_dir, dataset, config, split, feature):
    feature_text = hyphenated(feature)
    dataset_cache_name = f"{dataset}_{config}_{split}_{feature_text}"
    local_dset_cache_dir = out_dir + "/" + dataset_cache_name
    return dataset_cache_name, local_dset_cache_dir

def initialize_cache_hub_repo(local_cache_dir, dataset_cache_name):
    """
    This function tries to initialize a dataset cache on the huggingface hub. The
    function expects you to have HUB_CACHE_ORGANIZATION=<the organization you've set up on the hub to store your cache>
    and HF_TOKEN=<your hf token> on separate lines in a file named .env at the root of this repo.

    Args:
        local_cache_dir (string):
            The path to the local dataset cache.
        dataset_cache_name (string):
            The name of the dataset repo on the huggingface hub that you want.
    """

    hub_cache_organization, hf_token = _load_dotenv_for_cache_on_hub()
    clone_source = pjoin(hub_cache_organization, dataset_cache_name)
    repo = Repository(local_dir=local_cache_dir,
                      clone_from=clone_source,
                      repo_type="dataset", use_auth_token=hf_token)
    repo.lfs_track(["*.feather"])
    return repo

def pull_cache_from_hub(cache_path, dset_cache_dir):
    """
    This function tries to pull a datasets cache from the huggingface hub if a
    cache for the dataset does not already exist locally. The function expects you
    to have you HUB_CACHE_ORGANIZATION=<the organization you've set up on the hub to store your cache>
    and HF_TOKEN=<your hf token> on separate lines in a file named .env at the root of this repo.

    Args:
        cache_path (string):
            The path to the local dataset cache that you want.
        dset_cache_dir (string):
            The name of the dataset repo on the huggingface hub.

    """

    hub_cache_organization, hf_token = _load_dotenv_for_cache_on_hub()
    clone_source = pjoin(hub_cache_organization, dset_cache_dir)

    if isdir(cache_path):
        logs.warning("Already a local cache for the dataset, so not pulling from the hub.")
    else:
        # Here, dataset_info.id is of the form: <hub cache organization>/<dataset cache dir>
        if dset_cache_dir in [
            dataset_info.id.split("/")[-1] for dataset_info in
            list_datasets(author=hub_cache_organization,
                          use_auth_token=hf_token)]:
            Repository(local_dir=cache_path,
                       clone_from=clone_source,
                       repo_type="dataset", use_auth_token=hf_token)
            logs.info("Pulled cache from hub!")
        else:
            logs.warning("Asking to pull cache from hub but cannot find cached repo on the hub.")


def load_truncated_dataset(
    dataset_name,
    config_name,
    split_name,
    num_rows=_MAX_ROWS,
    use_cache=True,
    cache_dir=CACHE_DIR,
    use_streaming=True,
    save=True,
):
    """
    This function loads the first `num_rows` items of a dataset for a
    given `config_name` and `split_name`.
    If `use_cache` and `cache_name` exists, the truncated dataset is loaded from
    `cache_name`.
    Otherwise, a new truncated dataset is created and immediately saved
    to `cache_name`.
    When the dataset is streamable, we iterate through the first
    `num_rows` examples in streaming mode, write them to a jsonl file,
    then create a new dataset from the json.
    This is the most direct way to make a Dataset from an IterableDataset
    as of datasets version 1.6.1.
    Otherwise, we download the full dataset and select the first
    `num_rows` items
    Args:
        dataset_name (string):
            dataset id in the dataset library
        config_name (string):
            dataset configuration
        split_name (string):
            split name
        num_rows (int) [optional]:
            number of rows to truncate the dataset to
        cache_dir (string):
            name of the cache directory
        use_cache (bool):
            whether to load from the cache if it exists
        use_streaming (bool):
            whether to use streaming when the dataset supports it
        save (bool):
            whether to save the dataset locally
    Returns:
        Dataset: the (truncated if specified) dataset as a Dataset object
    """
    logs.info("Loading or preparing dataset saved in %s " % cache_dir)
    if use_cache and exists(cache_dir):
        dataset = load_from_disk(cache_dir)
    else:
        if use_streaming and dataset_name in _STREAMABLE_DATASET_LIST:
            iterable_dataset = load_dataset(
                dataset_name,
                name=config_name,
                split=split_name,
                streaming=True,
            ).take(num_rows)
            rows = list(iterable_dataset)
            f = open("temp.jsonl", "w", encoding="utf-8")
            for row in rows:
                _ = f.write(json.dumps(row) + "\n")
            f.close()
            dataset = Dataset.from_json(
                "temp.jsonl", features=iterable_dataset.features, split=NamedSplit(split_name)
            )
        else:
            full_dataset = load_dataset(
                dataset_name,
                name=config_name,
                split=split_name,
            )
            if len(full_dataset) >= num_rows:
                dataset = full_dataset.select(range(num_rows))
                # Make the directory name clear that it's not the full dataset.
                cache_dir = pjoin(cache_dir, ("_%s" % num_rows))
            else:
                dataset = full_dataset
        if save:
            dataset.save_to_disk(cache_dir)
    return dataset

def hyphenated(features):
    """When multiple features are asked for, hyphenate them together when they're used for filenames or titles"""
    return '-'.join(features)

def get_typed_features(features, ftype="string", parents=None):
    """
    Recursively get a list of all features of a certain dtype
    :param features:
    :param ftype:
    :param parents:
    :return: a list of tuples > e.g. ('A', 'B', 'C') for feature example['A']['B']['C']
    """
    if parents is None:
        parents = []
    typed_features = []
    for name, feat in features.items():
        if isinstance(feat, dict):
            if feat.get("dtype", None) == ftype or feat.get("feature", {}).get(
                ("dtype", None) == ftype
            ):
                typed_features += [tuple(parents + [name])]
            elif "feature" in feat:
                if feat["feature"].get("dtype", None) == ftype:
                    typed_features += [tuple(parents + [name])]
                elif isinstance(feat["feature"], dict):
                    typed_features += get_typed_features(
                        feat["feature"], ftype, parents + [name]
                    )
            else:
                for k, v in feat.items():
                    if isinstance(v, dict):
                        typed_features += get_typed_features(
                            v, ftype, parents + [name, k]
                        )
        elif name == "dtype" and feat == ftype:
            typed_features += [tuple(parents)]
    return typed_features


def get_label_features(features, parents=None):
    """
    Recursively get a list of all features that are ClassLabels
    :param features:
    :param parents:
    :return: pairs of tuples as above and the list of class names
    """
    if parents is None:
        parents = []
    label_features = []
    for name, feat in features.items():
        if isinstance(feat, dict):
            if "names" in feat:
                label_features += [(tuple(parents + [name]), feat["names"])]
            elif "feature" in feat:
                if "names" in feat:
                    label_features += [
                        (tuple(parents + [name]), feat["feature"]["names"])
                    ]
                elif isinstance(feat["feature"], dict):
                    label_features += get_label_features(
                        feat["feature"], parents + [name]
                    )
            else:
                for k, v in feat.items():
                    if isinstance(v, dict):
                        label_features += get_label_features(v, parents + [name, k])
        elif name == "names":
            label_features += [(tuple(parents), feat)]
    return label_features


# get the info we need for the app sidebar in dict format
def dictionarize_info(dset_info):
    info_dict = asdict(dset_info)
    res = {
        "config_name": info_dict["config_name"],
        "splits": {
            spl: spl_info["num_examples"]
            for spl, spl_info in info_dict["splits"].items()
        },
        "features": {
            "string": get_typed_features(info_dict["features"], "string"),
            "int32": get_typed_features(info_dict["features"], "int32"),
            "float32": get_typed_features(info_dict["features"], "float32"),
            "label": get_label_features(info_dict["features"]),
        },
        "description": dset_info.description,
    }
    return res

def get_dataset_info_dicts(dataset_id=None):
    """
    Creates a dict from dataset configs.
    Uses the datasets lib's get_dataset_infos
    :return: Dictionary mapping dataset names to their configurations
    """
    if dataset_id is not None:
        ds_name_to_conf_dict = {
            dataset_id: {
                config_name: dictionarize_info(config_info)
                for config_name, config_info in get_dataset_infos(dataset_id).items()
            }
        }
    else:
        ds_name_to_conf_dict = {
            ds_id: {
                config_name: dictionarize_info(config_info)
                for config_name, config_info in get_dataset_infos(ds_id).items()
            }
            for ds_id in _DATASET_LIST
        }
    return ds_name_to_conf_dict


# get all instances of a specific field in a dataset
def extract_field(examples, field_path, new_field_name=None):
    if new_field_name is None:
        new_field_name = "_".join(field_path)
    field_list = []
    # TODO: Breaks the CLI if this isn't checked.
    if isinstance(field_path, str):
        field_path = [field_path]
    item_list = examples[field_path[0]]
    for field_name in field_path[1:]:
        item_list = [
            next_item
            for item in item_list
            for next_item in (
                item[field_name]
                if isinstance(item[field_name], list)
                else [item[field_name]]
            )
        ]
    field_list += [
        field
        for item in item_list
        for field in (item if isinstance(item, list) else [item])
    ]
    return {new_field_name: field_list}

def make_path(path):
    os.makedirs(path, exist_ok=True)

def counter_dict_to_df(dict_input):
    df_output = pd.DataFrame(dict_input, index=[0]).T
    df_output.columns = ["count"]
    return df_output.sort_values(by="count", ascending=False)

def write_plotly(fig, fid):
    write_json(plotly.io.to_json(fig), fid)

def read_plotly(fid):
    fig = plotly.io.from_json(json.load(open(fid, encoding="utf-8")))
    return fig

def write_json_as_html(input_json, html_fid):
    html_dict = json2html.convert(json=input_json)
    with open(html_fid, "w+") as f:
        f.write(html_dict)

def df_to_write_html(input_df, html_fid):
    """Writes a dataframe to an HTML file"""
    input_df.to_HTML(html_fid)

def read_df(df_fid):
    return pd.DataFrame.from_dict(read_json(df_fid), orient="index")

def write_df(df, df_fid):
    """In order to preserve the index of our dataframes, we can't
    use the compressed pandas dataframe file format .feather.
    There's a preference for json amongst HF devs, so we use that here."""
    df_dict = df.to_dict('index')
    write_json(df_dict, df_fid)

def write_json(json_dict, json_fid):
    with open(json_fid, "w", encoding="utf-8") as f:
        json.dump(json_dict, f)

def read_json(json_fid):
    json_dict = json.load(open(json_fid, encoding="utf-8"))
    return json_dict