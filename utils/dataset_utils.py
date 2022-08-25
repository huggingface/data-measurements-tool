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
from dataclasses import asdict
from os.path import exists, isdir, join as pjoin
import plotly
import pyarrow.feather as feather
import pandas as pd
from datasets import Dataset, get_dataset_infos, load_dataset, load_from_disk, \
    NamedSplit
from huggingface_hub import Repository, list_datasets
from json2html import *
from dotenv import load_dotenv
from pathlib import Path
from os import getenv

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

def initialize_cache_hub_repo(cache_path, dataset_cache_dir):
    """
    This function tries to initialize a dataset cache on the huggingface hub. The
    function expects you to have HUB_CACHE_ORGANIZATION=<the organization you've set up on the hub to store your cache>
    and HF_TOKEN=<your hf token> on separate lines in a file named .env at the root of this repo.

    Args:
        cache_path (string):
            The path to the local dataset cache.
        dataset_cache_dir (string):
            The name of the dataset repo on the huggingface hub that you want.
    """

    hub_cache_organization, hf_token = _load_dotenv_for_cache_on_hub()
    clone_source = pjoin(hub_cache_organization, dataset_cache_dir)
    repo = Repository(local_dir=cache_path,
                      clone_from=clone_source,
                      repo_type="dataset", use_auth_token=hf_token)
    repo.lfs_track(["*.feather"])
    return repo

def pull_cache_from_hub(cache_path, dataset_cache_dir):
    """
    This function tries to pull a datasets cache from the huggingface hub if a
    cache for the dataset does not already exist locally. The function expects you
    to have you HUB_CACHE_ORGANIZATION=<the organization you've set up on the hub to store your cache>
    and HF_TOKEN=<your hf token> on separate lines in a file named .env at the root of this repo.

    Args:
        cache_path (string):
            The path to the local dataset cache that you want.
        dataset_cache_dir (string):
            The name of the dataset repo on the huggingface hub.

    Returns:
        string: a log about whether the cache was pulled or not
    """

    hub_cache_organization, hf_token = _load_dotenv_for_cache_on_hub()
    clone_source = pjoin(hub_cache_organization, dataset_cache_dir)

    log = "Pulled cache from hub!"
    if not isdir(cache_path):
        # Here, dataset_info.id is of the form: <hub cache organization>/<dataset cache dir>
        if dataset_cache_dir in [
            dataset_info.id.split("/")[-1] for dataset_info in
            list_datasets(author=hub_cache_organization,
                          use_auth_token=hf_token)]:
            repo = Repository(local_dir=cache_path,
                              clone_from=clone_source,
                              repo_type="dataset", use_auth_token=hf_token)
        else:
            log = "Asking to pull cache from hub but cannot find cached repo on the hub."
    else:
        log = "Already a local cache for the dataset, so not pulling from the hub."
    return log


def load_truncated_dataset(
    dataset_name,
    config_name,
    split_name,
    num_rows=_MAX_ROWS,
    cache_name=None,
    use_streaming=True,
):
    """
    This function loads the first `num_rows` items of a dataset for a
    given `config_name` and `split_name`.
    If `cache_name` exists, the truncated dataset is loaded from `cache_name`.
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
        num_rows (int):
            number of rows to truncate the dataset to
        cache_name (string):
            name of the cache directory
        use_streaming (bool):
            whether to use streaming when the dataset supports it
    Returns:
        Dataset: the truncated dataset as a Dataset object
    """
    if cache_name is None:
        cache_name = f"{dataset_name}_{config_name}_{split_name}_{num_rows}"
    if exists(cache_name):
        dataset = load_from_disk(cache_name)
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
            else:
                dataset = full_dataset
        dataset.save_to_disk(cache_name)
    return dataset


def intersect_dfs(df_dict):
    started = 0
    new_df = None
    for key, df in df_dict.items():
        if df is None:
            continue
        for key2, df2 in df_dict.items():
            if df2 is None:
                continue
            if key == key2:
                continue
            if started:
                new_df = new_df.join(df2, how="inner", lsuffix="1", rsuffix="2")
            else:
                new_df = df.join(df2, how="inner", lsuffix="1", rsuffix="2")
                started = 1
    return new_df.copy()


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

def make_path(input_path):
    os.makedirs(input_path, exist_ok=True)

def counter_dict_to_df(dict_input):
    df_output = pd.DataFrame(dict_input, index=[0]).T
    df_output.columns = [CNT]
    return df_output

def write_plotly(fig, fid):
    make_path(fid)
    write_json(plotly.io.to_json(fig), fid)

def read_plotly(fid):
    fig = plotly.io.from_json(json.load(open(fid, encoding="utf-8")))
    return fig

def write_json_as_html(input_json, html_fid):
    make_path(html_fid)
    html_dict = json2html.convert(json=input_json)
    with open(html_fid, "w+") as f:
        f.write(html_dict)

def df_to_write_html(input_df, html_fid):
    """Writes a dataframe to an HTML file"""
    make_path(html_fid)
    input_df.to_HTML(html_fid)

def read_df(df_fid):
    df = feather.read_feather(df_fid)
    return df


def write_df(df, df_fid):
    make_path(df_fid)
    feather.write_feather(df, df_fid)


def write_json(json_dict, json_fid):
    make_path(json_fid)
    with open(json_fid, "w", encoding="utf-8") as f:
        json.dump(json_dict, f)


def read_json(json_fid):
    json_dict = json.load(open(json_fid, encoding="utf-8"))
    return json_dict


class FileHandler:
    """Ensures a standardized naming scheme for cache files for
    different modules."""

    def __init__(self, module, name):
        module_dir = name
        module_json = name + ".json"
        module_fig_json = name + "_fig.json"
        module_html = name + ".html"
        module_png = name + ".png"
        self.module_result_json_fid = pjoin(module.cache_path, module_dir,
                                       module_json)
        self.module_result_fig_json_fid = pjoin(module.cache_path, module_dir,
                                       module_fig_json)
        self.module_result_html_fid = pjoin(module.cache_path, module_dir,
                                       module_html)
        self.module_result_png_fid = pjoin(module.cache_path, module_dir,
                                       module_png)

    def get_filenames(self, has_HTML=False, has_png=False, has_fig_json=False):
        filenames = {"statistics": self.module_result_json_fid}
        if has_HTML:
            filenames["html"] = self.module_result_html_fid
        if has_png:
            filenames["png"] = self.module_result_png_fid
        if has_fig_json:
            filenames["fig json"] = self.module_result_fig_json_fid
        return filenames