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
from os.path import exists
import plotly
import pyarrow.feather as feather
import pandas as pd
from datasets import Dataset, get_dataset_infos, load_dataset, load_from_disk, \
    NamedSplit

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
OUR_TEXT_FIELD = "text"
PERPLEXITY_FIELD = "perplexity"
OUR_LABEL_FIELD = "label"
TOKENIZED_FIELD = "tokenized_text"
EMBEDDING_FIELD = "embedding"
LENGTH_FIELD = "length"
VOCAB = "vocab"
WORD = "word"
CNT = "count"
PROP = "proportion"
TEXT_NAN_CNT = "text_nan_count"
TXT_LEN = "text lengths"
DEDUP_TOT = "dedup_total"
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


def load_truncated_dataset(
    dataset_name,
    config_name,
    split_name,
    num_rows=_MAX_ROWS,
    cache_name=None,
    use_cache=True,
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
        use_cache (bool):
            whether to load form the cache if it exists
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

def make_cache_path(cache_path):
    os.makedirs(cache_path, exist_ok=True)


def write_plotly(fig, fid):
    write_json(plotly.io.to_json(fig), fid)


def read_plotly(fid):
    fig = plotly.io.from_json(json.load(open(fid, encoding="utf-8")))
    return fig


def read_df(df_fid):
    df = feather.read_feather(df_fid)
    return df


def write_df(df, df_fid):
    feather.write_feather(df, df_fid)


def write_json(json_dict, json_fid):
    with open(json_fid, "w", encoding="utf-8") as f:
        json.dump(json_dict, f)


def read_json(json_fid):
    json_dict = json.load(open(json_fid, encoding="utf-8"))
    return json_dict