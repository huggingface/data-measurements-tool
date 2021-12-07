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

import math
from os.path import exists
from os.path import join as pjoin

import plotly.graph_objects as go
import torch
import transformers
from datasets import load_from_disk
from plotly.io import read_json
from tqdm import tqdm

from .dataset_utils import EMBEDDING_FIELD


def sentence_mean_pooling(model_output, attention_mask):
    """Mean pooling of token embeddings for a sentence."""
    token_embeddings = model_output[
        0
    ]  # First element of model_output contains all token embeddings
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


class Embeddings:
    def __init__(
        self,
        dstats=None,
        text_dset=None,
        text_field_name="text",
        cache_path="",
        use_cache=False,
    ):
        """Item embeddings and clustering"""
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model_name = "sentence-transformers/all-mpnet-base-v2"
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name)
        self.model = transformers.AutoModel.from_pretrained(self.model_name).to(
            self.device
        )
        self.text_dset = text_dset if dstats is None else dstats.text_dset
        self.text_field_name = (
            text_field_name if dstats is None else dstats.our_text_field
        )
        self.cache_path = cache_path if dstats is None else dstats.cache_path
        self.embeddings_dset_fid = pjoin(self.cache_path, "embeddings_dset")
        self.embeddings_dset = None
        self.node_list_fid = pjoin(self.cache_path, "node_list.th")
        self.node_list = None
        self.nid_map = None
        self.fig_tree_fid = pjoin(self.cache_path, "node_figure.json")
        self.fig_tree = None
        self.cached_clusters = {}
        self.use_cache = use_cache

    def compute_sentence_embeddings(self, sentences):
        """
        Takes a list of sentences and computes their embeddings
        using self.tokenizer and self.model (with output dimension D)
        followed by mean pooling of the token representations and normalization
        Args:
            sentences ([string]): list of N input sentences
        Returns:
            torch.Tensor: sentence embeddings, dimension NxD
        """
        batch = self.tokenizer(
            sentences, padding=True, truncation=True, return_tensors="pt"
        )
        batch = {k: v.to(self.device) for k, v in batch.items()}
        with torch.no_grad():
            model_output = self.model(**batch)
            sentence_embeds = sentence_mean_pooling(
                model_output, batch["attention_mask"]
            )
            sentence_embeds /= sentence_embeds.norm(dim=-1, keepdim=True)
            return sentence_embeds

    def make_embeddings(self):
        """
        Batch computes the embeddings of the Dataset self.text_dset,
        using the field self.text_field_name as input.
        Returns:
            Dataset: HF dataset object with a single EMBEDDING_FIELD field
                corresponding to the embeddings (list of floats)
        """

        def batch_embed_sentences(sentences):
            return {
                EMBEDDING_FIELD: [
                    embed.tolist()
                    for embed in self.compute_sentence_embeddings(
                        sentences[self.text_field_name]
                    )
                ]
            }

        self.embeddings_dset = self.text_dset.map(
            batch_embed_sentences,
            batched=True,
            batch_size=32,
            remove_columns=[self.text_field_name],
        )

        return self.embeddings_dset

    def make_text_embeddings(self):
        """Load embeddings dataset from cache or compute it."""
        if self.use_cache and exists(self.embeddings_dset_fid):
            self.embeddings_dset = load_from_disk(self.embeddings_dset_fid)
        else:
            self.embeddings_dset = self.make_embeddings()
            self.embeddings_dset.save_to_disk(self.embeddings_dset_fid)

    def make_hierarchical_clustering(
        self,
        batch_size=1000,
        approx_neighbors=1000,
        min_cluster_size=10,
    ):
        if self.use_cache and exists(self.node_list_fid):
            self.node_list, self.nid_map = torch.load(self.node_list_fid)
        else:
            self.make_text_embeddings()
            embeddings = torch.Tensor(self.embeddings_dset[EMBEDDING_FIELD])
            self.node_list = fast_cluster(
                embeddings, batch_size, approx_neighbors, min_cluster_size
            )
            self.nid_map = dict(
                [(node["nid"], nid) for nid, node in enumerate(self.node_list)]
            )
            torch.save((self.node_list, self.nid_map), self.node_list_fid)
        if self.use_cache and exists(self.fig_tree_fid):
            self.fig_tree = read_json(self.fig_tree_fid)
        else:
            self.fig_tree = make_tree_plot(
                self.node_list, self.text_dset, self.text_field_name
            )
            self.fig_tree.write_json(self.fig_tree_fid)

    def find_cluster_beam(self, sentence, beam_size=20):
        """
        This function finds the `beam_size` leaf clusters that are closest to the
        proposed sentence and returns the full path from the root to the cluster
        along with the dot product between the sentence embedding and the
        cluster centroid
        Args:
            sentence (string): input sentence for which to find clusters
            beam_size (int): this is a beam size algorithm to explore the tree
        Returns:
            [([int], float)]: list of (path_from_root, score) sorted by score
        """
        embed = self.compute_sentence_embeddings([sentence])[0].to("cpu")
        active_paths = [([0], torch.dot(embed, self.node_list[0]["centroid"]).item())]
        finished_paths = []
        children_ids_list = [
            [
                self.nid_map[nid]
                for nid in self.node_list[path[-1]]["children_ids"]
                if nid in self.nid_map
            ]
            for path, score in active_paths
        ]
        while len(active_paths) > 0:
            next_ids = sorted(
                [
                    (
                        beam_id,
                        nid,
                        torch.dot(embed, self.node_list[nid]["centroid"]).item(),
                    )
                    for beam_id, children_ids in enumerate(children_ids_list)
                    for nid in children_ids
                ],
                key=lambda x: x[2],
                reverse=True,
            )[:beam_size]
            paths = [
                (active_paths[beam_id][0] + [next_id], score)
                for beam_id, next_id, score in next_ids
            ]
            active_paths = []
            for path, score in paths:
                if (
                    len(
                        [
                            nid
                            for nid in self.node_list[path[-1]]["children_ids"]
                            if nid in self.nid_map
                        ]
                    )
                    > 0
                ):
                    active_paths += [(path, score)]
                else:
                    finished_paths += [(path, score)]
            children_ids_list = [
                [
                    self.nid_map[nid]
                    for nid in self.node_list[path[-1]]["children_ids"]
                    if nid in self.nid_map
                ]
                for path, score in active_paths
            ]
        return sorted(
            finished_paths,
            key=lambda x: x[-1],
            reverse=True,
        )[:beam_size]


def prepare_merges(embeddings, batch_size=1000, approx_neighbors=1000, low_thres=0.5):
    """
    Prepares an initial list of merges for hierarchical
    clustering. First compute the `approx_neighbors` nearest neighbors,
    then propose a merge for any two points that are closer than `low_thres`

    Note that if a point has more than `approx_neighbors` neighbors
    closer than `low_thres`, this approach will miss some of those merges

    Args:
        embeddings (toch.Tensor): Tensor of sentence embeddings - dimension NxD
        batch_size (int): compute nearest neighbors of `batch_size` points at a time
        approx_neighbors (int): only keep `approx_neighbors` nearest neighbors of a point
        low_thres (float): only return merges where the dot product is greater than `low_thres`
    Returns:
        torch.LongTensor: proposed merges ([i, j] with i>j) - dimension: Mx2
        torch.Tensor: merge scores - dimension M
    """
    top_idx_pre = torch.cat(
        [torch.LongTensor(range(embeddings.shape[0]))[:, None]] * batch_size, dim=1
    )
    top_val_all = torch.Tensor(0, approx_neighbors)
    top_idx_all = torch.LongTensor(0, approx_neighbors)
    n_batches = math.ceil(len(embeddings) / batch_size)
    for b in tqdm(range(n_batches)):
        # TODO: batch across second dimension
        cos_scores = torch.mm(
            embeddings[b * batch_size : (b + 1) * batch_size], embeddings.t()
        )
        for i in range(cos_scores.shape[0]):
            cos_scores[i, (b * batch_size) + i :] = -1
        top_val_large, top_idx_large = cos_scores.topk(
            k=approx_neighbors, dim=-1, largest=True
        )
        top_val_all = torch.cat([top_val_all, top_val_large], dim=0)
        top_idx_all = torch.cat([top_idx_all, top_idx_large], dim=0)
        max_neighbor_dist = top_val_large[:, -1].max().item()
        if max_neighbor_dist > low_thres:
            print(
                f"WARNING: with the current set of neireast neighbor, the farthest is {max_neighbor_dist}"
            )

    all_merges = torch.cat(
        [
            top_idx_pre[top_val_all > low_thres][:, None],
            top_idx_all[top_val_all > low_thres][:, None],
        ],
        dim=1,
    )
    all_merge_scores = top_val_all[top_val_all > low_thres]

    return (all_merges, all_merge_scores)


def merge_nodes(nodes, current_thres, previous_thres, all_merges, all_merge_scores):
    """
    Merge all nodes if the max dot product between any of their descendants
    is greater than current_thres.

    Args:
        nodes ([dict]): list of dicts representing the current set of nodes
        current_thres (float): merge all nodes closer than current_thres
        previous_thres (float): nodes closer than previous_thres are already merged
        all_merges (torch.LongTensor): proposed merges ([i, j] with i>j) - dimension: Mx2
        all_merge_scores (torch.Tensor): merge scores - dimension M
    Returns:
        [dict]: extended list with the newly created internal nodes
    """
    merge_ids = (all_merge_scores <= previous_thres) * (
        all_merge_scores > current_thres
    )
    if merge_ids.sum().item() > 0:
        merges = all_merges[merge_ids]
        for a, b in merges.tolist():
            node_a = nodes[a]
            while node_a["parent_id"] != -1:
                node_a = nodes[node_a["parent_id"]]
            node_b = nodes[b]
            while node_b["parent_id"] != -1:
                node_b = nodes[node_b["parent_id"]]
            if node_a["nid"] == node_b["nid"]:
                continue
            else:
                # merge if threshold allows
                if (node_a["depth"] + node_b["depth"]) > 0 and min(
                    node_a["merge_threshold"], node_b["merge_threshold"]
                ) == current_thres:
                    merge_to = None
                    merge_from = None
                    if node_a["nid"] < node_b["nid"]:
                        merge_from = node_a
                        merge_to = node_b
                    if node_a["nid"] > node_b["nid"]:
                        merge_from = node_b
                        merge_to = node_a
                    merge_to["depth"] = max(merge_to["depth"], merge_from["depth"])
                    merge_to["weight"] += merge_from["weight"]
                    merge_to["children_ids"] += (
                        merge_from["children_ids"]
                        if merge_from["depth"] > 0
                        else [merge_from["nid"]]
                    )
                    for cid in merge_from["children_ids"]:
                        nodes[cid]["parent_id"] = merge_to["nid"]
                    merge_from["parent_id"] = merge_to["nid"]
                # else new node
                else:
                    new_nid = len(nodes)
                    new_node = {
                        "nid": new_nid,
                        "parent_id": -1,
                        "depth": max(node_a["depth"], node_b["depth"]) + 1,
                        "weight": node_a["weight"] + node_b["weight"],
                        "children": [],
                        "children_ids": [node_a["nid"], node_b["nid"]],
                        "example_ids": [],
                        "merge_threshold": current_thres,
                    }
                    node_a["parent_id"] = new_nid
                    node_b["parent_id"] = new_nid
                    nodes += [new_node]
    return nodes


def finalize_node(node, nodes, min_cluster_size):
    """Post-process nodes to sort children by descending weight,
    get full list of leaves in the sub-tree, and direct links
    to the cildren nodes, then recurses to all children.

    Nodes with fewer than `min_cluster_size` descendants are collapsed
    into a single leaf.
    """
    node["children"] = sorted(
        [
            finalize_node(nodes[cid], nodes, min_cluster_size)
            for cid in node["children_ids"]
        ],
        key=lambda x: x["weight"],
        reverse=True,
    )
    if node["depth"] > 0:
        node["example_ids"] = [
            eid for child in node["children"] for eid in child["example_ids"]
        ]
    node["children"] = [
        child for child in node["children"] if child["weight"] >= min_cluster_size
    ]
    assert node["weight"] == len(node["example_ids"]), print(node)
    return node


def fast_cluster(
    embeddings,
    batch_size=1000,
    approx_neighbors=1000,
    min_cluster_size=10,
    low_thres=0.5,
):
    """
    Computes an approximate hierarchical clustering based on example
    embeddings. The join criterion is min clustering, i.e. two clusters
    are joined if any pair of their descendants are closer than a threshold

    The approximate comes from the fact that only the `approx_neighbors` nearest
    neighbors of an example are considered for merges
    """
    batch_size = min(embeddings.shape[0], batch_size)
    all_merges, all_merge_scores = prepare_merges(
        embeddings, batch_size, approx_neighbors, low_thres
    )
    # prepare leaves
    nodes = [
        {
            "nid": nid,
            "parent_id": -1,
            "depth": 0,
            "weight": 1,
            "children": [],
            "children_ids": [],
            "example_ids": [nid],
            "merge_threshold": 1.0,
        }
        for nid in range(embeddings.shape[0])
    ]
    # one level per threshold range
    for i in range(10):
        p_thres = 1 - i * 0.05
        c_thres = 0.95 - i * 0.05
        nodes = merge_nodes(nodes, c_thres, p_thres, all_merges, all_merge_scores)
    # make root
    root_children = [
        node
        for node in nodes
        if node["parent_id"] == -1 and node["weight"] >= min_cluster_size
    ]
    root = {
        "nid": len(nodes),
        "parent_id": -1,
        "depth": max([node["depth"] for node in root_children]) + 1,
        "weight": sum([node["weight"] for node in root_children]),
        "children": [],
        "children_ids": [node["nid"] for node in root_children],
        "example_ids": [],
        "merge_threshold": -1.0,
    }
    nodes += [root]
    for node in root_children:
        node["parent_id"] = root["nid"]
    # finalize tree
    tree = finalize_node(root, nodes, min_cluster_size)
    node_list = []

    def rec_map_nodes(node, node_list):
        node_list += [node]
        for child in node["children"]:
            rec_map_nodes(child, node_list)

    rec_map_nodes(tree, node_list)
    # get centroids and distances
    for node in node_list:
        node_embeds = embeddings[node["example_ids"]]
        node["centroid"] = node_embeds.sum(dim=0)
        node["centroid"] /= node["centroid"].norm()
        node["centroid_dot_prods"] = torch.mv(node_embeds, node["centroid"])
        node["sorted_examples_centroid"] = sorted(
            [
                (eid, edp.item())
                for eid, edp in zip(node["example_ids"], node["centroid_dot_prods"])
            ],
            key=lambda x: x[1],
            reverse=True,
        )
    return node_list


def make_tree_plot(node_list, text_dset, text_field_name):
    """
    Makes a graphical representation of the tree encoded
    in node-list. The hover label for each node shows the number
    of descendants and the 5 examples that are closest to the centroid
    """
    nid_map = dict([(node["nid"], nid) for nid, node in enumerate(node_list)])

    for nid, node in enumerate(node_list):
        # get list of
        node_examples = {}
        for sid, score in node["sorted_examples_centroid"]:
            node_examples[text_dset[sid][text_field_name]] = score
            if len(node_examples) >= 5:
                break
        node["label"] = node.get(
            "label",
            f"{nid:2d} - {node['weight']:5d} items <br>"
            + "<br>".join(
                [
                    f" {score:.2f} > {txt[:64]}" + ("..." if len(txt) >= 63 else "")
                    for txt, score in node_examples.items()
                ]
            ),
        )

    # make plot nodes
    labels = [node["label"] for node in node_list]

    root = node_list[0]
    root["X"] = 0
    root["Y"] = 0

    def rec_make_coordinates(node):
        total_weight = 0
        add_weight = len(node["example_ids"]) - sum(
            [child["weight"] for child in node["children"]]
        )
        for child in node["children"]:
            child["X"] = node["X"] + total_weight
            child["Y"] = node["Y"] - 1
            total_weight += child["weight"] + add_weight / len(node["children"])
            rec_make_coordinates(child)

    rec_make_coordinates(root)

    E = []  # list of edges
    Xn = []
    Yn = []
    Xe = []
    Ye = []
    for nid, node in enumerate(node_list):
        Xn += [node["X"]]
        Yn += [node["Y"]]
        for child in node["children"]:
            E += [(nid, nid_map[child["nid"]])]
            Xe += [node["X"], child["X"], None]
            Ye += [node["Y"], child["Y"], None]

    # make figure
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=Xe,
            y=Ye,
            mode="lines",
            line=dict(color="rgb(210,210,210)", width=1),
            hoverinfo="none",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=Xn,
            y=Yn,
            mode="markers",
            name="nodes",
            marker=dict(
                symbol="circle-dot",
                size=18,
                color="#6175c1",
                line=dict(color="rgb(50,50,50)", width=1)
                # '#DB4551',
            ),
            text=labels,
            hoverinfo="text",
            opacity=0.8,
        )
    )
    return fig
