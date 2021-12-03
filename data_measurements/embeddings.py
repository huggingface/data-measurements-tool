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
from tqdm import tqdm

from .dataset_utils import EMBEDDING_FIELD, OUR_TEXT_FIELD


def sentence_mean_pooling(model_output, attention_mask):
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
    def __init__(self, dstats, use_cache=False):
        """Item embeddings and clustering"""
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.node_list = None
        self.nid_map = None
        self.embeddings_dset = None
        self.fig_tree = None
        self.cached_clusters = {}
        self.dstats = dstats
        self.cache_path = dstats.cache_path
        self.node_list_fid = pjoin(self.cache_path, "node_list.th")
        self.use_cache = use_cache
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            "sentence-transformers/all-mpnet-base-v2"
        )
        self.model = transformers.AutoModel.from_pretrained(
            "sentence-transformers/all-mpnet-base-v2"
        ).to(self.device)

    def make_text_embeddings(self):
        embeddings_dset_fid = pjoin(self.cache_path, "embeddings_dset")
        if self.use_cache and exists(embeddings_dset_fid):
            self.embeddings_dset = load_from_disk(embeddings_dset_fid)
        else:
            self.embeddings_dset = self.make_embeddings()
            self.embeddings_dset.save_to_disk(embeddings_dset_fid)

    def make_hierarchical_clustering(self):
        if self.use_cache and exists(self.node_list_fid):
            self.node_list = torch.load(self.node_list_fid)
        else:
            self.make_text_embeddings()
            self.node_list = self.fast_cluster(self.embeddings_dset, EMBEDDING_FIELD)
            torch.save(self.node_list, self.node_list_fid)
        self.nid_map = dict(
            [(node["nid"], nid) for nid, node in enumerate(self.node_list)]
        )
        self.fig_tree = make_tree_plot(self.node_list, self.dstats.text_dset)

    def compute_sentence_embeddings(self, sentences):
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
        def batch_embed_sentences(sentences):
            return {
                EMBEDDING_FIELD: [
                    embed.tolist()
                    for embed in self.compute_sentence_embeddings(
                        sentences[OUR_TEXT_FIELD]
                    )
                ]
            }

        text_dset_embeds = self.dstats.text_dset.map(
            batch_embed_sentences,
            batched=True,
            batch_size=32,
            remove_columns=[self.dstats.our_text_field],
        )

        return text_dset_embeds

    @staticmethod
    def prepare_merges(embeddings, batch_size, low_thres=0.5):
        top_idx_pre = torch.cat(
            [torch.LongTensor(range(embeddings.shape[0]))[:, None]] * batch_size, dim=1
        )
        top_val_all = torch.Tensor(0, batch_size)
        top_idx_all = torch.LongTensor(0, batch_size)
        n_batches = math.ceil(len(embeddings) / batch_size)
        for b in tqdm(range(n_batches)):
            cos_scores = torch.mm(
                embeddings[b * batch_size : (b + 1) * batch_size], embeddings.t()
            )
            for i in range(cos_scores.shape[0]):
                cos_scores[i, (b * batch_size) + i :] = -1
            top_val_large, top_idx_large = cos_scores.topk(
                k=batch_size, dim=-1, largest=True
            )
            top_val_all = torch.cat([top_val_all, top_val_large], dim=0)
            top_idx_all = torch.cat([top_idx_all, top_idx_large], dim=0)

        all_merges = torch.cat(
            [
                top_idx_pre[top_val_all > low_thres][:, None],
                top_idx_all[top_val_all > low_thres][:, None],
            ],
            dim=1,
        )
        all_merge_scores = top_val_all[top_val_all > low_thres]
        return (all_merges, all_merge_scores)

    @staticmethod
    def merge_nodes(nodes, current_thres, previous_thres, all_merges, all_merge_scores):
        merge_ids = (all_merge_scores <= previous_thres) * (
            all_merge_scores > current_thres
        )
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

    def finalize_node(self, node, nodes, min_cluster_size):
        node["children"] = sorted(
            [
                self.finalize_node(nodes[cid], nodes, min_cluster_size)
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
        self,
        text_dset_embeds,
        embedding_field,
        batch_size=1000,
        min_cluster_size=10,
        low_thres=0.5,
    ):
        embeddings = torch.Tensor(text_dset_embeds[embedding_field])
        batch_size = min(embeddings.shape[0], batch_size)
        all_merges, all_merge_scores = self.prepare_merges(
            embeddings, batch_size, low_thres
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
            nodes = self.merge_nodes(
                nodes, c_thres, p_thres, all_merges, all_merge_scores
            )
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
        tree = self.finalize_node(root, nodes, min_cluster_size)
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

    def find_cluster_beam(self, sentence, beam_size=20):
        """
        This function finds the `beam_size` lef clusters that are closest to the
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


def make_tree_plot(node_list, text_dset):
    nid_map = dict([(node["nid"], nid) for nid, node in enumerate(node_list)])

    for nid, node in enumerate(node_list):
        node["label"] = node.get(
            "label",
            f"{nid:2d} - {node['weight']:5d} items <br>"
            + "<br>".join(
                [
                    "> " + txt[:64] + ("..." if len(txt) >= 63 else "")
                    for txt in list(
                        set(text_dset.select(node["example_ids"])[OUR_TEXT_FIELD])
                    )[:5]
                ]
            ),
        )

    # make plot nodes
    # TODO: something more efficient than set to remove duplicates
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
