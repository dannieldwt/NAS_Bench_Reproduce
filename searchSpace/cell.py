import math, random, torch
import warnings
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from .cell_operations import OPS

class NAS201SearchCell(nn.Module):
    def __init__(
        self,
        C_in,
        C_out,
        stride,
        max_nodes,
        op_names,
        affine=False,
        track_running_stats=True,
    ):
        super(NAS201SearchCell, self).__init__()

        self.op_names = deepcopy(op_names)
        self.edges = nn.ModuleDict()
        self.max_nodes = max_nodes
        self.in_dim = C_in
        self.out_dim = C_out
        for i in range(1, max_nodes):
            for j in range(i):
                node_str = "{:}<-{:}".format(i, j)
                if j == 0:
                    xlists = [
                        OPS[op_name](C_in, C_out, stride, affine, track_running_stats)
                        for op_name in op_names
                    ]
                else:
                    xlists = [
                        OPS[op_name](C_in, C_out, 1, affine, track_running_stats)
                        for op_name in op_names
                    ]
                self.edges[node_str] = nn.ModuleList(xlists)
        self.edge_keys = sorted(list(self.edges.keys()))
        self.edge2index = {key: i for i, key in enumerate(self.edge_keys)}
        self.num_edges = len(self.edges)

    def extra_repr(self):
        string = "info :: {max_nodes} nodes, inC={in_dim}, outC={out_dim}".format(
            **self.__dict__
        )
        return string

    def forward(self, inputs, weightss):
        nodes = [inputs]
        for i in range(1, self.max_nodes):
            inter_nodes = []
            for j in range(i):
                node_str = "{:}<-{:}".format(i, j)
                weights = weightss[self.edge2index[node_str]]
                inter_nodes.append(
                    sum(
                        layer(nodes[j]) * w
                        for layer, w in zip(self.edges[node_str], weights)
                    )
                )
            nodes.append(sum(inter_nodes))
        return nodes[-1]

    # forward with a specific structure
    def forward_dynamic(self, inputs, structure):
        nodes = [inputs]
        for i in range(1, self.max_nodes):
            cur_op_node = structure.nodes[i - 1]
            inter_nodes = []
            for op_name, j in cur_op_node:
                node_str = "{:}<-{:}".format(i, j)
                op_index = self.op_names.index(op_name)
                inter_nodes.append(self.edges[node_str][op_index](nodes[j]))
            nodes.append(sum(inter_nodes))
        return nodes[-1]