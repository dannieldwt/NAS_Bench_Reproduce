import os, sys, time, glob, random, argparse
import numpy as np, collections
from copy import deepcopy
from pathlib import Path
import torch
import torch.nn as nn
from torch.distributions import Categorical
from models.genotypes import Structure as CellStructure
from algorithm.algo_base import AlgorithmBase

class Random(AlgorithmBase):
    def __init__(self, xargs, logger):
        super(Random, self).__init__()
        self.max_nodes = xargs.max_nodes;
        self.op_names = xargs.search_space;
        self.logger = logger


    def generate_arch(self):
        genotypes = []
        for i in range(self.max_nodes):
            xlist = []
            for j in range(i):
                node_str = '{:}<-{:}'.format(i, j)
                opname = random.choice(self.op_names)
                xlist.append((opname, j))
            genotypes.append(tuple(xlist))

        return CellStructure(genotypes)

    def optimize(self, reward=0):
        pass