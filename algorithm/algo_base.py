import os, sys, time, glob, random, argparse
import numpy as np, collections
from copy import deepcopy
from pathlib import Path
import torch
import torch.nn as nn
from torch.distributions import Categorical
from models.genotypes import Structure as CellStructure

class AlgorithmBase(object):
    def __init__(self):
        super(AlgorithmBase, self).__init__()

    def generate_arch(self):
        pass

    def optimize(self, reward=0):
        pass