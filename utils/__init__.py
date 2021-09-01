import os, sys, time, glob, random, argparse
import numpy as np, collections
from copy import deepcopy
from pathlib import Path
import torch
import torch.nn as nn
from torch.distributions import Categorical
from models.genotypes import Structure as CellStructure

class Policy(nn.Module):
    def __init__(self, max_nodes, search_space):
        super(Policy, self).__init__()
        self.max_nodes = max_nodes
        self.search_space = search_space
        self.edge2index = {}
        for i in range(1, max_nodes):
            for j in range(i):
                node_str = "{:}<-{:}".format(i, j)
                self.edge2index[node_str] = len(self.edge2index)
        self.arch_parameters = nn.Parameter(
            1e-3 * torch.randn(len(self.edge2index), len(search_space))
        )

    def generate_arch(self, actions):
        genotypes = []
        for i in range(1, self.max_nodes):
            xlist = []
            for j in range(i):
                node_str = "{:}<-{:}".format(i, j)
                op_name = self.search_space[actions[self.edge2index[node_str]]]
                xlist.append((op_name, j))
            genotypes.append(tuple(xlist))
        return CellStructure(genotypes)

    def genotype(self):
        genotypes = []
        for i in range(1, self.max_nodes):
            xlist = []
            for j in range(i):
                node_str = "{:}<-{:}".format(i, j)
                with torch.no_grad():
                    weights = self.arch_parameters[self.edge2index[node_str]]
                    op_name = self.search_space[weights.argmax().item()]
                xlist.append((op_name, j))
            genotypes.append(tuple(xlist))
        return CellStructure(genotypes)

    def forward(self):
        alphas = nn.functional.softmax(self.arch_parameters, dim=-1)
        return alphas

class ExponentialMovingAverage(object):
    """Class that maintains an exponential moving average."""

    def __init__(self, momentum):
        self._numerator = 0
        self._denominator = 0
        self._momentum = momentum

    def update(self, value):
        self._numerator = (
            self._momentum * self._numerator + (1 - self._momentum) * value
        )
        self._denominator = self._momentum * self._denominator + (1 - self._momentum)

    def value(self):
        """Return the current value of the moving average"""
        return self._numerator / self._denominator


def select_action(policy):
    probs = policy()
    m = Categorical(probs)
    action = m.sample()
    # policy.saved_log_probs.append(m.log_prob(action))
    return m.log_prob(action), action.cpu().tolist()


# class REINFORCE(AlgorithmBase):
#     def __init__(self, xargs, logger):
#         super(REINFORCE, self).__init__()
#         self.args = xargs
#         self.policy = Policy(xargs['max_nodes'], xargs['search_space'])
#         self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=xargs['learning_rate'])
#         self.eps = np.finfo(np.float32).eps.item()
#         self.baseline = ExponentialMovingAverage(xargs['EMA_momentum'])
#         self.logger = logger

#         logger.log("policy    : {:}".format(self.policy))
#         logger.log("optimizer : {:}".format(self.optimizer))
#         logger.log("eps       : {:}".format(self.eps))

#     def generate_arch(self):
#         self.log_prob, action = select_action(self.policy)
#         arch = self.policy.generate_arch(action)
#         return arch

#     def optimize(self, reward=0):
#         self.baseline.update(reward)
#         # calculate loss
#         policy_loss = (-self.log_prob * (reward - self.baseline.value())).sum()
#         self.optimizer.zero_grad()
#         policy_loss.backward()
#         self.optimizer.step()
#         self.logger.log(
#             "REINFORCE : average-reward={:.3f} : policy_loss={:.4f} : {:}".format(
#                  self.baseline.value(), policy_loss.item(), self.policy.genotype()
#             )
#         )