from algorithm.random import Random
from reinforce import REINFORCE


def build_algo(xargs, logger):
    maps = dict(
        REINFORCE=REINFORCE,
        RANDOM=Random,
    )
    return maps[xargs.algorithm](xargs, logger)