from algorithm.random import Random
from algorithm.reinforce import REINFORCE


def build_algo(xargs, logger):
    maps = dict(
        REINFORCE=REINFORCE,
        RANDOM=Random,
    )
    logger.log("test: {}".format(xargs))
    return maps[xargs['algorithm']](xargs, logger)