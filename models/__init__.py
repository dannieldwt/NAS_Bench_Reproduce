from utils.utils import dict2config
from typing import List, Text

def get_cell_based_super_net(config):
    if isinstance(config, dict):
        config = dict2config(config, None)
    super_type = getattr(config, "super_type", "basic")
    group_names = ["DARTS-V1", "DARTS-V2", "GDAS", "SETN", "ENAS", "RANDOM", "generic"]
    if super_type == 'basic' and config.name in group_names: 
        pass


def get_search_spaces(xtype, name) -> List[Text]:
    if xtype == "cell" or xtype == "tss":  # The topology search space.
        from .cell_operations import SearchSpaceNames

        assert name in SearchSpaceNames, "invalid name [{:}] in {:}".format(
            name, SearchSpaceNames.keys()
        )
        return SearchSpaceNames[name]
    elif xtype == "sss":  # The size search space.
        if name in ["nats-bench", "nats-bench-size"]:
            return {"candidates": [8, 16, 24, 32, 40, 48, 56, 64], "numbers": 5}
        else:
            raise ValueError("Invalid name : {:}".format(name))
    else:
        raise ValueError("invalid search-space type is {:}".format(xtype))
