from utils.utils import prepare_seed, prepare_logger,\
    dict2config
from .supernet_enas import TinyNetworkENAS

nas201_super_nets = {
    "ENAS": TinyNetworkENAS,
}

# Cell-based NAS Models
def get_cell_based_tiny_net(config):
    if isinstance(config, dict):
        config = dict2config(config, None)  # to support the argument being a dict
    super_type = getattr(config, "super_type", "basic")
    group_names = ["DARTS-V1", "DARTS-V2", "GDAS", "SETN", "ENAS", "RANDOM", "generic"]
    if super_type == "basic" and config.name in group_names:
        try:
            return nas201_super_nets[config.name](
                config.C,
                config.N,
                config.max_nodes,
                config.num_classes,
                config.space,
                config.affine,
                config.track_running_stats,
            )
        except:
            return nas201_super_nets[config.name](
                config.C, config.N, config.max_nodes, config.num_classes, config.space
            )
    else:
        raise ValueError("invalid network name : {:}".format(config.name))