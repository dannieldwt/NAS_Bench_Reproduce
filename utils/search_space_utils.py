from os import path as osp
from shutil import copyfile
from collections import namedtuple
from copy import deepcopy
from nas_201_api import NASBench201API as API

support_types = ("str", "int", "bool", "float", "none")

def get_search_spaces(xtype, name):
    if xtype == "cell" or xtype == "tss":  # The topology search space.
        from models.cell_operations import SearchSpaceNames

        assert name in SearchSpaceNames, "invalid name [{:}] in {:}".format(
            name, SearchSpaceNames.keys()
        )
        return SearchSpaceNames[name]
    else:
        raise ValueError("invalid search-space type is {:}".format(xtype))


def train_and_eval(arch, nas_bench, extra_info, dataname="cifar10-valid", use_012_epoch_training=True):

    if use_012_epoch_training and nas_bench is not None:
        arch_index = nas_bench.query_index_by_arch(arch)
        assert arch_index >= 0, "can not find this arch : {:}".format(arch)
        info = nas_bench.get_more_info(
            arch_index, dataname, iepoch=None, hp="12", is_random=True
        )
        valid_acc, time_cost = (
            info["valid-accuracy"],
            info["train-all-time"] + info["valid-per-time"],
        )
        # _, valid_acc = info.get_metrics('cifar10-valid', 'x-valid' , 25, True) # use the validation accuracy after 25 training epochs
    elif not use_012_epoch_training and nas_bench is not None:
        # Please contact me if you want to use the following logic, because it has some potential issues.
        # Please use `use_012_epoch_training=False` for cifar10 only.
        # It did return values for cifar100 and ImageNet16-120, but it has some potential issues. (Please email me for more details)
        arch_index, nepoch = nas_bench.query_index_by_arch(arch), 25
        assert arch_index >= 0, "can not find this arch : {:}".format(arch)
        xoinfo = nas_bench.get_more_info(
            arch_index, "cifar10-valid", iepoch=None, hp="12"
        )
        xocost = nas_bench.get_cost_info(arch_index, "cifar10-valid", hp="200")
        info = nas_bench.get_more_info(
            arch_index, dataname, nepoch, hp="200", is_random=True
        )  # use the validation accuracy after 25 training epochs, which is used in our ICLR submission (not the camera ready).
        cost = nas_bench.get_cost_info(arch_index, dataname, hp="200")
        # The following codes are used to estimate the time cost.
        # When we build NAS-Bench-201, architectures are trained on different machines and we can not use that time record.
        # When we create checkpoints for converged_LR, we run all experiments on 1080Ti, and thus the time for each architecture can be fairly compared.
        nums = {
            "ImageNet16-120-train": 151700,
            "ImageNet16-120-valid": 3000,
            "cifar10-valid-train": 25000,
            "cifar10-valid-valid": 25000,
            "cifar100-train": 50000,
            "cifar100-valid": 5000,
        }
        estimated_train_cost = (
            xoinfo["train-per-time"]
            / nums["cifar10-valid-train"]
            * nums["{:}-train".format(dataname)]
            / xocost["latency"]
            * cost["latency"]
            * nepoch
        )
        estimated_valid_cost = (
            xoinfo["valid-per-time"]
            / nums["cifar10-valid-valid"]
            * nums["{:}-valid".format(dataname)]
            / xocost["latency"]
            * cost["latency"]
        )
        try:
            valid_acc, time_cost = (
                info["valid-accuracy"],
                estimated_train_cost + estimated_valid_cost,
            )
        except:
            valid_acc, time_cost = (
                info["valtest-accuracy"],
                estimated_train_cost + estimated_valid_cost,
            )
    else:
        # train a model from scratch.
        raise ValueError("NOT IMPLEMENT YET")
    return valid_acc, time_cost


    
