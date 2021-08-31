import os, sys, torch, random, PIL, copy, numpy as np
from os import path as osp
from shutil import copyfile
from collections import namedtuple
from copy import deepcopy
from PIL import Image
import time
import json
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.utils.data as data
from nas_201_api import NASBench201API as API

support_types = ("str", "int", "bool", "float", "none")

Dataset2Class = {
    "cifar10": 10,
    "cifar100": 100,
    "imagenet-1k-s": 1000,
    "imagenet-1k": 1000,
    "ImageNet16": 1000,
    "ImageNet16-150": 150,
    "ImageNet16-120": 120,
    "ImageNet16-200": 200,
}


def prepare_seed(rand_seed):
    random.seed(rand_seed)
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)
    torch.cuda.manual_seed_all(rand_seed)

def prepare_logger(xargs):
    args = copy.deepcopy(xargs)
    from utils.logger import Logger

    logger = Logger(args['save_dir'], args['rand_seed'])
    logger.log("Main Function with logger : {:}".format(logger))
    logger.log("Arguments : -------------------------------")
    logger.log("Python  Version  : {:}".format(sys.version.replace("\n", " ")))
    logger.log("Pillow  Version  : {:}".format(PIL.__version__))
    logger.log("PyTorch Version  : {:}".format(torch.__version__))
    logger.log("cuDNN   Version  : {:}".format(torch.backends.cudnn.version()))
    logger.log("CUDA available   : {:}".format(torch.cuda.is_available()))
    logger.log("CUDA GPU numbers : {:}".format(torch.cuda.device_count()))
    logger.log(
        "CUDA_VISIBLE_DEVICES : {:}".format(
            os.environ["CUDA_VISIBLE_DEVICES"]
            if "CUDA_VISIBLE_DEVICES" in os.environ
            else "None"
        )
    )
    return logger

def save_checkpoint(state, filename, logger):
    if osp.isfile(filename):
        if hasattr(logger, "log"):
            logger.log(
                "Find {:} exist, delete is at first before saving".format(filename)
            )
        os.remove(filename)
    torch.save(state, filename)
    assert osp.isfile(
        filename
    ), "save filename : {:} failed, which is not found.".format(filename)
    if hasattr(logger, "log"):
        logger.log("save checkpoint into {:}".format(filename))
    return filename


def copy_checkpoint(src, dst, logger):
    if osp.isfile(dst):
        if hasattr(logger, "log"):
            logger.log("Find {:} exist, delete is at first before saving".format(dst))
        os.remove(dst)
    copyfile(src, dst)
    if hasattr(logger, "log"):
        logger.log("copy the file from {:} into {:}".format(src, dst))

def convert_param(original_lists):
    assert isinstance(original_lists, list), "The type is not right : {:}".format(
        original_lists
    )
    ctype, value = original_lists[0], original_lists[1]
    assert ctype in support_types, "Ctype={:}, support={:}".format(ctype, support_types)
    is_list = isinstance(value, list)
    if not is_list:
        value = [value]
    outs = []
    for x in value:
        if ctype == "int":
            x = int(x)
        elif ctype == "str":
            x = str(x)
        elif ctype == "bool":
            x = bool(int(x))
        elif ctype == "float":
            x = float(x)
        elif ctype == "none":
            if x.lower() != "none":
                raise ValueError(
                    "For the none type, the value must be none instead of {:}".format(x)
                )
            x = None
        else:
            raise TypeError("Does not know this type : {:}".format(ctype))
        outs.append(x)
    if not is_list:
        outs = outs[0]
    return outs


def load_config(path, extra, logger):
    path = str(path)
    if hasattr(logger, "log"):
        logger.log(path)
    assert os.path.exists(path), "Can not find {:}".format(path)

    # Reading data back
    with open(path, "r") as f:
        data = json.load(f)
    content = {k: convert_param(v) for k, v in data.items()}

    assert extra is None or isinstance(
        extra, dict
    ), "invalid type of extra : {:}".format(extra)
    if isinstance(extra, dict):
        content = {**content, **extra}

    Arguments = namedtuple("Configure", " ".join(content.keys()))
    content = Arguments(**content)

    if hasattr(logger, "log"):
        logger.log("{:}".format(content))
    return content

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

def time_string():
    ISOTIMEFORMAT = "%Y-%m-%d %X"
    string = "[{:}]".format(time.strftime(ISOTIMEFORMAT, time.gmtime(time.time())))
    return string

def dict2config(xdict, logger):
    assert isinstance(xdict, dict), "invalid type : {:}".format(type(xdict))
    Arguments = namedtuple("Configure", " ".join(xdict.keys()))
    content = Arguments(**xdict)
    if hasattr(logger, "log"):
        logger.log("{:}".format(content)) 
    return content

def convert_secs2time(epoch_time, return_str=False):
    need_hour = int(epoch_time / 3600)
    need_mins = int((epoch_time - 3600 * need_hour) / 60)
    need_secs = int(epoch_time - 3600 * need_hour - 60 * need_mins)
    if return_str:
        str = "[{:02d}:{:02d}:{:02d}]".format(need_hour, need_mins, need_secs)
        return str
    else:
        return need_hour, need_mins, need_secs

def obtain_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

    
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return "{name}(val={val}, avg={avg}, count={count})".format(
            name=self.__class__.__name__, **self.__dict__
        )