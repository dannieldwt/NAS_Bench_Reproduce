#coding:utf-8
# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import os
import numpy as np
import time
import yaml
import argparse
import random
import torch
import sys
parent_path = os.path.realpath('..') # 取决于我python命令的当前目录
if parent_path not in sys.path:
    sys.path.append(parent_path)
    print(sys.path)

from utils.utils import prepare_seed, prepare_logger,\
    load_config, get_search_spaces, train_and_eval, time_string
from nas_201_api import NASBench201API as API
from models.genotypes import Structure as CellStructure
#
# api = API('./bench/NAS-Bench-201-v1_1-096897.pth')

# Suppose you are trying to load pre-trained resnet model in directory- models\resnet
os.environ['TORCH_HOME'] = '/mnt/cephfs/home/dengweitao/codes/NAS_Bench_201/dataset'

def generate_arch(max_nodes, op_names):
    print("max_nodes: ", max_nodes)
    print("op_names: ", op_names)
    genotypes = []
    for i in range(1, max_nodes):
        xlist = []
        for j in range(i):
            node_str = '{:}<-{:}'.format(i, j)
            opname = random.choice(op_names)
            xlist.append((opname, j))
        genotypes.append(tuple(xlist))
    print("genotypes: ", genotypes)
    return CellStructure(genotypes)

def main(xargs, nas_bench):
    
    #step 1: 全局属性配置 
    assert torch.cuda.is_available(), "CUDA is not available."
    torch.backends.cudnn.enabled = True 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(xargs['workers'])
    prepare_seed(xargs['rand_seed'])
    logger = prepare_logger(xargs)

    # step 2 : 加载数据集和配置
    if xargs['dataset'] == "cifar10":
        dataname = "cifar10-valid" # 为什么要用这个名？
    else:
        dataname = xargs['dataset']

    if xargs['data_path'] is not None:
        pass
    else:
        config_path = "../config/algo.config"
        config = load_config(config_path, None, logger)
        extra_info = {"config": config, "train_loader": None, "valid_loader": None}
        logger.log("||||||| {:10s} ||||||| Config={:}".format(xargs['dataset'], config))

    # step 3: 搭建搜索空间
    search_space = get_search_spaces("cell", xargs['search_space_name'])
    xargs['search_space'] = search_space

    x_start_time = time.time()
    logger.log(
        "Will start searching with time budget of {:} s.".format(xargs['time_budget'])
    )
    total_steps, total_costs, trace, total_query = 0, 0, [], 0

    while total_costs < xargs['time_budget']:
        start_time = time.time()
        arch = generate_arch(xargs['max_nodes'], xargs['search_space'])
        reward, cost_time = train_and_eval(arch, nas_bench, extra_info, dataname)
        total_query += 1
        trace.append((reward, arch))
        # accumulate time
        if total_costs + cost_time < xargs['time_budget']:
            total_costs += cost_time
        else:
            break
        # accumulate time
        total_costs += time.time() - start_time
        total_steps += 1

    logger.log("algorithm query: {:}".format(total_query))
    best_arch = max(trace, key=lambda x: x[0])[1]
    logger.log(
        "algorithm {:} finish with {:} steps and {:.1f} s (real cost={:.3f}).".format(xargs['time_budget'],
            total_steps, total_costs, time.time() - x_start_time
        )
    )

    best_index = nas_bench.query_index_by_arch(best_arch)
    best_valid_metrics = nas_bench.get_more_info(
        best_arch, dataname, iepoch=None, hp="200", is_random=True
    )
    best_test_metrics = nas_bench.get_more_info(
        best_arch, "cifar10", iepoch=None, hp="200", is_random=True
    )
    # best_valid_acc = best_metrics["valid-accuracy"]
    logger.log("look at the output type for metric info: {:}".format(type(best_valid_metrics)))
    logger.log(best_valid_metrics)
    logger.log("look at the output type for metric info: {:}".format(type(best_test_metrics)))
    logger.log(best_test_metrics)
    best_valid_acc = best_valid_metrics['valid-accuracy']
    best_test_acc = best_test_metrics['test-accuracy']

    info = nas_bench.query_by_arch(best_arch, "200")
    if info is None:
        logger.log("Did not find this architecture : {:}.".format(best_arch))
    else:
        logger.log("{:}".format(info))
    logger.log("-" * 100)
    logger.close()
    return logger.log_dir, nas_bench.query_index_by_arch(best_arch), best_valid_acc, best_test_acc, total_query


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser("The NAS-BENCH-201 Algorithm")
    parser.add_argument("--config_file", type=str, default='../config/random/random.yaml', help="config file path")
    args = parser.parse_args()

    file = open(args.config_file, 'r', encoding="utf-8")
    config = yaml.load(file)
    print("config {}".format(config))
    print("test: {}".format(config['algorithm']))
    config['save_dir'] = "../output/{}-{}-{}".format(config['algorithm'], config['dataset'], config['exp_name'])
    print("config save dir: {}".format(config['save_dir']))


    # if args.rand_seed is None or args.rand_seed < 0: args.rand_seed = random.randint(1, 100000)
    if config['arch_nas_dataset'] is None or not os.path.isfile(config['arch_nas_dataset']):
        nas_bench = None
    else:
        print(
            "{:} build NAS-Benchmark-API from {:}".format(
                time_string(), config['arch_nas_dataset']
            )
        )
        nas_bench = API(config['arch_nas_dataset'])

    print("nas_bench: ", nas_bench)

    if config['rand_seed'] < 0:
        save_dir, all_indexes, num, valid_total, test_total, query_total = None, [], 10, 0, 0, 0
        for i in range(num):
            print("{:} : {:03d}/{:03d}".format(time_string(), i, num))
            config['rand_seed'] = random.randint(1, 100000)
            save_dir, index, valid, test, query = main(config, nas_bench)
            valid_total += valid
            test_total += test
            query_total += query
            all_indexes.append(index)
        print("Average accuracy for cifar10: {:} for valid and {:} for test and query {:}"
        .format(valid_total / num, test_total / num, query_total / num))   
        torch.save(all_indexes, save_dir / "results.pth")
    else:
        main(config, nas_bench)