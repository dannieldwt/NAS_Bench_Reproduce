#coding:utf-8
# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import os, sys, time, glob, random, argparse
import numpy as np, collections
from copy import deepcopy
import torch
import torch.nn as nn
from pathlib import Path
import yaml
import sys
parent_path = os.path.realpath('..') # 取决于我python命令的当前目录
if parent_path not in sys.path:
    sys.path.append(parent_path)
    print(sys.path)
from utils.utils import prepare_seed, prepare_logger,\
    load_config, get_search_spaces, train_and_eval, time_string
from nas_201_api import NASBench201API as API
from models.genotypes import Structure as CellStructure
from utils.REA.utils import Model
#
# api = API('./bench/NAS-Bench-201-v1_1-096897.pth')

# Suppose you are trying to load pre-trained resnet model in directory- models\resnet
os.environ['TORCH_HOME'] = '/mnt/cephfs/home/dengweitao/codes/NAS_Bench_201/dataset'

def random_architecture_func(max_nodes, op_names):
    genotypes = []
    for i in range(1, max_nodes):
        xlist = []
        for j in range(i):
            node_str = "{:}<-{:}".format(i, j)
            op_name = random.choice(op_names)
            xlist.append((op_name, j))
        genotypes.append(tuple(xlist))
    return CellStructure(genotypes)

def mutate_arch_func(parent_arch, op_names):

    child_arch = deepcopy(parent_arch)
    node_id = random.randint(0, len(child_arch.nodes) - 1)
    node_info = list(child_arch.nodes[node_id])
    snode_id = random.randint(0, len(node_info) - 1)
    xop = random.choice(op_names)
    while xop == node_info[snode_id][0]:
        xop = random.choice(op_names)
    node_info[snode_id] = (xop, node_info[snode_id][1])
    child_arch.nodes[node_id] = tuple(node_info)
    return child_arch

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
        # config_path = "../config/algo.config"
        config = load_config(xargs['config_path'], None, logger)
        extra_info = {"config": config, "train_loader": None, "valid_loader": None}
        logger.log("||||||| {:10s} ||||||| Config={:}".format(xargs['dataset'], config))

    # step 3: 搭建搜索空间
    search_space = get_search_spaces("cell", xargs['search_space_name'])
    xargs['search_space'] = search_space

    x_start_time = time.time()
    logger.log(
        "Will start searching with time budget of {:} s.".format(xargs['time_budget'])
    )

    population = collections.deque()
    history, total_time_cost, total_query = ([], 0, 0) 
    # Initialize the population with random models.
    while len(population) < xargs['ea_population']:
        model = Model()
        model.arch = random_architecture_func(xargs['max_nodes'], xargs['search_space'])
        model.accuracy, time_cost = train_and_eval(model.arch, nas_bench, extra_info, dataname)
        total_query += 1
        population.append(model)
        history.append(model)
        total_time_cost += time_cost

    while total_time_cost < xargs["time_budget"]:
        # Sample randomly chosen models from the current population.
        start_time, sample = time.time(), []
        while len(sample) < xargs['ea_sample_size']:
            candidate = random.choice(list(population))
            sample.append(candidate)

        parent = max(sample, key=lambda i: i.accuracy)

        child = Model()
        child.arch = mutate_arch_func(parent.arch, xargs['search_space'])
        print("child arch: ", child.arch)
        total_time_cost += time.time() - start_time
        print("evaluate before child arch: ", child.arch)
        print("evaluate before nas_bench: ", nas_bench)
        child.accuracy, time_cost = train_and_eval(child.arch, nas_bench, extra_info, dataname)
        total_query += 1
        if total_time_cost + time_cost > xargs['time_budget']:  
            break
        else:
            total_time_cost += time_cost
        population.append(child)
        history.append(child)

        # for REA
        population.popleft()

    logger.log("algorithm query: {:}".format(total_query))
    logger.log(
        "{:} regularized_evolution finish with history of {:} arch with {:.1f} s (real-cost={:.2f} s).".format(
            time_string(), len(history), total_time_cost, time.time() - x_start_time
        )
    )
    best_arch = max(history, key=lambda i: i.accuracy)
    best_arch = best_arch.arch
    logger.log("{:} best arch is {:}".format(time_string(), best_arch))

    info = nas_bench.query_by_arch(best_arch, "200")
    if info is None:
        logger.log("Did not find this architecture : {:}.".format(best_arch))
    else:
        logger.log("{:}".format(info))
    logger.log("-" * 100)
    logger.close()
    return logger.log_dir, nas_bench.query_index_by_arch(best_arch)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser("The NAS-BENCH-201 Algorithm")
    parser.add_argument("--config_file", type=str, default='../config/REA/R_EA.yaml', help="config file path")
    args = parser.parse_args()

    file = open(args.config_file, 'r', encoding="utf-8")
    config = yaml.load(file)
    print("config {}".format(config))
    print("test: {}".format(config['algorithm']))
    config['save_dir'] = "../output/{}-{}-{}".format(config['algorithm'], config['dataset'], config['exp_name'])

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

    if config['rand_seed'] < 0:
        save_dir, all_indexes, num = None, [], 10
        for i in range(num):
            print("{:} : {:03d}/{:03d}".format(time_string(), i, num))
            config['rand_seed'] = random.randint(1, 100000)
            save_dir, index = main(config, nas_bench)
            all_indexes.append(index)
        print("all_indexes: ", all_indexes)
        torch.save(all_indexes, save_dir / "results.pth")
    else:
        main(config, nas_bench)