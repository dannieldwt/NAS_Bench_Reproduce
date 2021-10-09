#coding:utf-8
# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import torch
import os
import numpy as np
import time
import yaml
import argparse
import random
import numpy as np
from copy import deepcopy
import torch.nn as nn

from utils.utils import prepare_seed, prepare_logger,\
    load_config, get_search_spaces, time_string, \
    dict2config, convert_secs2time, AverageMeter, obtain_accuracy
from controller.enas_controller import Controller
from dataset.searchDataset import get_nas_search_loaders, get_datasets

# Suppose you are trying to load pre-trained resnet model in directory- models\resnet
os.environ['TORCH_HOME'] = '/mnt/cephfs/home/dengweitao/codes/NAS_Bench_201/dataset'

def load_yaml_config(file_path):
    file = open(file_path, 'r', encoding="utf-8")
    config = yaml.load(file)
    config['save_dir'] = "./output/{}-{}-{}".format(config['algorithm'], config['dataset'], config['exp_name'])
    return config

def train_super_net(
    xloader,
    super_net,
    controller,
    criterion,
    scheduler,
    optimizer,
    epoch_str,
    print_freq,
    logger,
):
    data_time, batch_time = AverageMeter(), AverageMeter()
    losses, top1s, top5s, xend = (
        AverageMeter(),
        AverageMeter(),
        AverageMeter(),
        time.time(),
    )
    super_net.train()
    controller.eval()

    for step, (input, target) in enumerate(xloader):
        scheduler.update(None, 1.0 * step / len(xloader)) #参数格式？
        target = target.cuda(non_blocking=True)
        data_time.update(time.time() - xend)  

        with torch.no_grad():
            _, _, sample_arch = controller()

        optimizer.zero_grad()
        super_net.module.update_arch(sampled_arch)
        _, logits = super_net(input) #返回值格式？
        loss = criterion(logits, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(super_net.parameters(), 5) #作用？
        optimizer.step()

        # record
        prec1, prec5 = obtain_accuracy(logits.data, targets.data, topk=(1, 5)) # 该函数定义？
        losses.update(loss.item(), inputs.size(0))
        top1s.update(prec1.item(), inputs.size(0))
        top5s.update(prec5.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - xend)
        xend = time.time()

        if step % print_freq == 0 or step + 1 == len(xloader):
            Sstr = (
                "*Train-Shared-CNN* "
                + time_string()
                + " [{:}][{:03d}/{:03d}]".format(epoch_str, step, len(xloader))
            )
            Tstr = "Time {batch_time.val:.2f} ({batch_time.avg:.2f}) Data {data_time.val:.2f} ({data_time.avg:.2f})".format(
                batch_time=batch_time, data_time=data_time
            )
            Wstr = "[Loss {loss.val:.3f} ({loss.avg:.3f})  Prec@1 {top1.val:.2f} ({top1.avg:.2f}) Prec@5 {top5.val:.2f} ({top5.avg:.2f})]".format(
                loss=losses, top1=top1s, top5=top5s
            )
            logger.log(Sstr + " " + Tstr + " " + Wstr)
    return losses.avg, top1s.avg, top5s.avg

def train_controller(
    xloader,
    shared_cnn,
    controller,
    criterion,
    optimizer,
    config,
    epoch_str,
    print_freq,
    logger,
):
    data_time, batch_time = AverageMeter(), AverageMeter()
    (
        GradnormMeter,
        LossMeter,
        ValAccMeter,
        EntropyMeter,
        BaselineMeter,
        RewardMeter,
        xend,
    ) = (
        AverageMeter(),
        AverageMeter(),
        AverageMeter(),
        AverageMeter(),
        AverageMeter(),
        AverageMeter(),
        time.time(),
    )

    shared_cnn.eval()
    controller.train()
    controller.zero_grad()
    # for step, (inputs, targets) in enumerate(xloader):
    loader_iter = iter(xloader)
    for step in range(config.ctl_train_steps * config.ctl_num_aggre):  #两个参数的意义？
        try:
            inputs, targets = next(loader_iter)
        except:
            loader_iter = iter(xloader)
            inputs, targets = next(loader_iter)
        targets = targets.cuda(non_blocking=True)
        # measure data loading time
        data_time.update(time.time() - xend)

        log_prob, entropy, sampled_arch = controller()
        with torch.no_grad():
            shared_cnn.module.update_arch(sampled_arch)
            _, logits = shared_cnn(inputs)
            val_top1, val_top5 = obtain_accuracy(logits.data, targets.data, topk=(1, 5))
            val_top1 = val_top1.view(-1) / 100
        reward = val_top1 + config.ctl_entropy_w * entropy # 用加的？ 注意是计算reward not loss
        if config.baseline is None:
            baseline = val_top1
        else:
            baseline = config.baseline - (1 - config.ctl_bl_dec) * (
                config.baseline - reward
            )

        loss = -1 * log_prob * (reward - baseline)

        # account
        RewardMeter.update(reward.item())
        BaselineMeter.update(baseline.item())
        ValAccMeter.update(val_top1.item() * 100)
        LossMeter.update(loss.item())
        EntropyMeter.update(entropy.item())

        # Average gradient over controller_num_aggregate samples
        loss = loss / config.ctl_num_aggre
        loss.backward(retain_graph=True)

        # measure elapsed time
        batch_time.update(time.time() - xend)
        xend = time.time()
        if (step + 1) % config.ctl_num_aggre == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(controller.parameters(), 5.0)
            GradnormMeter.update(grad_norm)
            optimizer.step()
            controller.zero_grad()

        if step % print_freq == 0:
            Sstr = (
                "*Train-Controller* "
                + time_string()
                + " [{:}][{:03d}/{:03d}]".format(
                    epoch_str, step, config.ctl_train_steps * config.ctl_num_aggre
                )
            )
            Tstr = "Time {batch_time.val:.2f} ({batch_time.avg:.2f}) Data {data_time.val:.2f} ({data_time.avg:.2f})".format(
                batch_time=batch_time, data_time=data_time
            )
            Wstr = "[Loss {loss.val:.3f} ({loss.avg:.3f})  Prec@1 {top1.val:.2f} ({top1.avg:.2f}) Reward {reward.val:.2f} ({reward.avg:.2f})] Baseline {basel.val:.2f} ({basel.avg:.2f})".format(
                loss=LossMeter,
                top1=ValAccMeter,
                reward=RewardMeter,
                basel=BaselineMeter,
            )
            Estr = "Entropy={:.4f} ({:.4f})".format(EntropyMeter.val, EntropyMeter.avg)
            logger.log(Sstr + " " + Tstr + " " + Wstr + " " + Estr)

    return (
        LossMeter.avg,
        ValAccMeter.avg,
        BaselineMeter.avg,
        RewardMeter.avg,
        baseline.item(),
    )


def get_best_arch(controller, super_net, xloader, n_samples=10):
    with torch.no_grad():
        controller.eval()
        super_net.eval()
        archs, valid_accs = [], []
        loader_iter = iter(xloader)
        for i in range(n_samples):
            try:
                inputs, targets = next(loader_iter)
            except:
                loader_iter = iter(xloader)
                inputs, targets = next(loader_iter)

            _, _, sampled_arch = controller()
            arch = super_net.module.update_arch(sampled_arch)
            _, logits = super_net(inputs)
            val_top1, val_top5 = obtain_accuracy(
                logits.cpu().data, targets.data, topk=(1, 5)
            )

            archs.append(arch)
            valid_accs.append(val_top1.item())

        best_idx = np.argmax(valid_accs)
        best_arch, best_valid_acc = archs[best_idx], valid_accs[best_idx]
        return best_arch, best_valid_acc

def valid_func(xloader, network, criterion):
    data_time, batch_time = AverageMeter(), AverageMeter()
    arch_losses, arch_top1, arch_top5 = AverageMeter(), AverageMeter(), AverageMeter()
    network.eval()
    end = time.time()
    with torch.no_grad():
        for step, (arch_inputs, arch_targets) in enumerate(xloader):
            arch_targets = arch_targets.cuda(non_blocking=True)
            # measure data loading time
            data_time.update(time.time() - end)
            # prediction
            _, logits = network(arch_inputs)
            arch_loss = criterion(logits, arch_targets)
            # record
            arch_prec1, arch_prec5 = obtain_accuracy(
                logits.data, arch_targets.data, topk=(1, 5)
            )
            arch_losses.update(arch_loss.item(), arch_inputs.size(0))
            arch_top1.update(arch_prec1.item(), arch_inputs.size(0))
            arch_top5.update(arch_prec5.item(), arch_inputs.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
    return arch_losses.avg, arch_top1.avg, arch_top5.avg


def main(xargs):
    #step 1: 加载配置 
    assert torch.cuda.is_available(), "CUDA is not available."
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(xargs['workers'])
    prepare_seed(xargs['rand_seed'])
    logger = prepare_logger(xargs)

    # step 2 : 加载数据集
    train_data, test_data, xshape, class_num = get_datasets(xargs['dataset'], xargs['data_path'], -1)
    config = load_config(
        xargs['config_path'], {"class_num": class_num, "xshape": xshape}, logger
    )
    _, train_loader, valid_loader = get_nas_search_loaders(
        train_data,
        test_data,
        xargs["dataset"],
        "../config", # todo: 注意路径修改
        config.batch_size,
        xargs["workers"],
    )
    valid_loader.dataset.transform = deepcopy(train_loader.dataset.transform)
    if hasattr(valid_loader.dataset, "transforms"):
        valid_loader.dataset.transforms = deepcopy(train_loader.dataset.transforms)

    # step 3: 搭建搜索空间
    search_space = get_search_spaces("cell", xargs['search_space_name'])
    xargs['search_space'] = search_space
    model_config = dict2config(
        {
            "name": "ENAS",
            "C": xargs['channel'],
            "N": xargs['num_cells'],
            "max_nodes": xargs['max_nodes'],
            "num_classes": class_num,
            "space": search_space,
            "affine": False,
            "track_running_stats": bool(xargs['track_running_stats']),
        },
        None,
    )
    shared_cnn = get_cell_based_super_net(model_config)
    controller = shared_cnn.create_controller()
    w_optimizer, w_scheduler, criterion = get_optim_scheduler(
        shared_cnn.parameters(), config
    )
    a_optimizer = torch.optim.Adam(
        controller.parameters(),
        lr=config.controller_lr,
        betas=config.controller_betas,
        eps=config.controller_eps,
    )
    if xargs['arch_nas_dataset'] is None:
        api = None
    else:
        api = API(xargs['arch_nas_dataset'])
    logger.log("{:} create API = {:} done".format(time_string(), api))
    shared_cnn, controller, criterion = (
        torch.nn.DataParallel(shared_cnn).cuda(),
        controller.cuda(),
        criterion.cuda(),
    )

    last_info, model_base_path, model_best_path = (
        logger.path("info"),
        logger.path("model"),
        logger.path("best"),
    )
    if last_info.exists():  # automatically resume from previous checkpoint
        logger.log(
            "=> loading checkpoint of the last-info '{:}' start".format(last_info)
        )
        last_info = torch.load(last_info)
        start_epoch = last_info["epoch"]
        checkpoint = torch.load(last_info["last_checkpoint"])
        genotypes = checkpoint["genotypes"]
        baseline = checkpoint["baseline"]
        valid_accuracies = checkpoint["valid_accuracies"]
        shared_cnn.load_state_dict(checkpoint["shared_cnn"])
        controller.load_state_dict(checkpoint["controller"])
        w_scheduler.load_state_dict(checkpoint["w_scheduler"])
        w_optimizer.load_state_dict(checkpoint["w_optimizer"])
        a_optimizer.load_state_dict(checkpoint["a_optimizer"])
        logger.log(
            "=> loading checkpoint of the last-info '{:}' start with {:}-th epoch.".format(
                last_info, start_epoch
            )
        )
    else:
        logger.log("=> do not find the last-info file : {:}".format(last_info))
        start_epoch, valid_accuracies, genotypes, baseline = 0, {"best": -1}, {}, None

    # step 4: 开始训练
    start_time, search_time, epoch_time, total_epoch = (
        time.time(),
        AverageMeter(),
        AverageMeter(),
        config.epochs + config.warmup,
    )
    for epoch in range(start_epoch, total_epoch):
        w_scheduler.update(epoch, 0.0)
        need_time = "Time Left: {:}".format(
            convert_secs2time(epoch_time.val * (total_epoch - epoch), True)
        )
        epoch_str = "{:03d}-{:03d}".format(epoch, total_epoch)
        logger.log(
            "\n[Search the {:}-th epoch] {:}, LR={:}, baseline={:}".format(
                epoch_str, need_time, min(w_scheduler.get_lr()), baseline
            )
        )

        cnn_loss, cnn_top1, cnn_top5 = train_super_net(
            train_loader,
            shared_cnn,
            controller,
            criterion,
            w_scheduler,
            w_optimizer,
            epoch_str,
            xargs['print_freq'],
            logger,
        )
        logger.log(
            "[{:}] shared-cnn : loss={:.2f}, accuracy@1={:.2f}%, accuracy@5={:.2f}%".format(
                epoch_str, cnn_loss, cnn_top1, cnn_top5
            )
        )
        ctl_loss, ctl_acc, ctl_baseline, ctl_reward, baseline = train_controller(
            valid_loader,
            shared_cnn,
            controller,
            criterion,
            a_optimizer,
            dict2config(
                {
                    "baseline": baseline,
                    "ctl_train_steps": xargs['controller_train_steps'],
                    "ctl_num_aggre": xargs['controller_num_aggregate'],
                    "ctl_entropy_w": xargs['controller_entropy_weight'],
                    "ctl_bl_dec": xargs['controller_bl_dec'],
                },
                None,
            ),
            epoch_str,
            xargs['print_freq'],
            logger,
        )
        search_time.update(time.time() - start_time)
        logger.log(
            "[{:}] controller : loss={:.2f}, accuracy={:.2f}%, baseline={:.2f}, reward={:.2f}, current-baseline={:.4f}, time-cost={:.1f} s".format(
                epoch_str,
                ctl_loss,
                ctl_acc,
                ctl_baseline,
                ctl_reward,
                baseline,
                search_time.sum,
            )
        )
        best_arch, _ = get_best_arch(controller, shared_cnn, valid_loader)
        shared_cnn.module.update_arch(best_arch)
        _, best_valid_acc, _ = valid_func(valid_loader, shared_cnn, criterion)

        genotypes[epoch] = best_arch
        # check the best accuracy
        valid_accuracies[epoch] = best_valid_acc
        if best_valid_acc > valid_accuracies["best"]:
            valid_accuracies["best"] = best_valid_acc
            genotypes["best"] = best_arch
            find_best = True
        else:
            find_best = False

        logger.log(
            "<<<--->>> The {:}-th epoch : {:}".format(epoch_str, genotypes[epoch])
        )
        # save checkpoint
        save_path = save_checkpoint(
            {
                "epoch": epoch + 1,
                "args": deepcopy(xargs),
                "baseline": baseline,
                "shared_cnn": shared_cnn.state_dict(),
                "controller": controller.state_dict(),
                "w_optimizer": w_optimizer.state_dict(),
                "a_optimizer": a_optimizer.state_dict(),
                "w_scheduler": w_scheduler.state_dict(),
                "genotypes": genotypes,
                "valid_accuracies": valid_accuracies,
            },
            model_base_path,
            logger,
        )
        last_info = save_checkpoint(
            {
                "epoch": epoch + 1,
                "args": deepcopy(args),
                "last_checkpoint": save_path,
            },
            logger.path("info"),
            logger,
        )
        if find_best:
            logger.log(
                "<<<--->>> The {:}-th epoch : find the highest validation accuracy : {:.2f}%.".format(
                    epoch_str, best_valid_acc
                )
            )
            copy_checkpoint(model_base_path, model_best_path, logger)
        if api is not None:
            logger.log("{:}".format(api.query_by_arch(genotypes[epoch], "200")))
        # measure elapsed time
        epoch_time.update(time.time() - start_time)
        start_time = time.time()

    logger.log("\n" + "-" * 100)
    logger.log(
        "During searching, the best architecture is {:}".format(genotypes["best"])
    )
    logger.log("Its accuracy is {:.2f}%".format(valid_accuracies["best"]))
    logger.log(
        "Randomly select {:} architectures and select the best.".format(
            xargs["controller_num_samples"]
        )
    )
    start_time = time.time()
    final_arch, _ = get_best_arch(
        controller, shared_cnn, valid_loader, xargs['controller_num_samples']
    )
    search_time.update(time.time() - start_time)
    shared_cnn.module.update_arch(final_arch)
    final_loss, final_top1, final_top5 = valid_func(valid_loader, shared_cnn, criterion)
    logger.log("The Selected Final Architecture : {:}".format(final_arch))
    logger.log(
        "Loss={:.3f}, Accuracy@1={:.2f}%, Accuracy@5={:.2f}%".format(
            final_loss, final_top1, final_top5
        )
    )
    logger.log(
        "ENAS : run {:} epochs, cost {:.1f} s, last-geno is {:}.".format(
            total_epoch, search_time.sum, final_arch
        )
    )
    if api is not None:
        logger.log("{:}".format(api.query_by_arch(final_arch)))
    logger.close()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser("The NAS-BENCH-201 Algorithm")
    parser.add_argument("--config_file", type=str, default='./config/ENAS/ENAS.yaml', help="config file path")
    args = parser.parse_args()
    config = load_yaml_config(args.config_file)

    if config['rand_seed'] is None or config['rand_seed'] < 0:
        args.rand_seed = random.randint(1, 100000)

    main(config)