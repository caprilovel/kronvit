# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from losses import DistillationLoss
import utils
from local_utils.model_utils import freeze_A, freeze_B, freeze_S
from local_utils.model_utils import unfreeze_A, unfreeze_B, unfreeze_S


def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, kron=False, args = None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    
    # if (epoch // 10) % 2 == 0 and args.kron:
    #     freeze_A(model)
    #     freeze_S(model)
    #     unfreeze_B(model)
        
    # if (epoch // 10) % 2 == 1 and args.kron:
    #     unfreeze_A(model)
    #     unfreeze_S(model)
    #     freeze_B(model)

    
    # if epoch >= 20 and k1l and epoch % 20 == 0:
    #     from local_utils.model_utils import k1l_update_model
    #     k1l_update_model(model)
        
        
    
    if args.cosub:
        criterion = torch.nn.BCEWithLogitsLoss()
        
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
            
        if args.cosub:
            samples = torch.cat((samples,samples),dim=0)
            
        if args.bce_loss:
            targets = targets.gt(0.0).type(targets.dtype)
         
        with torch.cuda.amp.autocast():
            outputs = model(samples)
            if not args.cosub:
                loss = criterion(samples, outputs, targets)
                if kron:
                    from local_utils.loss import norm_s
                    max_epoch = 300
                    lr = 0.1
                    # print(norm_s(model, lr=0.01, p=1)[0] / norm_s(model, lr=0.01, p=1)[1])
                    loss += lr *((epoch*2 + max_epoch)/max_epoch) * norm_s(model, lr=0.01, p=1)[0] / norm_s(model, lr=0.01, p=1)[1] 
                if args.group_lasso:
                    from local_utils.loss import group_lasso
                    gl, num  = group_lasso(model, pattern=[4, 4], lr=0.01)
                    max_epoch = 300
                    gl_lr = 1
                    loss += gl_lr *((epoch*2 + max_epoch)/max_epoch) * gl_lr * gl / num 
                if args.elastic_group_lasso:
                    from local_utils.loss import elastic_group_lasso
                    max_epoch = 300
                    gl_lr = 1
                    loss += gl_lr *((epoch*2 + max_epoch)/max_epoch) * gl_lr * elastic_group_lasso(model, pattern=[4, 4], lr=0.1, alpha=0.05)
                
            else:
                outputs = torch.split(outputs, outputs.shape[0]//2, dim=0)

                loss = 0.25 * criterion(outputs[0], targets) 
                loss = loss + 0.25 * criterion(outputs[1], targets) 
                loss = loss + 0.25 * criterion(outputs[0], outputs[1].detach().sigmoid())
                loss = loss + 0.25 * criterion(outputs[1], outputs[0].detach().sigmoid()) 
                # print(norm_s(model, lr=0.01, p=1)[0] / norm_s(model, lr=0.01, p=1)[1])
                loss += norm_s(model, lr=0.01, p=1)[0] / norm_s(model, lr=0.01, p=1)[1]
                

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    if kron:
        from local_utils.loss import sparse_s
        print(sparse_s(model, thresold=1e-4)[1] / sparse_s(model, thresold=1e-4)[0], norm_s(model, lr=0.01, p=1)[0].detach() / norm_s(model, lr=0.01, p=1)[1])
    else:
        from local_utils.loss import sparse_linear
        print(sparse_linear(model, thresold=1e-4)[1] / sparse_linear(model, thresold=1e-4)[0])
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
