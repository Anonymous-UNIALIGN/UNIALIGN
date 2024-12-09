# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# ELECTRA https://github.com/google-research/electra
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import json
import torch
import torch.optim as optim
from timm.scheduler import CosineLRScheduler

from util.logger import print_log

def param_groups_lrd(model, weight_decay=0.05, no_weight_decay_list=[], layer_decay=.75):
    """
    Parameter groups for layer-wise lr decay
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    """
    param_group_names = {}
    param_groups = {}

    num_layers = len(model.blocks) + 1

    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # no decay: all 1D parameters and model specific ones
        if p.ndim == 1 or n in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.
        else:
            g_decay = "decay"
            this_decay = weight_decay
            
        layer_id = get_layer_id_for_vit(n, num_layers)
        group_name = "layer_%d_%s" % (layer_id, g_decay)

        if group_name not in param_group_names:
            this_scale = layer_scales[layer_id]

            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }

        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)

    # print("parameter groups: \n%s" % json.dumps(param_group_names, indent=2))

    return list(param_groups.values())

def param_groups_multimodal(model, train_modality_list, eff_batch_size, real_batch_size, loss_balancer=None, weight_decay=0.05, no_weight_decay_list=[]):

    param_group_names = {}
    param_groups = {}

    
    modal_scales = []
    for batch in real_batch_size:
        modal_scales.append(batch/eff_batch_size)

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        
        modal_group = None
        modal_idx = None
        for i, modality in enumerate(train_modality_list):
            if modality in n and 'lora_' not in n:
                modal_group = modality
                modal_idx = i
                break
        
        
        # no decay: all 1D parameters and model specific ones
        if p.ndim == 1 or n in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.
        else:
            g_decay = "decay"
            this_decay = weight_decay
            
        if modal_group is None:
            group_name = f'shared_{g_decay}'
        else:
            group_name = f'{modal_group}_{g_decay}'

        if group_name not in param_group_names:
            if modal_group is None:
                this_scale = 1.0
            else:
                this_scale = modal_scales[modal_idx]

            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }

        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)
    
    if loss_balancer is not None:
        
        for n,p in loss_balancer.named_parameters():
            if not p.requires_grad:
                continue
            group_name = "loss_balancer"
            if group_name not in param_group_names:
                param_group_names[group_name] = {
                    "lr_scale": 1.0,
                    "weight_decay": 0.0,
                    "params": [],
                }
                param_groups[group_name] = {
                    "lr_scale": 1.0,
                    "weight_decay": 0.0,
                    "params": [],
                }
            param_group_names[group_name]["params"].append(n)
            param_groups[group_name]["params"].append(p)

    # print_log("parameter groups: \n%s" % json.dumps(param_group_names, indent=2),'param_groups_multimodal')

    return list(param_groups.values())

def get_layer_id_for_vit(name, num_layers):
    """
    Assign a parameter with its layer id
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
    """
    if name in ['cls_token', 'pos_embed']:
        return 0
    elif name.startswith('patch_embed'):
        return 0
    elif name.startswith('blocks'):
        # name = name.replace('blocks.blocks', 'blocks')
        return int(name.split('.')[1]) + 1
    else:
        return num_layers
    # if 'cls_token' in name:
    #     return 0
    # elif 'pos_embed' in name:
    #     return 0
    # elif 'patch_embed' in name:
    #     return 0
    # elif 'blocks' in name:
    #     return int(name.split('.')[2]) + 1
    # else:
    #     return num_layers

def build_opti_sche(base_model):
    
    
    lr = 1e-5
    weight_decay = 0.05
    skip_list = ()

    decay = []
    no_decay = []
    finetune_head = []
    for name, param in base_model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if "head" in name or 'cls_token' in name:
            print("10 * LR: ", name)
            finetune_head.append(param)
        elif (
            len(param.shape) == 1
            or name.endswith(".bias")
            or "token" in name
            or name in skip_list
        ):
            no_decay.append(param)
        else:
            decay.append(param)
    param_groups = [
        {"params": finetune_head, "lr": lr * 10},
        {"params": no_decay, "weight_decay": 0.0, "lr": lr},
        {"params": decay, "weight_decay": weight_decay, "lr": lr},
    ]
    # if opti_type == "AdamW":
    #     optimizer = optim.AdamW(param_groups, **opti_config.kwargs)
    # elif opti_type == "Adam":
    #     optimizer = optim.Adam(param_groups, **opti_config.kwargs)
    # elif opti_type == "RAdam":
    #     optimizer = optim.RAdam(param_groups, **opti_config.kwargs)
    # elif opti_type == "SGD":
    #     optimizer = optim.SGD(param_groups, nesterov=True, **opti_config.kwargs)
    # else:
    #     raise NotImplementedError()
    optimizer=torch.optim.AdamW(param_groups, lr=lr,betas=args.betas,
                                    eps=args.eps, weight_decay=args.wd)

    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=300,
        # t_mul=1,
        lr_min=1e-6,
        # decay_rate=0.1,
        warmup_lr_init=1e-6,
        warmup_t=10,
        cycle_limit=1,
        t_in_epochs=True,
    )
    
    return optimizer, scheduler
