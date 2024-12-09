# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import sys
from typing import Iterable, Optional, Sequence, Union, Callable
from itertools import islice
from functools import partial
import torch
import torch.distributed as dist
import torch.nn.functional as F
from timm.data import Mixup
from timm.utils import accuracy

import util.misc as misc
import util.lr_sched as lr_sched
from datasets import data_transforms
from pointnet2_ops import pointnet2_utils
from torchvision import transforms
import numpy as np
from util.stat import calculate_stats, concat_all_gather
from util.logger import get_logger, print_log

from datasets.ModelNetDataset import farthest_point_sample
from datasets.aug_random import AugRandomContext, np_random
from timm.loss import SoftTargetCrossEntropy
import json
import collections
from tqdm import tqdm
import einops
import pandas as pd

# import clip
from datasets.manager import TxtManager

from util.loss import (
    multi_cross_modal_kd_loss,
    single_cross_modal_kd_loss,
    multi_cross_modal_cls,
    cross_modal_cls,
    multi_uni_modal_nce_loss,
    single_uni_modal_nce_loss,
    single_kd_clip_loss,
)

from util.clip_loss import (
    single_uni_modal_clip_loss_gather,
    multi_uni_modal_clip_loss_gather,
    cross_modal_cls_gather,
    multi_cross_modal_cls_gather,
    single_cross_modal_kd_loss_gather,
    multi_cross_modal_kd_loss_gather,
    single_kd_clip_loss_gather,
    single_cross_modal_kd_loss_visual,
    single_cross_modal_kd_loss_plm,
)

from datasets.constants import PC_META_DATA_DIR
from datasets.modal_depth.data.scene_cls_template import SCENE_CLS_TEMPLATE
from datasets.modal_audio.data.sound_cls_template import SOUND_AS_IMAGE_TEMPLATE
from datasets.metrics import Accuracy, MAP, Recall
from datasets.zero_shot_metadata import OPENAI_IMAGENET_TEMPLATES, IMAGENET_CLASSNAMES
from clip.simple_tokenizer import SimpleTokenizer


def kd_normalize(logit):
    mean = logit.mean(dim=-1, keepdims=True)
    stdv = logit.std(dim=-1, keepdims=True)
    return (logit - mean) / (1e-7 + stdv)


def get_rank():
    if not dist.is_available() or not dist.is_initialized():
        return 0
    return dist.get_rank()


pc_train_transforms = transforms.Compose(
    [
        # data_transforms.PointcloudScale(),
        # data_transforms.PointcloudRotate(),
        # data_transforms.PointcloudTranslate(),
        # data_transforms.PointcloudJitter(),
        # data_transforms.PointcloudRandomInputDropout(),
        # data_transforms.RandomHorizontalFlip(),
        data_transforms.PointcloudScaleAndTranslate(),
    ]
)

pc_test_transforms = transforms.Compose(
    [
        # data_transforms.PointcloudScale(),
        # data_transforms.PointcloudRotate(),
        # data_transforms.PointcloudTranslate(),
        data_transforms.PointcloudScaleAndTranslate(),
    ]
)


def batched(iterable, n):
    """Batch data into lists of length *n*. The last batch may be shorter.
    NOTE based on more-itertools impl, to be replaced by python 3.12 itertools.batched impl
    """
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            break
        yield batch


def build_zero_shot_classifier(
    model,
    tokenizer,
    classnames: Sequence[str],
    templates: Sequence[Union[Callable, str]],
    num_classes_per_batch: Optional[int] = 10,
    device: Union[str, torch.device] = "cpu",
    use_tqdm: bool = False,
):
    """Build zero-shot classifier weights by iterating over class names in batches
    Args:
        model: CLIP model instance
        tokenizer: CLIP tokenizer instance
        classnames: A sequence of class (label) names
        templates: A sequence of callables or format() friendly strings to produce templates per class name
        num_classes_per_batch: The number of classes to batch together in each forward, all if None
        device: Device to use.
        use_tqdm: Enable TQDM progress bar.
    """
    assert isinstance(templates, Sequence) and len(templates) > 0
    assert isinstance(classnames, Sequence) and len(classnames) > 0
    use_format = isinstance(templates[0], str)
    num_templates = len(templates)
    num_classes = len(classnames)
    if use_tqdm:
        import tqdm

        num_iter = (
            1
            if num_classes_per_batch is None
            else ((num_classes - 1) // num_classes_per_batch + 1)
        )
        iter_wrap = partial(tqdm.tqdm, total=num_iter, unit_scale=num_classes_per_batch)
    else:
        iter_wrap = iter

    def _process_batch(batch_classnames):
        num_batch_classes = len(batch_classnames)
        texts = [
            template.format(c) if use_format else template(c)
            for c in batch_classnames
            for template in templates
        ]
        texts = tokenizer(texts).to(device)
        class_embeddings = model.encode_text(texts, normalize=True)
        class_embeddings = class_embeddings.reshape(
            num_batch_classes, num_templates, -1
        ).mean(dim=1)
        class_embeddings = class_embeddings / class_embeddings.norm(dim=1, keepdim=True)
        class_embeddings = class_embeddings.T
        return class_embeddings

    with torch.no_grad():
        if num_classes_per_batch:
            batched_embeds = [
                _process_batch(batch)
                for batch in iter_wrap(batched(classnames, num_classes_per_batch))
            ]
            zeroshot_weights = torch.cat(batched_embeds, dim=1)
        else:
            zeroshot_weights = _process_batch(classnames)
    return zeroshot_weights


def acc(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res, correct


def cond_acc(output, target, idx_mapping, merge_idx=100, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)

        for idx in idx_mapping:
            target[target == idx] = merge_idx
            pred[pred == idx] = merge_idx

        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].float().max(dim=0)[0].sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res, correct


def scaled_all_reduce(tensors, is_scale=True):
    """Performs the scaled all_reduce operation on the provided tensors.
    The input tensors are modified in-place. Currently supports only the sum
    reduction operator. The reduced values are scaled by the inverse size of the
    world size.
    """
    world_size = misc.get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors
    # Queue the reductions
    reductions = []
    for tensor in tensors:
        reduction = dist.all_reduce(tensor, async_op=True)
        reductions.append(reduction)
    # Wait for reductions to finish
    for reduction in reductions:
        reduction.wait()
    # Scale the results
    if is_scale:
        for tensor in tensors:
            tensor.mul_(1.0 / world_size)
    return tensors


def train_one_epoch(
    model: torch.nn.Module,
    img_criterion: torch.nn.Module,
    audio_criterion: torch.nn.Module,
    pc_criterion: torch.nn.Module,
    img_data_loader: Iterable,
    audio_data_loader: Iterable,
    pc_data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = 0,
    mixup_fn: Optional[Mixup] = None,
    log_writer=None,
    args=None,
):
    model.train(True)
    logger = get_logger(args.log_name)
    metric_logger = misc.MetricLogger(delimiter=" ", logger=logger)
    header = "Multi Tasks Merging ==== Epoch: [{}]".format(epoch)
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))

    print_freq = 100

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print("log_dir: {}".format(log_writer.log_dir))

    # data_iter_step = 0

    # audio_topk = args.audio_topk
    # pc_topk = args.pc_topk

    npoints = args.pc_n_points

    whole_data_len = min(
        len(img_data_loader), len(audio_data_loader), len(pc_data_loader)
    )

    for data_iter_step, (
        (img_samples, img_targets),
        (audio_samples, audio_targets, _vids),
        (points, pc_targets),
    ) in enumerate(
        zip(
            img_data_loader,
            audio_data_loader,
            metric_logger.log_every(pc_data_loader, print_freq, header),
        )
    ):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer, data_iter_step / whole_data_len + epoch, args
            )

        img_samples = img_samples.to(device, non_blocking=True)
        img_targets = img_targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            img_samples, img_targets = mixup_fn(img_samples, img_targets)

        audio_samples = audio_samples.to(device, non_blocking=True)
        audio_targets = audio_targets.to(device, non_blocking=True)

        points = points.to(device, non_blocking=True)
        pc_targets = pc_targets.to(device, non_blocking=True)

        if npoints == 1024:
            point_all = 1200
        elif npoints == 2048:
            point_all = 2400
        elif npoints == 4096:
            point_all = 4800
        elif npoints == 8192:
            point_all = 8192
        else:
            raise NotImplementedError()

        if points.size(1) < point_all:
            point_all = points.size(1)

        fps_idx = pointnet2_utils.furthest_point_sample(
            points, point_all
        )  # (B, npoint)
        fps_idx = fps_idx[:, np.random.choice(point_all, npoints, False)]
        points = (
            pointnet2_utils.gather_operation(
                points.transpose(1, 2).contiguous(), fps_idx
            )
            .transpose(1, 2)
            .contiguous()
        )  # (B, N, 3)
        # import pdb; pdb.set_trace()
        points = pc_train_transforms(points)

        with torch.cuda.amp.autocast():
            if args.use_loramoe:
                img_logits, audio_logits, pc_logits, blcls = model(
                    img_samples,
                    audio_samples,
                    points,
                    mask_t_prob=args.mask_t_prob,
                    mask_f_prob=args.mask_f_prob,
                )
            else:
                img_logits, audio_logits, pc_logits = model(
                    img_samples,
                    audio_samples,
                    points,
                    mask_t_prob=args.mask_t_prob,
                    mask_f_prob=args.mask_f_prob,
                )
        img_loss = img_criterion(img_logits, img_targets)
        audio_loss = audio_criterion(audio_logits, audio_targets)
        pc_loss = pc_criterion(pc_logits, pc_targets.long())

        img_loss_value = img_loss.item()
        audio_loss_value = audio_loss.item()
        pc_loss_value = pc_loss.item()

        loss = (
            args.img_weight * img_loss
            + args.audio_weight * audio_loss
            + args.pc_weight * pc_loss
        )
        if args.use_loramoe:
            loss += blcls

        loss_value = loss.item()

        if not math.isfinite(img_loss_value):
            print("Image Loss is {}, stopping training".format(img_loss_value))
            sys.exit(1)

        loss /= accum_iter

        loss_scaler(
            loss,
            optimizer,
            clip_grad=max_norm,
            parameters=model.parameters(),
            create_graph=False,
            update_grad=(data_iter_step + 1) % accum_iter == 0,
        )  # backward
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(img_loss=img_loss_value)
        metric_logger.update(audio_loss=audio_loss_value)
        metric_logger.update(pc_loss=pc_loss_value)
        if args.use_loramoe:
            metric_logger.update(blcls=blcls.item())

        min_lr = 10.0
        max_lr = 0.0
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        img_loss_value_reduce = misc.all_reduce_mean(img_loss_value)
        audio_loss_value_reduce = misc.all_reduce_mean(audio_loss_value)
        pc_loss_value_reduce = misc.all_reduce_mean(pc_loss_value)

        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / whole_data_len + epoch) * 1000)
            log_writer.add_scalar("loss", loss_value_reduce, epoch_1000x)
            log_writer.add_scalar("lr", max_lr, epoch_1000x)
            log_writer.add_scalar("img_loss", img_loss_value_reduce, epoch_1000x)
            log_writer.add_scalar("audio_loss", audio_loss_value_reduce, epoch_1000x)
            log_writer.add_scalar("pc_loss", pc_loss_value_reduce, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch_grad_norm(
    model: torch.nn.Module,
    img_criterion: torch.nn.Module,
    audio_criterion: torch.nn.Module,
    pc_criterion: torch.nn.Module,
    img_data_loader: Iterable,
    audio_data_loader: Iterable,
    pc_data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = 0,
    mixup_fn: Optional[Mixup] = None,
    log_writer=None,
    args=None,
):
    model.train(True)
    logger = get_logger(args.log_name)
    metric_logger = misc.MetricLogger(delimiter=" ", logger=logger)
    header = "Multi Tasks Merging with grad norm==== Epoch: [{}]".format(epoch)
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))

    print_freq = 100

    accum_iter = 1
    loss_scaler = torch.cuda.amp.GradScaler()

    if log_writer is not None:
        print("log_dir: {}".format(log_writer.log_dir))

    # data_iter_step = 0

    # audio_topk = args.audio_topk
    # pc_topk = args.pc_topk

    npoints = args.pc_n_points

    whole_data_len = min(
        len(img_data_loader), len(audio_data_loader), len(pc_data_loader)
    )

    for data_iter_step, (
        (img_samples, img_targets),
        (audio_samples, audio_targets, _vids),
        (points, pc_targets),
    ) in enumerate(
        zip(
            img_data_loader,
            audio_data_loader,
            metric_logger.log_every(pc_data_loader, print_freq, header),
        )
    ):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer, data_iter_step / whole_data_len + epoch, args
            )

        img_samples = img_samples.to(device, non_blocking=True)
        img_targets = img_targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            img_samples, img_targets = mixup_fn(img_samples, img_targets)

        audio_samples = audio_samples.to(device, non_blocking=True)
        audio_targets = audio_targets.to(device, non_blocking=True)

        points = points.to(device, non_blocking=True)
        pc_targets = pc_targets.to(device, non_blocking=True)

        if npoints == 1024:
            point_all = 1200
        elif npoints == 2048:
            point_all = 2400
        elif npoints == 4096:
            point_all = 4800
        elif npoints == 8192:
            point_all = 8192
        else:
            raise NotImplementedError()

        if points.size(1) < point_all:
            point_all = points.size(1)

        fps_idx = pointnet2_utils.furthest_point_sample(
            points, point_all
        )  # (B, npoint)
        fps_idx = fps_idx[:, np.random.choice(point_all, npoints, False)]
        points = (
            pointnet2_utils.gather_operation(
                points.transpose(1, 2).contiguous(), fps_idx
            )
            .transpose(1, 2)
            .contiguous()
        )  # (B, N, 3)
        # import pdb; pdb.set_trace()
        points = pc_train_transforms(points)

        with torch.cuda.amp.autocast():
            img_logits, audio_logits, pc_logits = model(
                img_samples,
                audio_samples,
                points,
                mask_t_prob=args.mask_t_prob,
                mask_f_prob=args.mask_f_prob,
            )
        img_loss = img_criterion(img_logits, img_targets)
        audio_loss = audio_criterion(audio_logits, audio_targets)
        pc_loss = pc_criterion(pc_logits, pc_targets.long())

        task_loss = torch.stack([img_loss, audio_loss, pc_loss])
        weighted_task_loss = torch.mul(model.task_weights, task_loss)
        if data_iter_step == 0:
            # set L(0)
            if torch.cuda.is_available():
                initial_task_loss = task_loss.data.cpu()
            else:
                initial_task_loss = task_loss.data
            initial_task_loss = initial_task_loss.numpy()
        # get the total loss
        loss = torch.sum(weighted_task_loss)
        # loss /= accum_iter
        optimizer.zero_grad()

        # do the backward pass to compute the gradients for the whole set of weights
        # This is equivalent to compute each \nabla_W L_i(t)
        loss_scaler.scale(loss).backward(retain_graph=True)

        # loss_scaler(
        #     loss,
        #     optimizer,
        #     clip_grad=max_norm,
        #     parameters=model.parameters(),
        #     retain_graph=True,
        #     create_graph=False,
        #     update_grad=False,
        # ) # backward

        # set the gradients of w_i(t) to zero because these gradients have to be updated using the GradNorm loss
        # print('Before turning to 0: {}'.format(model.weights.grad))
        model.task_weights.grad.data = model.task_weights.grad.data * 0.0

        # get layer of shared weights
        W = model.get_last_shared_layer()
        # get the gradient norms for each of the tasks
        # G^{(i)}_w(t)
        norms = []
        for i in range(len(task_loss)):
            # get the gradient of this task loss with respect to the shared parameters
            gygw = torch.autograd.grad(task_loss[i], W.parameters(), retain_graph=True)
            # compute the norm
            norms.append(torch.norm(torch.mul(model.task_weights[i], gygw[0])))
        norms = torch.stack(norms)
        # print('G_w(t): {}'.format(norms))

        # compute the inverse training rate r_i(t)
        # \curl{L}_i
        if torch.cuda.is_available():
            loss_ratio = task_loss.data.cpu().numpy() / initial_task_loss
        else:
            loss_ratio = task_loss.data.numpy() / initial_task_loss
        # r_i(t)
        inverse_train_rate = loss_ratio / np.mean(loss_ratio)
        # print('r_i(t): {}'.format(inverse_train_rate))

        # compute the mean norm \tilde{G}_w(t)
        if torch.cuda.is_available():
            mean_norm = np.mean(norms.data.cpu().numpy())
        else:
            mean_norm = np.mean(norms.data.numpy())
        # print('tilde G_w(t): {}'.format(mean_norm))

        # compute the GradNorm loss
        # this term has to remain constant
        constant_term = torch.tensor(
            mean_norm * (inverse_train_rate**args.grad_norm_alpha), requires_grad=False
        )
        if torch.cuda.is_available():
            constant_term = constant_term.cuda()
        # print('Constant term: {}'.format(constant_term))
        # this is the GradNorm loss itself
        grad_norm_loss = torch.sum(torch.abs(norms - constant_term))
        # print('GradNorm loss {}'.format(grad_norm_loss))

        # compute the gradient for the weights
        model.task_weights.grad = torch.autograd.grad(
            grad_norm_loss, model.task_weights
        )[0]

        # loss_scaler(
        #     loss,
        #     optimizer,
        #     clip_grad=max_norm,
        #     parameters=model.parameters(),
        #     create_graph=False,
        #     update_grad=(data_iter_step + 1) % accum_iter == 0,
        # ) # backward

        if max_norm is not None:
            loss_scaler.unscale_(
                optimizer
            )  # unscale the gradients of optimizer's assigned params in-place
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            from util.misc import get_grad_norm_

            loss_scaler.unscale_(optimizer)
            norm = get_grad_norm_(model.parameters())
        loss_scaler.step(optimizer)
        loss_scaler.update()

        # torch.cuda.synchronize()

        metric_logger.update(loss_ratios=np.sum(loss_ratio).item())
        metric_logger.update(grad_norm_loss=grad_norm_loss.item())
        metric_logger.update(img_loss=task_loss[0].item())
        metric_logger.update(audio_loss=task_loss[1].item())
        metric_logger.update(pc_loss=task_loss[2].item())
        metric_logger.update(img_weight=model.task_weights[0].item())
        metric_logger.update(audio_weight=model.task_weights[1].item())
        metric_logger.update(pc_weight=model.task_weights[2].item())

        min_lr = 10.0
        max_lr = 0.0
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        # grad_norm_loss_value_reduce = misc.all_reduce_mean(grad_norm_loss.item())
        # img_loss_value_reduce = misc.all_reduce_mean(task_loss[0].item())
        # audio_loss_value_reduce = misc.all_reduce_mean(task_loss[1].item())
        # pc_loss_value_reduce = misc.all_reduce_mean(task_loss[2].item())
        # img_w_value_reduce = misc.all_reduce_mean(model.task_weights[0].item())
        # audio_w_value_reduce = misc.all_reduce_mean(model.modulel.task_weights[1].item())
        # pc_w_value_reduce = misc.all_reduce_mean(model.module.task_weights[2].item())

        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / whole_data_len + epoch) * 1000)
            log_writer.add_scalar("grad_norm_loss", grad_norm_loss.item(), epoch_1000x)
            log_writer.add_scalar("lr", max_lr, epoch_1000x)
            log_writer.add_scalar("img_loss", task_loss[0].item(), epoch_1000x)
            log_writer.add_scalar("audio_loss", task_loss[1].item(), epoch_1000x)
            log_writer.add_scalar("pc_loss", task_loss[2].item(), epoch_1000x)
            log_writer.add_scalar(
                "img_weight", model.task_weights[0].item(), epoch_1000x
            )
            log_writer.add_scalar(
                "audio_weight", model.task_weights[1].item(), epoch_1000x
            )
            log_writer.add_scalar(
                "pc_weight", model.task_weights[2].item(), epoch_1000x
            )

    # renormalize
    normalize_coeff = args.modal_nums / torch.sum(model.task_weights.data, dim=0)
    model.task_weights.data = model.task_weights.data * normalize_coeff

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch_concat(
    model: torch.nn.Module,
    open_clip_text_model: None,
    loss_balancer: torch.nn.Module,
    criterion: dict,
    train_data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = 0,
    mixup_fn: Optional[Mixup] = None,
    log_writer=None,
    args=None,
):
    model.train(True)
    logger = get_logger(args.log_name)
    metric_logger = misc.MetricLogger(delimiter=" ", logger=logger)
    header = "Multi Tasks Merging with use_one mode ==== Epoch: [{}]".format(epoch)
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))

    print_freq = 100

    accum_iter = args.accum_iter

    if log_writer is not None:
        print_log("log_dir: {}".format(log_writer.log_dir), logger=logger)

    has_cls_head = model.module.has_cls_head

    world_size = misc.get_world_size()
    rank = misc.get_rank()
    item_size = args.text_embed_dim * 2
    modal_labels_features = {}
    # plm_labels_features = {}
    if args.use_text_branch:
        tokenizer = SimpleTokenizer()
    for m in args.train_modal_list:
        if m == "image":
            # if args.text_embed_dim == 1536:
            image_manager = TxtManager(
                "text_features/Image/imagenet_openai", item_size, rank
            )
            # image_labels_features = []
            # for label in range(args.image_nb_classes):
            #     label = str(label)
            #     bstr = image_manager.read(label)
            #     text_feature = np.frombuffer(bstr[:item_size], dtype=np.float16).copy()
            #     text_feature = torch.from_numpy(text_feature).to(
            #         device=device, dtype=torch.float16
            #     )
            #     image_labels_features.append(text_feature)
            # image_labels_features = torch.stack(image_labels_features)
            # plm_labels_features["image"] = image_labels_features
            # else:
            classifier = build_zero_shot_classifier(
                open_clip_text_model.module
                if args.distributed
                else open_clip_text_model,
                tokenizer=tokenizer,
                classnames=IMAGENET_CLASSNAMES,
                templates=OPENAI_IMAGENET_TEMPLATES,
                num_classes_per_batch=10,
                device=device,
                use_tqdm=True,
            )
            modal_labels_features["image"] = classifier.T

        if m == "audio":
            # if args.text_embed_dim == 1536:
            # audio_manager = TxtManager(
            #     "text_features/Audio/audioset_openai", item_size, rank
            # )
            # audio_labels_features = []
            # for label in range(args.audio_nb_classes):
            #     label = str(label)
            #     bstr = audio_manager.read(label)
            #     text_feature = np.frombuffer(bstr[:item_size], dtype=np.float16).copy()
            #     text_feature = torch.from_numpy(text_feature).to(
            #         device=device, dtype=torch.float16
            #     )
            #     audio_labels_features.append(text_feature)
            # audio_labels_features = torch.stack(audio_labels_features)
            # plm_labels_features["audio"] = audio_labels_features

            # else:
            if args.multi_modal_distill:
                idx2label = train_data_loader.datasets["audio"].dataset.idx2label
            else:
                idx2label = train_data_loader.datasets["audio"].idx2label
            audio_labels_features = []
            for label in idx2label:
                texts = [t(label) for t in SOUND_AS_IMAGE_TEMPLATE]
                texts = tokenizer(texts).cuda(args.device, non_blocking=True)
                if len(texts.shape) < 2:
                    texts = texts[None, ...]
                with torch.no_grad():
                    if args.distributed:
                        class_embeddings = open_clip_text_model.module.encode_text(
                            texts
                        )
                    else:
                        class_embeddings = open_clip_text_model.encode_text(texts)
                class_embeddings = class_embeddings / class_embeddings.norm(
                    dim=-1, keepdim=True
                )
                class_embeddings = class_embeddings.mean(dim=0)
                class_embeddings = class_embeddings / class_embeddings.norm(
                    dim=-1, keepdim=True
                )
                audio_labels_features.append(class_embeddings)
            modal_labels_features["audio"] = torch.stack(audio_labels_features, dim=0)

        if m == "point":
            # if args.text_embed_dim == 1536:
            # point_manager = TxtManager(
            #     "text_features/Point_cloud/shapenet55_openai", item_size, rank
            # )
            # point_labels_features = []
            # for label in range(args.pc_nb_classes):
            #     label = str(label)
            #     bstr = point_manager.read(label)
            #     text_feature = np.frombuffer(bstr[:item_size], dtype=np.float16).copy()
            #     text_feature = torch.from_numpy(text_feature).to(
            #         device=device, dtype=torch.float16
            #     )
            #     point_labels_features.append(text_feature)
            # point_labels_features = torch.stack(point_labels_features)
            # plm_labels_features["point"] = point_labels_features

            # else:
            if args.point_train_data == "shapenet":
                labels = []
                for index, row in pd.read_json(
                    f"{PC_META_DATA_DIR}/shapenet55.json"
                ).iterrows():
                    labels.append(row["describe"])

                print_log("=> encoding 3dpc captions", logger=logger)

            with open(f"{PC_META_DATA_DIR}/templates.json") as f:
                templates = json.load(f)[args.point_train_data_prompt]
            point_labels_features = []
            for label in labels:
                texts = [t.format(label) for t in templates]
                texts = tokenizer(texts).cuda(args.device, non_blocking=True)
                if len(texts.shape) < 2:
                    texts = texts[None, ...]
                with torch.no_grad():
                    if args.distributed:
                        class_embeddings = open_clip_text_model.module.encode_text(
                            texts
                        )
                    else:
                        class_embeddings = open_clip_text_model.encode_text(texts)
                class_embeddings = class_embeddings / class_embeddings.norm(
                    dim=-1, keepdim=True
                )
                class_embeddings = class_embeddings.mean(dim=0)
                class_embeddings = class_embeddings / class_embeddings.norm(
                    dim=-1, keepdim=True
                )
                point_labels_features.append(class_embeddings)
            modal_labels_features["point"] = torch.stack(point_labels_features, dim=0)

        # if m == "video":
        #     video_manager = TxtManager(args.video_text_feature_path, item_size, rank)
        #     video_labels_features = []
        #     for label in range(args.video_nb_classes):
        #         label = str(label)
        #         bstr = video_manager.read(label)
        #         text_feature = np.frombuffer(bstr[:item_size], dtype=np.float16).copy()
        #         text_feature = torch.from_numpy(text_feature).to(
        #             device=device, dtype=torch.float16
        #         )
        #         video_labels_features.append(text_feature)
        #     video_labels_features = torch.stack(video_labels_features)
        #     modal_labels_features["video"] = video_labels_features

        if m == "rgbd":
            rgbd_labels_features = []
            if args.multi_modal_distill:
                labels = train_data_loader.datasets["rgbd"].dataset.idx2label
            else:
                labels = train_data_loader.datasets["rgbd"].idx2label
            # # if args.text_embed_dim == 1536:
            # rgbd_train_text_features = torch.load(
            #     "text_features/RGBD/sunrgbd_train_openai.pth"
            # )
            # for label in labels:
            #     text_feature = rgbd_train_text_features[label].to(
            #         device=device, dtype=torch.float16
            #     )
            #     rgbd_labels_features.append(text_feature)
            # rgbd_labels_features = torch.stack(rgbd_labels_features)
            # plm_labels_features["rgbd"] = rgbd_labels_features
            # else:
            templates = SCENE_CLS_TEMPLATE
            for label in labels:
                texts = [t(label) for t in templates]
                texts = tokenizer(texts).cuda(args.device, non_blocking=True)
                if len(texts.shape) < 2:
                    texts = texts[None, ...]
                with torch.no_grad():
                    if args.distributed:
                        class_embeddings = open_clip_text_model.module.encode_text(
                            texts
                        )
                    else:
                        class_embeddings = open_clip_text_model.encode_text(texts)
                class_embeddings = class_embeddings / class_embeddings.norm(
                    dim=-1, keepdim=True
                )
                class_embeddings = class_embeddings.mean(dim=0)
                class_embeddings = class_embeddings / class_embeddings.norm(
                    dim=-1, keepdim=True
                )
                rgbd_labels_features.append(class_embeddings)
            modal_labels_features["rgbd"] = torch.stack(rgbd_labels_features, dim=0)

    accum_modal_iter = 0
    modal_lens = len(args.train_modal_list)

    trgets_dict = {}
    modal_list = []
    anchor_list = []
    input_list = []

    accum_features = {}
    accum_text_features = {}
    accum_kd_s_features = {}
    accum_kd_t_features = {}
    accum_kd_text_features = {}
    accum_kd_image_features = {}

    for data_iter_step, batch in enumerate(
        metric_logger.log_every(
            train_data_loader, print_freq, header, pre_iter=int(modal_lens * accum_iter)
        )
    ):
        cur_modal = None
        input = None

        (input_data, mini_batch_modal) = batch

        if mini_batch_modal == "image":
            cur_modal = "image"

            modal_list.append("image")
            cur_anchor = ["image"]
            anchor_list.append(["image"])

            img_samples, img_targets, img_texts = (
                input_data["image"],
                input_data["target"],
                input_data["caption"],
            )

            img_samples = img_samples.to(device, non_blocking=True)

            img_texts = tokenizer(img_texts).to(device, non_blocking=True)

            if isinstance(img_targets, list):
                img_targets = torch.LongTensor(img_targets)
            img_targets = img_targets.to(device, non_blocking=True)

            if mixup_fn is not None:
                img_samples, img_targets = mixup_fn(img_samples, img_targets)
            input = {"image": img_samples}
            input_list.append(input)
            cur_text = img_texts
            # text_list.append(cur_text)

            if "image" in trgets_dict:
                trgets_dict["image"].append(img_targets)
            else:
                trgets_dict["image"] = [img_targets]

            if args.use_text and not args.use_text_branch:
                image_text_features = []
                for label in img_targets:
                    label = str(label.item())
                    bstr = image_manager.read(label)
                    text_feature = np.frombuffer(
                        bstr[:item_size], dtype=np.float16
                    ).copy()
                    text_feature = torch.from_numpy(text_feature).to(
                        device=device, dtype=torch.float16
                    )
                    image_text_features.append(text_feature)
                image_text_features = torch.stack(image_text_features)
                if "image" in accum_text_features:
                    accum_text_features["image"].append(image_text_features)
                else:
                    accum_text_features["image"] = [image_text_features]

            accum_modal_iter += 1

        elif mini_batch_modal == "audio":
            cur_modal = "audio"

            modal_list.append("audio")
            cur_anchor = ["audio"]
            if args.audio_load_vision:
                anchor_list.append(["audio", "image"])
            else:
                anchor_list.append(["audio"])
            if args.use_text_template and not args.use_text_branch:
                audio_samples, audio_targets, audio_text_features = (
                    input_data["audio"],
                    input_data["target"],
                    input_data["text_feature"],
                )
                audio_text_features = audio_text_features.to(device, non_blocking=True)
                if "audio" in accum_text_features:
                    accum_text_features["audio"].append(audio_text_features)
                else:
                    accum_text_features["audio"] = [audio_text_features]
            else:
                audio_samples, audio_targets, audio_texts = (
                    input_data["audio"],
                    input_data["target"],
                    input_data["caption"],
                )
                audio_texts = audio_texts.to(device=device, non_blocking=True)

                cur_text = audio_texts

                # text_list.append(cur_text)

            if args.multi_modal_distill:
                t_visual_features = input_data["key"]
                t_visual_features = t_visual_features.to(device, non_blocking=True)
                if "audio" in accum_kd_t_features:
                    accum_kd_t_features["audio"].append(t_visual_features)
                else:
                    accum_kd_t_features["audio"] = [t_visual_features]

                t_text_features = input_data["key_text"]
                t_text_features = t_text_features.to(device, non_blocking=True)

                t_image_features = input_data["key_image"]
                t_image_features = t_image_features.to(device, non_blocking=True)

                if "audio" in accum_kd_text_features:
                    accum_kd_text_features["audio"].append(t_text_features)
                else:
                    accum_kd_text_features["audio"] = [t_text_features]

                if "audio" in accum_kd_image_features:
                    accum_kd_image_features["audio"].append(t_image_features)
                else:
                    accum_kd_image_features["audio"] = [t_image_features]

            audio_samples = audio_samples.to(device, non_blocking=True)
            if isinstance(audio_targets, list):
                audio_targets = torch.LongTensor(audio_targets)
            audio_targets = audio_targets.to(device, non_blocking=True)

            if args.audio_load_vision:
                video_samples = input_data["image"].to(device, non_blocking=True)
                input = {"audio": audio_samples, "image": video_samples}
            else:
                input = {"audio": audio_samples}
            input_list.append(input)
            if "audio" in trgets_dict:
                trgets_dict["audio"].append(audio_targets)
            else:
                trgets_dict["audio"] = [audio_targets]

            # if args.use_text and not args.use_text_template:
            #     audio_text_features = []
            #     for label in audio_targets:
            #         label = str(label.argmax(dim=-1).item())
            #         bstr = audio_manager.read(label)
            #         text_feature = np.frombuffer(
            #             bstr[:item_size], dtype=np.float16
            #         ).copy()
            #         text_feature = torch.from_numpy(text_feature).to(
            #             device=device, dtype=torch.float16
            #         )
            #         audio_text_features.append(text_feature)
            #     audio_text_features = torch.stack(audio_text_features)
            #     if "audio" in accum_text_features:
            #         accum_text_features["audio"].append(audio_text_features)
            #     else:
            #         accum_text_features["audio"] = [audio_text_features]

            accum_modal_iter += 1

        elif mini_batch_modal == "point":
            cur_modal = "point"

            modal_list.append("point")
            if args.use_pc_image:
                cur_anchor = ["point", "image"]
                anchor_list.append(["point", "image"])
            else:
                cur_anchor = ["point"]
                anchor_list.append(["point"])

            if args.use_text_template and not args.use_text_branch:
                if args.use_pc_image:
                    points, pc_img, pc_targets, point_text_features = (
                        input_data["pc"],
                        input_data["image"],
                        input_data["target"],
                        input_data["text_feature"],
                    )
                else:
                    points, pc_targets, point_text_features = (
                        input_data["pc"],
                        input_data["target"],
                        input_data["text_feature"],
                    )

                point_text_features = point_text_features.to(device, non_blocking=True)
                if "point" in accum_text_features:
                    accum_text_features["point"].append(point_text_features)
                else:
                    accum_text_features["point"] = [point_text_features]

            else:
                if args.use_pc_image:
                    points, pc_img, pc_targets, pc_texts = (
                        input_data["pc"],
                        input_data["image"],
                        input_data["target"],
                        input_data["caption"],
                    )
                else:
                    points, pc_targets, pc_texts = (
                        input_data["pc"],
                        input_data["target"],
                        input_data["caption"],
                    )
                pc_texts = pc_texts.to(device=device, non_blocking=True)

                cur_text = pc_texts
                # text_list.append(cur_text)

            if args.multi_modal_distill:
                t_visual_features = input_data["key"]
                t_visual_features = t_visual_features.to(device, non_blocking=True)
                if "point" in accum_kd_t_features:
                    accum_kd_t_features["point"].append(t_visual_features)
                else:
                    accum_kd_t_features["point"] = [t_visual_features]

                if 'key_text' in input_data:
                    t_text_features = input_data["key_text"]
                    t_text_features = t_text_features.to(device, non_blocking=True)
                    
                    if "point" in accum_kd_text_features:
                        accum_kd_text_features["point"].append(t_text_features)
                    else:
                        accum_kd_text_features["point"] = [t_text_features]

                if 'key_image' in input_data:
                    t_image_features = input_data["key_image"]
                    t_image_features = t_image_features.to(device, non_blocking=True)

                    if "point" in accum_kd_image_features:
                        accum_kd_image_features["point"].append(t_image_features)
                    else:
                        accum_kd_image_features["point"] = [t_image_features]

            points = points.to(device, non_blocking=True)
            if isinstance(pc_targets, list):
                pc_targets = torch.LongTensor(pc_targets)
            pc_targets = pc_targets.to(device, non_blocking=True)

            points = pc_train_transforms(points)
            if args.use_pc_image:
                input = {"point": points, "image": pc_img}
            else:
                input = {"point": points}
            input_list.append(input)
            if "point" in trgets_dict:
                trgets_dict["point"].append(pc_targets.long())
            else:
                trgets_dict["point"] = [pc_targets.long()]

            # if args.use_text and not args.use_text_template:
            #     point_text_features = []
            #     for label in pc_targets:
            #         label = str(label.item())
            #         bstr = point_manager.read(label)
            #         text_feature = np.frombuffer(
            #             bstr[:item_size], dtype=np.float16
            #         ).copy()
            #         text_feature = torch.from_numpy(text_feature).to(
            #             device=device, dtype=torch.float16
            #         )
            #         point_text_features.append(text_feature)
            #     point_text_features = torch.stack(point_text_features)
            #     if "point" in accum_text_features:
            #         accum_text_features["point"].append(point_text_features)
            #     else:
            #         accum_text_features["point"] = [point_text_features]

        elif mini_batch_modal == "rgbd":
            cur_modal = "rgbd"

            modal_list.append("rgbd")
            cur_anchor = ["rgbd", "image"]
            anchor_list.append(["rgbd", "image"])
            if args.use_text_template and not args.use_text_branch:
                depth_samples, rgb_samples, rgbd_targets, rgbd_text_features = (
                    input_data["depth"],
                    input_data["image"],
                    input_data["label"],
                    input_data["text_feature"],
                )

                rgbd_text_features = rgbd_text_features.to(device, non_blocking=True)
                if "rgbd" in accum_text_features:
                    accum_text_features["rgbd"].append(rgbd_text_features)
                else:
                    accum_text_features["rgbd"] = [rgbd_text_features]

            else:
                depth_samples, rgb_samples, rgbd_targets, rgbd_texts = (
                    input_data["depth"],
                    input_data["image"],
                    input_data["label"],
                    input_data["caption"],
                )
                rgbd_texts = rgbd_texts.to(device=device, non_blocking=True)

                cur_text = rgbd_texts
                # text_list.append(cur_text)

            if args.multi_modal_distill:
                t_visual_features = input_data["key"]
                t_visual_features = t_visual_features.to(device, non_blocking=True)
                if "rgbd" in accum_kd_t_features:
                    accum_kd_t_features["rgbd"].append(t_visual_features)
                else:
                    accum_kd_t_features["rgbd"] = [t_visual_features]

                t_text_features = input_data["key_text"]
                t_text_features = t_text_features.to(device, non_blocking=True)

                t_image_features = input_data["key_image"]
                t_image_features = t_image_features.to(device, non_blocking=True)

                if "rgbd" in accum_kd_text_features:
                    accum_kd_text_features["rgbd"].append(t_text_features)
                else:
                    accum_kd_text_features["rgbd"] = [t_text_features]

                if "rgbd" in accum_kd_image_features:
                    accum_kd_image_features["rgbd"].append(t_image_features)
                else:
                    accum_kd_image_features["rgbd"] = [t_image_features]

            depth_samples = depth_samples.to(device, non_blocking=True)
            rgb_samples = rgb_samples.to(device, non_blocking=True)
            if isinstance(rgbd_targets, list):
                rgbd_targets = torch.LongTensor(rgbd_targets)
            rgbd_targets = rgbd_targets.to(device, non_blocking=True)

            input = {
                "rgbd": depth_samples,
                "image": rgb_samples,
            }

            input_list.append(input)
            if "rgbd" in trgets_dict:
                trgets_dict["rgbd"].append(rgbd_targets)
            else:
                trgets_dict["rgbd"] = [rgbd_targets]

        elif mini_batch_modal == "video":
            cur_modal = "video"

            modal_list.append("video")
            cur_anchor = ["video"]
            anchor_list.append(["video"])
            if args.use_text_template and not args.use_text_branch:
                video_samples, video_targets, video_text_features = (
                    input_data["video"],
                    input_data["target"],
                    input_data["text_feature"],
                )

                video_text_features = video_text_features.to(device, non_blocking=True)
                if "video" in accum_text_features:
                    accum_text_features["video"].append(video_text_features)
                else:
                    accum_text_features["video"] = [video_text_features]

                video_targets = video_targets.to(device, non_blocking=True)
                if "video" in trgets_dict:
                    trgets_dict["video"].append(video_targets)
                else:
                    trgets_dict["video"] = [video_targets]
            else:
                video_texts, video_samples = input_data["caption"], input_data["video"]

                video_texts = video_texts.to(device=device, non_blocking=True)
                cur_text = video_texts

            if args.multi_modal_distill:
                t_visual_features = input_data["key"]
                t_visual_features = t_visual_features.to(device, non_blocking=True)
                if "video" in accum_kd_t_features:
                    accum_kd_t_features["video"].append(t_visual_features)
                else:
                    accum_kd_t_features["video"] = [t_visual_features]

                t_text_features = input_data["key_text"]
                t_text_features = t_text_features.to(device, non_blocking=True)

                t_image_features = input_data["key_image"]
                t_image_features = t_image_features.to(device, non_blocking=True)

                if "video" in accum_kd_text_features:
                    accum_kd_text_features["video"].append(t_text_features)
                else:
                    accum_kd_text_features["video"] = [t_text_features]

                if "video" in accum_kd_image_features:
                    accum_kd_image_features["video"].append(t_image_features)
                else:
                    accum_kd_image_features["video"] = [t_image_features]

            video_samples = video_samples.to(device, non_blocking=True)
            input = {"video": video_samples}
            input_list.append(input)

            # if args.use_text and not args.use_text_template:
            #     video_text_features = []
            #     for label in video_targets:
            #         label = str(label.item())
            #         bstr = video_manager.read(label)
            #         text_feature = np.frombuffer(
            #             bstr[:item_size], dtype=np.float16
            #         ).copy()
            #         text_feature = torch.from_numpy(text_feature).to(
            #             device=device, dtype=torch.float16
            #         )
            #         video_text_features.append(text_feature)
            #     video_text_features = torch.stack(video_text_features)
            #     if "video" in accum_text_features:
            #         accum_text_features["video"].append(video_text_features)
            #     else:
            #         accum_text_features["video"] = [video_text_features]

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % (accum_iter * modal_lens) == 0:
            lr_sched.adjust_learning_rate(
                optimizer, data_iter_step / len(train_data_loader) + epoch, args
            )

        optimizer.zero_grad()

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                output = model(
                    [input],
                    [cur_modal],
                    anchor_list=cur_anchor,
                    mask_t_prob=args.mask_t_prob,
                    mask_f_prob=args.mask_f_prob,
                    extract_feature=args.multi_modal_distill,
                )
                output.pop("logit_scale")
                output.pop("logits")

                for key, val in output["features"].items():
                    if key in accum_features:
                        for anchor, anchor_val in val.items():
                            accum_features[key][anchor].append(anchor_val)
                    else:
                        accum_features[key] = {}
                        for anchor, anchor_val in val.items():
                            accum_features[key][anchor] = [anchor_val]

                if args.multi_modal_distill:
                    for key, val in output["teacher_features"].items():
                        if key in accum_kd_s_features:
                            for anchor, anchor_val in val.items():
                                accum_kd_s_features[key][anchor].append(anchor_val)
                        else:
                            accum_kd_s_features[key] = {}
                            for anchor, anchor_val in val.items():
                                accum_kd_s_features[key][anchor] = [anchor_val]
                else:
                    output.pop("teacher_features")

                if args.use_text_branch:
                    if args.distributed:
                        text_embeds = open_clip_text_model.module.encode_text(
                            cur_text, normalize=True
                        )
                    else:
                        text_embeds = open_clip_text_model.encode_text(
                            cur_text, normalize=True
                        )
                    if cur_modal in accum_text_features:
                        accum_text_features[cur_modal].append(text_embeds)
                    else:
                        accum_text_features[cur_modal] = [text_embeds]

        del input, output
        if (data_iter_step + 1) % (accum_iter * modal_lens) > 0:
            continue

        # Now, ready to take gradients for the last accum_freq batches.
        # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
        # Call backwards each time, but only step optimizer at the end.
        optimizer.zero_grad()
        samples_text_features = {}
        for key, val in accum_text_features.items():
            samples_text_features[key] = torch.cat(val, dim=0)

        if args.multi_modal_distill:
            samples_kd_t_features = {}
            samples_kd_text_features = {}
            samples_kd_image_features = {}
            for key, val in accum_kd_t_features.items():
                samples_kd_t_features[key] = torch.cat(val, dim=0)

            for key, val in accum_kd_text_features.items():
                samples_kd_text_features[key] = torch.cat(val, dim=0)

            for key, val in accum_kd_image_features.items():
                samples_kd_image_features[key] = torch.cat(val, dim=0)

        for i in range(accum_iter):
            # loss = 0
            # loss_dict={}

            for j in range(modal_lens):
                cur_modal = modal_list[i * modal_lens + j]
                cur_anchor = anchor_list[i * modal_lens + j]
                cur_input = input_list.pop(0)
                loss = 0
                loss_dict = {}

                # if cur_modal not in align_train:
                #     continue

                with torch.cuda.amp.autocast():
                    output = model(
                        [cur_input],
                        [cur_modal],
                        anchor_list=cur_anchor,
                        mask_t_prob=args.mask_t_prob,
                        mask_f_prob=args.mask_f_prob,
                        extract_feature=args.multi_modal_distill,
                    )

                    samples_proj2text_features = {}

                    for key, val in accum_features.items():
                        proj2text_feature = {}
                        if key == cur_modal:
                            for anchor, anchor_val in val.items():
                                if len(anchor_val) > 1:
                                    proj2text_feature[anchor] = torch.cat(
                                        anchor_val[:i]
                                        + [output["features"][key][anchor]]
                                        + anchor_val[i + 1 :],
                                        dim=0,
                                    )
                                else:
                                    proj2text_feature[anchor] = output["features"][key][
                                        anchor
                                    ]
                        else:
                            for anchor, anchor_val in val.items():
                                proj2text_feature[anchor] = torch.cat(anchor_val, dim=0)

                        samples_proj2text_features[key] = proj2text_feature

                    if (
                        args.multi_modal_distill
                        and cur_modal in args.multi_modal_distill_modal_list
                    ):
                        samples_kd_s_features = {}
                        for key, val in accum_kd_s_features.items():
                            proj2text_feature = {}
                            if key == cur_modal:
                                for anchor, anchor_val in val.items():
                                    if len(anchor_val) > 1:
                                        proj2text_feature[anchor] = torch.cat(
                                            anchor_val[:i]
                                            + [output["teacher_features"][key][anchor]]
                                            + anchor_val[i + 1 :],
                                            dim=0,
                                        )
                                    else:
                                        proj2text_feature[anchor] = output[
                                            "teacher_features"
                                        ][key][anchor]
                            else:
                                for anchor, anchor_val in val.items():
                                    proj2text_feature[anchor] = torch.cat(anchor_val, dim=0)
                            samples_kd_s_features[key] = proj2text_feature

                    if has_cls_head[cur_modal]:
                        loss_cls = criterion[cur_modal](
                            output["logits"][cur_modal], trgets_dict[cur_modal][i]
                        )
                        if args.task_balancer == "uncertainty":
                            if cur_modal == "audio":
                                loss_cls = loss_cls * args.audio_weight
                            loss_dict.update({f"{cur_modal}_cls_loss": loss_cls})
                        else:
                            if cur_modal == "audio":
                                loss += loss_cls * args.audio_weight
                                metric_logger.update(
                                    **{
                                        f"{cur_modal}_cls_loss": loss_cls.item()
                                        * args.audio_weight
                                    }
                                )
                            else:
                                loss += loss_cls

                                metric_logger.update(
                                    **{f"{cur_modal}_cls_loss": loss_cls.item()}
                                )

                    if args.cross_align and cur_modal != "image":
                        if args.use_clip_loss:
                            cm_kd_loss = single_cross_modal_kd_loss_gather(
                                cur_modal,
                                modal_list,
                                samples_proj2text_features,
                                samples_text_features,
                                modal_labels_features,
                                output["logit_scale"],
                                local_loss=args.local_loss,
                                gather_with_grad=args.gather_with_grad,
                                rank=rank,
                                world_size=world_size,
                            )
                            # cm_kd_loss = single_cross_modal_kd_loss_visual(
                            #     cur_modal,
                            #     modal_list,
                            #     samples_proj2text_features,
                            #     samples_text_features,
                            #     output["logit_scale"],
                            #     local_loss=args.local_loss,
                            #     gather_with_grad=args.gather_with_grad,
                            #     rank=rank,
                            #     world_size=world_size,
                            # )
                            # plm_text_features = plm_labels_features[cur_modal][torch.concat(trgets_dict[cur_modal],dim=0)]
                            # cm_kd_loss = single_cross_modal_kd_loss_plm(
                            #     cur_modal,
                            #     modal_list,
                            #     samples_proj2text_features,
                            #     modal_labels_features,
                            #     plm_text_features,
                            #     plm_labels_features,
                            #     output["logit_scale"],
                            #     local_loss=args.local_loss,
                            #     gather_with_grad=args.gather_with_grad,
                            #     rank=rank,
                            #     world_size=world_size,
                            # )

                        else:
                            cm_kd_loss = single_cross_modal_kd_loss(
                                cur_modal,
                                modal_list,
                                samples_proj2text_features,
                                samples_text_features,
                                modal_labels_features,
                                args,
                                output["logit_scale"],
                            )

                        if args.task_balancer == "uncertainty":
                            loss_dict.update(**cm_kd_loss)
                        else:
                            for key, val in cm_kd_loss.items():
                                loss += val
                            metric_logger.update(**cm_kd_loss)

                    # if args.anchor_align:
                    #     anchor_kd_loss = single_anchor_kd_loss(
                    #         cur_modal,
                    #         modal_list,
                    #         samples_proj2text_features,
                    #         samples_text_features,
                    #         args,
                    #         output["logit_scale"],
                    #     )

                    #     for key, val in anchor_kd_loss.items():
                    #         loss += val
                    #     metric_logger.update(**anchor_kd_loss)

                    if args.uni_align and (cur_modal not in ["image"]):
                        if args.use_clip_loss:
                            uni_clip_loss = single_uni_modal_clip_loss_gather(
                                cur_modal,
                                samples_proj2text_features,
                                samples_text_features,
                                output["logit_scale"],
                                local_loss=args.local_loss,
                                gather_with_grad=args.gather_with_grad,
                                rank=rank,
                                world_size=world_size,
                                args=args,
                            )

                            # uni_clip_loss = single_uni_modal_clip_loss(
                            #     cur_modal,
                            #     samples_proj2text_features,
                            #     samples_text_features,
                            #     output["logit_scale"],
                            #     local_loss=args.local_loss,
                            #     gather_with_grad=args.gather_with_grad,
                            #     rank=rank,
                            #     world_size=world_size,
                            #     args=args,
                            # )

                        else:
                            uni_clip_loss = single_uni_modal_nce_loss(
                                cur_modal,
                                samples_proj2text_features,
                                samples_text_features,
                                output["logit_scale"],
                                epoch,
                            )

                        if args.task_balancer == "uncertainty":
                            loss_dict.update(**uni_clip_loss)
                        else:
                            for key, val in uni_clip_loss.items():
                                loss += val

                            metric_logger.update(**uni_clip_loss)

                    if args.use_orthogonal_loss:
                        orth_loss = output["orth_loss"]
                        if args.task_balancer == "uncertainty":
                            loss_dict.update(**orth_loss)
                        else:
                            for key, val in orth_loss.items():
                                loss += val
                            metric_logger.update(**orth_loss)

                    if args.use_moe_loss:
                        moe_loss = output["moe_loss"]
                        if args.task_balancer == "uncertainty":
                            loss_dict.update(**moe_loss)
                        else:
                            for key, val in moe_loss.items():
                                loss += val
                            metric_logger.update(**moe_loss)

                    if args.use_aux_cls_loss and (
                        cur_modal not in ["image", "video", "point"]
                    ):
                        if args.use_clip_loss:
                            cm_cls_loss, cm_cls_acc = cross_modal_cls_gather(
                                cur_modal,
                                output["features"],
                                modal_labels_features,
                                trgets_dict[cur_modal][i],
                                output["logit_scale"],
                                local_loss=args.local_loss,
                                gather_with_grad=args.gather_with_grad,
                                rank=rank,
                                world_size=world_size,
                            )
                        else:
                            cm_cls_loss, cm_cls_acc = cross_modal_cls(
                                cur_modal,
                                output["features"],
                                modal_labels_features,
                                trgets_dict[cur_modal][i],
                                output["logit_scale"],
                            )
                        if args.task_balancer == "uncertainty":
                            loss_dict.update(**cm_cls_loss)
                        else:
                            for key, val in cm_cls_loss.items():
                                loss += val
                            metric_logger.update(**cm_cls_loss)
                        metric_logger.update(**cm_cls_acc)

                    if (
                        args.multi_modal_distill
                        and cur_modal in args.multi_modal_distill_modal_list
                    ):
                        if args.use_clip_loss:
                            kd_t_loss = single_kd_clip_loss_gather(
                                cur_modal,
                                samples_proj2text_features,
                                samples_text_features,
                                samples_kd_s_features,
                                samples_kd_t_features,
                                samples_kd_text_features,
                                samples_kd_image_features,
                                output["logit_scale"],
                                local_loss=args.local_loss,
                                gather_with_grad=args.gather_with_grad,
                                rank=rank,
                                world_size=world_size,
                                args=args,
                            )
                        else:
                            kd_t_loss = single_kd_clip_loss(
                                cur_modal,
                                samples_kd_s_features,
                                samples_kd_t_features,
                                args,
                            )
                        if args.task_balancer == "uncertainty":
                            loss_dict.update(**kd_t_loss)
                        else:
                            for key, val in kd_t_loss.items():
                                loss += val
                            metric_logger.update(**kd_t_loss)

                    # weighted_task_losses = loss_balancer.module[cur_modal](loss_dict)
                    # metric_logger.update(**weighted_task_losses)
                    # loss = sum(weighted_task_losses.values())
                    if args.task_balancer == "uncertainty":
                        weighted_task_losses, loss = loss_balancer(cur_modal, loss_dict)
                        metric_logger.update(**weighted_task_losses)

                    del samples_proj2text_features
                    loss_value = loss.item()

                    if not math.isfinite(loss_value):
                        print_log(
                            f"{cur_modal} Loss is {loss_value}, stopping training",
                            logger,
                        )
                        # continue
                        sys.exit(1)

                    # loss_dict.update({f'{cur_modal}_loss': loss_value})

                    # loss /= accum_iter

                if math.isfinite(loss_value):
                    modal_loss = {f"{cur_modal}_loss": loss_value}
                    metric_logger.update(**modal_loss)
                    loss_scaler(
                        loss,
                        optimizer,
                        clip_grad=max_norm,
                        parameters=model.parameters(),
                        create_graph=False,
                        update_grad=(data_iter_step + 1) % (accum_iter * modal_lens)
                        == 0,
                    )

                    torch.cuda.synchronize()

            # loss_dict = {f'{cur_modal}_loss': loss_value}
            # metric_logger.update(**loss_dict)
            # metric_logger.update(loss=loss_value)

        trgets_dict = {}
        modal_list = []
        input_list = []
        anchor_list = []

        accum_features = {}
        accum_text_features = {}
        accum_kd_s_features = {}
        accum_kd_t_features = {}
        accum_kd_text_features = {}
        accum_kd_image_features = {}

        min_lr = 10.0
        max_lr = 0.0
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        # loss_value_reduce = misc.all_reduce_mean(loss_value)

        # if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
        #     """ We use epoch_1000x as the x-axis in tensorboard.
        #     This calibrates different curves when batch size changes.
        #     """
        #     epoch_1000x = int((data_iter_step / len(train_data_loader) + epoch) * 1000)
        #     log_writer.add_scalar("loss", loss_value_reduce, epoch_1000x)
        #     log_writer.add_scalar("lr", max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print_log(f"Averaged stats:{metric_logger}", logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch_concat_use_all(
    model: torch.nn.Module,
    criterion: dict,
    train_data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = 0,
    mixup_fn: Optional[Mixup] = None,
    log_writer=None,
    args=None,
):
    model.train(True)
    logger = get_logger(args.log_name)
    metric_logger = misc.MetricLogger(delimiter=" ", logger=logger)
    header = "Multi Tasks Merging with use_all mode ==== Epoch: [{}]".format(epoch)
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))

    print_freq = 200

    accum_iter = args.accum_iter

    if log_writer is not None:
        print("log_dir: {}".format(log_writer.log_dir))

    # data_iter_step = 0

    npoints = args.pc_n_points
    if npoints == 1024:
        point_all = 1200
    elif npoints == 2048:
        point_all = 2400
    elif npoints == 4096:
        point_all = 4800
    elif npoints == 8192:
        point_all = 8192
    else:
        raise NotImplementedError()

    world_size = misc.get_world_size()
    rank = misc.get_rank()
    item_size = args.text_embed_dim * 2
    modal_labels_features = {}
    for m in args.train_modal_list:
        if m == "image":
            image_manager = TxtManager(args.img_text_feature_path, item_size, rank)
            image_labels_features = []
            for label in range(args.nb_classes):
                label = str(label)
                bstr = image_manager.read(label)
                text_feature = np.frombuffer(bstr[:item_size], dtype=np.float16).copy()
                text_feature = torch.from_numpy(text_feature).to(
                    device=device, dtype=torch.float16
                )
                image_labels_features.append(text_feature)
            image_labels_features = torch.stack(image_labels_features)
            modal_labels_features["image"] = image_labels_features

        if m == "audio":
            audio_manager = TxtManager(args.audio_text_feature_path, item_size, rank)
            audio_labels_features = []
            for label in range(args.audio_nb_classes):
                label = str(label)
                bstr = audio_manager.read(label)
                text_feature = np.frombuffer(bstr[:item_size], dtype=np.float16).copy()
                text_feature = torch.from_numpy(text_feature).to(
                    device=device, dtype=torch.float16
                )
                audio_labels_features.append(text_feature)
            audio_labels_features = torch.stack(audio_labels_features)
            modal_labels_features["audio"] = audio_labels_features

        if m == "point":
            point_manager = TxtManager(args.point_text_feature_path, item_size, rank)
            point_labels_features = []
            for label in range(args.pc_nb_classes):
                label = str(label)
                bstr = point_manager.read(label)
                text_feature = np.frombuffer(bstr[:item_size], dtype=np.float16).copy()
                text_feature = torch.from_numpy(text_feature).to(
                    device=device, dtype=torch.float16
                )
                point_labels_features.append(text_feature)
            point_labels_features = torch.stack(point_labels_features)
            modal_labels_features["point"] = point_labels_features

        if m == "video":
            video_manager = TxtManager(args.video_text_feature_path, item_size, rank)
            video_labels_features = []
            for label in range(args.video_nb_classes):
                label = str(label)
                bstr = video_manager.read(label)
                text_feature = np.frombuffer(bstr[:item_size], dtype=np.float16).copy()
                text_feature = torch.from_numpy(text_feature).to(
                    device=device, dtype=torch.float16
                )
                video_labels_features.append(text_feature)
            video_labels_features = torch.stack(video_labels_features)
            modal_labels_features["video"] = video_labels_features

        if m == "rgbd":
            rgbd_labels_features = []
            anno_path = "data/sunrgbd/OFFICIAL_SUNRGBD/SUN-RGBD_train.json"
            annotation = json.load(open(anno_path, "r"))
            labelset = set()
            for ann in annotation:
                cleaned_label = ann["cleaned_label"]
                labelset.add(cleaned_label)
            idx2label = list(labelset)
            rgbd_val_text_features = torch.load(args.rgbd_train_text_feature_path)
            for label in idx2label:
                text_feature = rgbd_val_text_features[label].to(
                    device=device, dtype=torch.float16
                )
                rgbd_labels_features.append(text_feature)
            rgbd_labels_features = torch.stack(rgbd_labels_features)
            modal_labels_features["rgbd"] = rgbd_labels_features

    trgets_dict = {}

    accum_features = {}
    accum_text_features = {}
    accum_input_list = []
    accum_modal_list = []

    for data_iter_step, batch in enumerate(
        metric_logger.log_every(
            train_data_loader, print_freq, header, pre_iter=int(accum_iter)
        )
    ):
        modal_list = []

        input_list = []

        for i, samples in enumerate(batch):
            (input_data, mini_batch_modal) = samples

            if mini_batch_modal[0] == "image":
                if "image" not in modal_list:
                    modal_list.append("image")

                mini_img_samples, mini_img_targets = input_data
                mini_img_samples = mini_img_samples.to(device, non_blocking=True)
                mini_img_targets = mini_img_targets.to(device, non_blocking=True)

                if mixup_fn is not None:
                    mini_img_samples, mini_img_targets = mixup_fn(
                        mini_img_samples, mini_img_targets
                    )

                # if img_samples is None:
                #     img_samples = mini_img_samples
                #     img_targets = mini_img_targets
                # else:
                #     img_samples = torch.cat((img_samples, mini_img_samples), dim=0)
                #     img_targets = torch.cat((img_targets, mini_img_targets), dim=0)

                input_list.append(mini_img_samples)
                if "image" in trgets_dict:
                    trgets_dict["image"].append(mini_img_targets)
                else:
                    trgets_dict["image"] = [mini_img_targets]

                if args.use_text:
                    image_text_features = []
                    for label in mini_img_targets:
                        label = str(label.item())
                        bstr = image_manager.read(label)
                        text_feature = np.frombuffer(
                            bstr[:item_size], dtype=np.float16
                        ).copy()
                        text_feature = torch.from_numpy(text_feature).to(
                            device=device, dtype=torch.float16
                        )
                        image_text_features.append(text_feature)
                    image_text_features = torch.stack(image_text_features)
                    if "image" in accum_text_features:
                        accum_text_features["image"].append(image_text_features)
                    else:
                        accum_text_features["image"] = [image_text_features]

            elif mini_batch_modal[0] == "audio":
                if "audio" not in modal_list:
                    modal_list.append("audio")
                if args.use_text_template:
                    (
                        mini_audio_samples,
                        mini_audio_targets,
                        _vids,
                        audio_text_features,
                    ) = input_data
                    audio_text_features = audio_text_features.to(
                        device, non_blocking=True
                    )
                    if "audio" in accum_text_features:
                        accum_text_features["audio"].append(audio_text_features)
                    else:
                        accum_text_features["audio"] = [audio_text_features]
                else:
                    mini_audio_samples, mini_audio_targets, _vids = input_data
                mini_audio_samples = mini_audio_samples.to(device, non_blocking=True)
                mini_audio_targets = mini_audio_targets.to(device, non_blocking=True)

                # if audio_samples is None:
                #     audio_samples = mini_audio_samples
                #     audio_targets = mini_audio_targets
                # else:
                #     audio_samples = torch.cat(
                #         (audio_samples, mini_audio_samples), dim=0
                #     )
                #     audio_targets = torch.cat(
                #         (audio_targets, mini_audio_targets), dim=0
                #     )

                input_list.append(mini_audio_samples)
                if "audio" in trgets_dict:
                    trgets_dict["audio"].append(mini_audio_targets)
                else:
                    trgets_dict["audio"] = [mini_audio_targets]

                if args.use_text and not args.use_text_template:
                    audio_text_features = []
                    for label in mini_audio_targets:
                        label = str(label.argmax(dim=-1).item())
                        bstr = audio_manager.read(label)
                        text_feature = np.frombuffer(
                            bstr[:item_size], dtype=np.float16
                        ).copy()
                        text_feature = torch.from_numpy(text_feature).to(
                            device=device, dtype=torch.float16
                        )
                        audio_text_features.append(text_feature)
                    audio_text_features = torch.stack(audio_text_features)
                    if "audio" in accum_text_features:
                        accum_text_features["audio"].append(audio_text_features)
                    else:
                        accum_text_features["audio"] = [audio_text_features]

            elif mini_batch_modal[0] == "point":
                if "point" not in modal_list:
                    modal_list.append("point")

                if args.use_text_template:
                    mini_points, mini_pc_targets, point_text_features = input_data
                    point_text_features = point_text_features.to(
                        device, non_blocking=True
                    )
                    if "point" in accum_text_features:
                        accum_text_features["point"].append(point_text_features)
                    else:
                        accum_text_features["point"] = [point_text_features]
                else:
                    mini_points, mini_pc_targets = input_data
                mini_points = mini_points.to(device, non_blocking=True)
                mini_pc_targets = mini_pc_targets.to(device, non_blocking=True)

                if mini_points.size(1) < point_all:
                    point_all = mini_points.size(1)

                fps_idx = pointnet2_utils.furthest_point_sample(
                    mini_points, point_all
                )  # (B, npoint)
                fps_idx = fps_idx[:, np.random.choice(point_all, npoints, False)]
                mini_points = (
                    pointnet2_utils.gather_operation(
                        mini_points.transpose(1, 2).contiguous(), fps_idx
                    )
                    .transpose(1, 2)
                    .contiguous()
                )  # (B, N, 3)
                # import pdb; pdb.set_trace()
                mini_points = pc_train_transforms(mini_points)
                # if points is None:
                #     points = mini_points
                #     pc_targets = mini_pc_targets
                # else:
                #     points = torch.cat((points, mini_points), dim=0)
                #     pc_targets = torch.cat((pc_targets, mini_pc_targets), dim=0)

                input_list.append(mini_points)
                if "point" in trgets_dict:
                    trgets_dict["point"].append(mini_pc_targets.long())
                else:
                    trgets_dict["point"] = [mini_pc_targets.long()]

                if args.use_text and not args.use_text_template:
                    point_text_features = []
                    for label in mini_pc_targets:
                        label = str(label.item())
                        bstr = point_manager.read(label)
                        text_feature = np.frombuffer(
                            bstr[:item_size], dtype=np.float16
                        ).copy()
                        text_feature = torch.from_numpy(text_feature).to(
                            device=device, dtype=torch.float16
                        )
                        point_text_features.append(text_feature)
                    point_text_features = torch.stack(point_text_features)
                    if "point" in accum_text_features:
                        accum_text_features["point"].append(point_text_features)
                    else:
                        accum_text_features["point"] = [point_text_features]

            elif mini_batch_modal[0] == "video":
                if "video" not in modal_list:
                    modal_list.append("video")

                if args.use_text_template:
                    mini_video_samples, mini_video_targets, _, video_text_features = (
                        input_data
                    )

                    video_text_features = video_text_features.to(
                        device, non_blocking=True
                    )
                    if "video" in accum_text_features:
                        accum_text_features["video"].append(video_text_features)
                    else:
                        accum_text_features["video"] = [video_text_features]
                else:
                    mini_video_samples, mini_video_targets, _, _ = input_data
                mini_video_samples = mini_video_samples.to(device, non_blocking=True)
                mini_video_targets = mini_video_targets.to(device, non_blocking=True)

                # if video_samples is None:
                #     video_samples = mini_video_samples
                #     video_targets = mini_video_targets
                # else:
                #     video_samples = torch.cat(
                #         (video_samples, mini_video_samples), dim=0
                #     )
                #     video_targets = torch.cat(
                #         (video_targets, mini_video_targets), dim=0
                #     )

                input_list.append(mini_video_samples)
                if "video" in trgets_dict:
                    trgets_dict["video"].append(mini_video_targets)
                else:
                    trgets_dict["video"] = [mini_video_targets]

                if args.use_text and not args.use_text_template:
                    video_text_features = []
                    for label in mini_video_targets:
                        label = str(label.item())
                        bstr = video_manager.read(label)
                        text_feature = np.frombuffer(
                            bstr[:item_size], dtype=np.float16
                        ).copy()
                        text_feature = torch.from_numpy(text_feature).to(
                            device=device, dtype=torch.float16
                        )
                        video_text_features.append(text_feature)
                    video_text_features = torch.stack(video_text_features)
                    if "video" in accum_text_features:
                        accum_text_features["video"].append(video_text_features)
                    else:
                        accum_text_features["video"] = [video_text_features]

            elif mini_batch_modal[0] == "rgbd":
                if "rgbd" not in modal_list:
                    modal_list.append("rgbd")
                if args.use_text_template:
                    mini_rgbd_samples, mini_rgbd_targets, rgbd_text_features = (
                        input_data
                    )

                    rgbd_text_features = rgbd_text_features.to(
                        device, non_blocking=True
                    )
                    if "rgbd" in accum_text_features:
                        accum_text_features["rgbd"].append(rgbd_text_features)
                    else:
                        accum_text_features["rgbd"] = [rgbd_text_features]
                else:
                    mini_rgbd_samples, mini_rgbd_targets = input_data
                # if args.use_depth_only:
                #     mini_rgbd_samples = mini_rgbd_samples[:, 3:, ...]
                mini_rgbd_samples = mini_rgbd_samples.to(device, non_blocking=True)
                mini_rgbd_targets = mini_rgbd_targets.to(device, non_blocking=True)

                # if rgbd_samples is None:
                #     rgbd_samples = mini_rgbd_samples
                #     rgbd_targets = mini_rgbd_targets
                # else:
                #     rgbd_samples = torch.cat((rgbd_samples, mini_rgbd_samples), dim=0)
                #     rgbd_targets = torch.cat((rgbd_targets, mini_rgbd_targets), dim=0)

                input_list.append(mini_rgbd_samples)
                if "rgbd" in trgets_dict:
                    trgets_dict["rgbd"].append(mini_rgbd_targets)
                else:
                    trgets_dict["rgbd"] = [mini_rgbd_targets]

                if args.use_text and not args.use_text_template:
                    rgbd_text_features = []
                    for label in mini_rgbd_targets:
                        text_feature = modal_labels_features["rgbd"][label]
                        rgbd_text_features.append(text_feature)
                    rgbd_text_features = torch.stack(rgbd_text_features)
                    if "rgbd" in accum_text_features:
                        accum_text_features["rgbd"].append(rgbd_text_features)
                    else:
                        accum_text_features["rgbd"] = [rgbd_text_features]

        assert len(input_list) == len(modal_list)
        accum_input_list.append(input_list)
        accum_modal_list.append(modal_list)

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer, data_iter_step / len(train_data_loader) + epoch, args
            )

        optimizer.zero_grad()

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                output = model(
                    input_list,
                    modal_list,
                    mask_t_prob=args.mask_t_prob,
                    mask_f_prob=args.mask_f_prob,
                    use_text=args.use_text,
                )

                for key, val in output["features"].items():
                    if key == "rgbd":
                        if key in accum_features:
                            accum_features[key]["image"].append(val["image"])
                            accum_features[key]["rgbd"].append(val["rgbd"])
                        else:
                            accum_features[key] = {
                                "image": [val["image"]],
                                "rgbd": [val["rgbd"]],
                            }
                    else:
                        if key in accum_features:
                            accum_features[key].append(val)
                        else:
                            accum_features[key] = [val]

        del input_list, output
        if (data_iter_step + 1) % (accum_iter) > 0:
            continue

        optimizer.zero_grad()

        samples_text_features = {}
        for key, val in accum_text_features.items():
            samples_text_features[key] = torch.cat(val, dim=0)

        for j in range(args.accum_iter):
            loss = 0
            with torch.cuda.amp.autocast():
                output = model(
                    accum_input_list[j],
                    accum_modal_list[j],
                    mask_t_prob=args.mask_t_prob,
                    mask_f_prob=args.mask_f_prob,
                    use_text=args.use_text,
                )

                samples_proj2text_features = {}

                for key, val in accum_features.items():
                    if key == "rgbd":
                        if len(val) > 1:
                            accum_depth_features = torch.cat(
                                val["rgbd"][:j]
                                + [output["features"][key]["rgbd"]]
                                + val["rgbd"][j + 1 :],
                                dim=0,
                            )
                            accum_rgb_features = torch.cat(
                                val["image"][:j]
                                + [output["features"][key]["image"]]
                                + val["image"][j + 1 :],
                                dim=0,
                            )

                            proj2text_feature = {
                                "rgbd": accum_depth_features,
                                "image": accum_rgb_features,
                            }
                        else:
                            accum_depth_features = torch.cat(
                                val["rgbd"][:j]
                                + [output["features"][key]["rgbd"]]
                                + val["rgbd"][j + 1 :],
                                dim=0,
                            )
                            proj2text_feature = {"rgbd": accum_depth_features}

                    else:
                        proj2text_feature = torch.cat(
                            val[:j] + [output["features"][key]] + val[j + 1 :],
                            dim=0,
                        )

                    samples_proj2text_features[key] = proj2text_feature

                for modal in accum_modal_list[j]:
                    if modal != "rgbd":
                        loss_cls = criterion[modal](
                            output["logits"][modal], trgets_dict[modal][j]
                        )
                        if modal == "audio":
                            loss += loss_cls * args.audio_weight
                        else:
                            loss += loss_cls

                        metric_logger.update(**{f"{modal}_cls_loss": loss_cls.item()})

                if args.use_text:
                    if args.cross_align:
                        if world_size > 1:
                            cm_kd_loss = multi_cross_modal_kd_loss_gather(
                                accum_modal_list[j],
                                samples_proj2text_features,
                                samples_text_features,
                                modal_labels_features,
                                output["logit_scale"],
                                local_loss=args.local_loss,
                                gather_with_grad=args.gather_with_grad,
                                rank=rank,
                                world_size=world_size,
                            )
                        else:
                            cm_kd_loss = multi_cross_modal_kd_loss(
                                accum_modal_list[j],
                                samples_proj2text_features,
                                samples_text_features,
                                modal_labels_features,
                                args,
                                output["logit_scale"],
                            )

                        for key, val in cm_kd_loss.items():
                            loss += val
                        metric_logger.update(**cm_kd_loss)

                    if args.uni_align:
                        if world_size > 1:
                            uni_clip_loss = multi_uni_modal_clip_loss_gather(
                                accum_modal_list[j],
                                samples_proj2text_features,
                                samples_text_features,
                                output["logit_scale"],
                                epoch,
                                local_loss=args.local_loss,
                                gather_with_grad=args.gather_with_grad,
                                rank=rank,
                                world_size=world_size,
                            )
                        else:
                            uni_clip_loss = multi_uni_modal_nce_loss(
                                accum_modal_list[j],
                                samples_proj2text_features,
                                samples_text_features,
                                output["logit_scale"],
                                epoch,
                            )

                        for key, val in uni_clip_loss.items():
                            loss += val

                        metric_logger.update(**uni_clip_loss)

                        if world_size > 1:
                            cm_cls_loss, cm_cls_acc = multi_cross_modal_cls_gather(
                                accum_modal_list[j],
                                output["features"],
                                modal_labels_features,
                                trgets_dict,
                                output["logit_scale"],
                                local_loss=args.local_loss,
                                gather_with_grad=args.gather_with_grad,
                                rank=rank,
                                world_size=world_size,
                                accum_iter=j,
                            )
                        else:
                            cm_cls_loss, cm_cls_acc = multi_cross_modal_cls(
                                accum_modal_list[j],
                                output["features"],
                                modal_labels_features,
                                trgets_dict,
                                output["logit_scale"],
                                j,
                            )

                        for key, val in cm_cls_loss.items():
                            loss += val

                        metric_logger.update(**cm_cls_loss)
                        metric_logger.update(**cm_cls_acc)

                del samples_proj2text_features
                loss_value = loss.item()
                if not math.isfinite(loss_value):
                    print_log(
                        f"Loss is {loss_value}, stopping training",
                        logger,
                    )
                    sys.exit(1)

            if math.isfinite(loss_value):
                metric_logger.update(loss=loss_value)
                loss_scaler(
                    loss,
                    optimizer,
                    clip_grad=max_norm,
                    parameters=model.parameters(),
                    create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0,
                )
                torch.cuda.synchronize()

        # metric_logger.update(loss=loss_value)

        # if args.use_loramoe:
        #     metric_logger.update(blcls=blcls.item())
        trgets_dict = {}

        accum_features = {}
        accum_text_features = {}
        accum_input_list = []
        accum_modal_list = []

        min_lr = 10.0
        max_lr = 0.0
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)

        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(train_data_loader) + epoch) * 1000)
            log_writer.add_scalar("loss", loss_value_reduce, epoch_1000x)
            log_writer.add_scalar("lr", max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print_log(f"Averaged stats:{metric_logger}", logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(
    img_data_loader,
    audio_data_loader,
    pc_data_loader,
    model,
    device,
    args,
):
    criterion = torch.nn.CrossEntropyLoss()

    img_metric_logger = misc.MetricLogger(delimiter="  ")
    img_header = "Image Task Test:"

    audio_metric_logger = misc.MetricLogger(delimiter="  ")
    audio_header = "Audio Task Test:"

    pc_metric_logger = misc.MetricLogger(delimiter="  ")
    pc_header = "Point Cloud Task Test:"

    # switch to evaluation mode
    model.eval()

    npoints = args.pc_n_points
    audio_outputs_all = []
    audio_targets_all = []
    vids = []

    pc_test_pred = []
    pc_test_label = []

    for img_samples, img_targets in img_metric_logger.log_every(
        img_data_loader, 100, img_header
    ):
        img_samples = img_samples.to(device, non_blocking=True)
        img_targets = img_targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            # if args.use_loramoe:
            #     output_img, _, _, _ = model(img_samples, None, None)
            # else:
            #     output_img, _, _ = model(img_samples, None, None)
            output = model(
                [img_samples],
                ["image"],
            )

            loss = criterion(output["logits"]["image"], img_targets)
        acc1, acc5 = accuracy(output["logits"]["image"], img_targets, topk=(1, 5))
        batch_size = img_samples.shape[0]
        img_metric_logger.update(loss=loss.item())
        img_metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        img_metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)

    for audio_samples, audio_targets, vid in audio_metric_logger.log_every(
        audio_data_loader, 100, audio_header
    ):
        audio_samples = audio_samples.to(device, non_blocking=True)
        audio_targets = audio_targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            # if args.use_loramoe:
            #     _, output_audio, _, _ = model(None, audio_samples, None)
            # else:
            #     _, output_audio, _ = model(None, audio_samples, None)
            output = model(
                [audio_samples],
                ["audio"],
            )

            if args.dist_eval:
                output_audio = concat_all_gather(output["logits"]["audio"])
                audio_targets = concat_all_gather(audio_targets)
            audio_outputs_all.append(output_audio)
            audio_targets_all.append(audio_targets)
            vids.append(vid)

    # audio
    audio_outputs_all = torch.cat(audio_outputs_all).cpu().numpy()
    audio_targets_all = torch.cat(audio_targets_all).cpu().numpy()
    # vids = [j for sub in vids for j in sub]
    # np.save(
    #     "inf_output.npy",
    #     {"vids": vids, "embs_527": audio_outputs_all, "targets": audio_targets_all},
    # )

    stats = calculate_stats(audio_outputs_all, audio_targets_all)
    if args.audio_dataset == "audioset":
        mAP = np.mean([stat["AP"] for stat in stats])
    elif args.audio_dataset == "speechcommands":
        audio_acc = stats[0]["acc"]
        mAUC = np.mean([stat["auc"] for stat in stats])

    for points, pc_targets in pc_metric_logger.log_every(pc_data_loader, 30, pc_header):
        points = points.to(device, non_blocking=True)
        pc_targets = pc_targets.to(device, non_blocking=True)

        points = misc.fps(points, npoints)

        with torch.cuda.amp.autocast():
            # if args.use_loramoe:
            #     _, _, output_pc, _ = model(None, None, points)
            # else:
            #     _, _, output_pc = model(None, None, points)
            output = model(
                [points],
                ["point"],
            )

        pc_targets = pc_targets.view(-1)
        pc_pred = output["logits"]["point"].argmax(-1).view(-1)
        pc_test_pred.append(pc_pred.detach())
        pc_test_label.append(pc_targets.detach())

    # point cloud
    pc_test_pred = torch.cat(pc_test_pred, dim=0)
    pc_test_label = torch.cat(pc_test_label, dim=0)
    if args.dist_eval:
        pc_test_pred = concat_all_gather(pc_test_pred)
        pc_test_label = concat_all_gather(pc_test_label)

    pc_acc = (
        (pc_test_pred == pc_test_label).sum() / float(pc_test_label.size(0)) * 100.0
    ).item()

    # gather the stats from all processes
    img_metric_logger.synchronize_between_processes()
    audio_metric_logger.synchronize_between_processes()
    pc_metric_logger.synchronize_between_processes()
    if args.audio_dataset == "audioset":
        print(
            "* img_Acc@1 {top1.global_avg:.3f} img_Acc@5 {top5.global_avg:.3f} img_loss {losses.global_avg:.3f} audio_mAP: {mAP:.6f} pc_acc: {pc_acc:.4f}".format(
                top1=img_metric_logger.acc1,
                top5=img_metric_logger.acc5,
                losses=img_metric_logger.loss,
                mAP=mAP,
                pc_acc=pc_acc,
            )
        )
        dict_output = {
            k: meter.global_avg for k, meter in img_metric_logger.meters.items()
        }
        dict_output["pc_acc"] = pc_acc
        dict_output["mAP"] = mAP
    elif args.audio_dataset == "speechcommands":
        print(
            "* img_Acc@1 {top1.global_avg:.3f} img_Acc@5 {top5.global_avg:.3f} img_loss {losses.global_avg:.3f} audio_acc: {audio_acc:.6f} audio_mAUC: {mAUC:.6f} pc_acc: {pc_acc:.4f}".format(
                top1=img_metric_logger.acc1,
                top5=img_metric_logger.acc5,
                losses=img_metric_logger.loss,
                audio_acc=audio_acc,
                mAUC=mAUC,
                pc_acc=pc_acc,
            )
        )
        dict_output = {
            k: meter.global_avg for k, meter in img_metric_logger.meters.items()
        }
        dict_output["pc_acc"] = pc_acc
        dict_output["audio_acc"] = audio_acc
        dict_output["mAUC"] = mAUC

    return dict_output


@torch.no_grad()
def evaluate_image(
    img_data_loader,
    model,
    open_clip_text_model,
    tokenizer,
    device,
    args,
):
    logger = get_logger(args.log_name)
    img_metric_logger = misc.MetricLogger(delimiter="  ", logger=logger)

    img_header = "Image Task Test:"

    # switch to evaluation mode
    model.eval()
    # rank = get_rank()
    # if args.text_embed_dim == 1536:
    #     item_size = args.text_embed_dim * 2
    #     image_manager = TxtManager(args.img_text_feature_path, item_size, rank)
    #     # audio_manager = TxtManager(args.audio_text_feature_path, item_size, rank)
    #     # point_manager = TxtManager(args.point_text_feature_path, item_size, rank)

    #     image_labels_features = []
    #     for label in range(args.nb_classes):
    #         label = str(label)
    #         bstr = image_manager.read(label)
    #         text_feature = np.frombuffer(bstr[:item_size], dtype=np.float16).copy()
    #         text_feature = torch.from_numpy(text_feature).to(
    #             device=device, dtype=torch.float16
    #         )
    #         image_labels_features.append(text_feature)
    #     image_labels_features = torch.stack(image_labels_features)
    # else:
    #     classifier = build_zero_shot_classifier(
    #         open_clip_text_model.module if args.distributed else open_clip_text_model,
    #         tokenizer=tokenizer,
    #         classnames=IMAGENET_CLASSNAMES,
    #         templates=OPENAI_IMAGENET_TEMPLATES,
    #         num_classes_per_batch=10,
    #         device=device,
    #         use_tqdm=True,
    #     )
    #     image_labels_features = classifier.T
    #     image_labels_features = image_labels_features.to(device=device, dtype=torch.float16)

    # image_clip_pred = []
    # image_clip_labels = []

    for input_data in img_metric_logger.log_every(img_data_loader, 100, img_header):
        img_samples = input_data["image"].to(device, non_blocking=True)
        img_targets = input_data["target"]
        if isinstance(img_targets, list):
            img_targets = torch.LongTensor(img_targets)
        img_targets = img_targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            output = model(
                [{"image": img_samples}],
                ["image"],
                ["image"],
            )

        acc1, acc5 = accuracy(output["logits"]["image"], img_targets, topk=(1, 5))
        batch_size = img_samples.shape[0]
        img_metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        img_metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)

        # image_features = output["features"]["image"]["image"]

        # similarity = (100.0 * image_features @ image_labels_features.T).softmax(dim=-1)

        # clip_pred = similarity.argmax(dim=1)
        # clip_labels = img_targets

        # acc1, acc5 = accuracy(similarity, img_targets, topk=(1, 5))
        # batch_size = img_samples.shape[0]
        # img_metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        # img_metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)

        # if args.dist_eval:
        #     clip_pred = concat_all_gather(clip_pred)
        #     clip_labels = concat_all_gather(clip_labels)

        # image_clip_pred.append(clip_pred)
        # image_clip_labels.append(clip_labels)

    # image_clip_pred = torch.cat(image_clip_pred).cpu()
    # image_clip_labels = torch.cat(image_clip_labels).cpu()

    # image_clip_acc = (image_clip_pred == image_clip_labels).float().mean()

    # gather the stats from all processes
    img_metric_logger.synchronize_between_processes()

    # print_log(
    #     "* img_Acc@1 {top1.global_avg:.3f} img_Acc@5 {top5.global_avg:.3f} image_clip_acc {image_clip_acc:.4f}".format(
    #         top1=img_metric_logger.acc1,
    #         top5=img_metric_logger.acc5,
    #         image_clip_acc=image_clip_acc.item(),
    #     ),
    #     logger=logger,
    # )
    print_log(
        "* img_Acc@1 {top1.global_avg:.3f} img_Acc@5 {top5.global_avg:.3f} ".format(
            top1=img_metric_logger.acc1,
            top5=img_metric_logger.acc5,
        ),
        logger=logger,
    )

    dict_output = {k: meter.global_avg for k, meter in img_metric_logger.meters.items()}

    return dict_output


@torch.no_grad()
def evaluate_audio(
    audio_data_loader,
    model,
    device,
    args,
):
    logger = get_logger(args.log_name)
    audio_metric_logger = misc.MetricLogger(delimiter="  ", logger=logger)
    audio_header = "Audio Task Test:"

    # switch to evaluation mode
    model.eval()

    audio_outputs_all = []
    audio_targets_all = []
    # vids = []

    audio_clip_pred = []
    audio_clip_labels = []

    rank = get_rank()
    item_size = args.text_embed_dim * 2
    # image_manager = TxtManager(args.img_text_feature_path, item_size, rank)
    audio_manager = TxtManager(args.audio_text_feature_path, item_size, rank)
    # point_manager = TxtManager(args.point_text_feature_path, item_size, rank)

    # image_labels_features=[]
    # for label in range(1000):
    #     label =str(label)
    #     bstr = image_manager.read(label)
    #     text_feature = np.frombuffer(bstr[: item_size], dtype=np.float16).copy()
    #     text_feature = torch.from_numpy(text_feature).to(device=device, dtype=torch.float16)
    #     image_labels_features.append(text_feature)
    # image_labels_features = torch.stack(image_labels_features)

    audio_labels_features = []
    for label in range(args.audio_nb_classes):
        label = str(label)
        bstr = audio_manager.read(label)
        text_feature = np.frombuffer(bstr[:item_size], dtype=np.float16).copy()
        text_feature = torch.from_numpy(text_feature).to(
            device=device, dtype=torch.float16
        )
        audio_labels_features.append(text_feature)
    audio_labels_features = torch.stack(audio_labels_features)

    # point_labels_features=[]
    # for label in range(40):
    #     label =str(label)
    #     bstr = point_manager.read(label)
    #     text_feature = np.frombuffer(bstr[: item_size], dtype=np.float16).copy()
    #     text_feature = torch.from_numpy(text_feature).to(device=device, dtype=torch.float16)
    #     point_labels_features.append(text_feature)
    # point_labels_features = torch.stack(point_labels_features)

    for audio_samples, audio_targets, vid in audio_metric_logger.log_every(
        audio_data_loader, 100, audio_header
    ):
        audio_samples = audio_samples.to(device, non_blocking=True)
        audio_targets = audio_targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            output = model(
                [{"audio": audio_samples}],
                ["audio"],
                ["audio"],
                use_text=True,
            )

        audio_features = output["features"]["audio"]["audio"]
        # audio_features /= audio_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * audio_features @ audio_labels_features.T).softmax(dim=-1)
        # clip_accuracy = (similarity.argmax(dim=1) == audio_targets.argmax(dim=1)).float().mean()
        # audio_metric_logger.update(audio_clip_accuracy=clip_accuracy)
        clip_pred = similarity.argmax(dim=1)
        clip_labels = audio_targets.argmax(dim=1)

        if args.dist_eval:
            output_audio = concat_all_gather(output["logits"]["audio"])
            audio_targets = concat_all_gather(audio_targets)
            clip_pred = concat_all_gather(clip_pred)
            clip_labels = concat_all_gather(clip_labels)

        audio_outputs_all.append(output_audio)
        audio_targets_all.append(audio_targets)
        # vids.append(vid)

        audio_clip_pred.append(clip_pred)
        audio_clip_labels.append(clip_labels)

    # audio
    audio_outputs_all = torch.cat(audio_outputs_all).cpu().numpy()
    audio_targets_all = torch.cat(audio_targets_all).cpu().numpy()

    audio_clip_pred = torch.cat(audio_clip_pred).cpu()
    audio_clip_labels = torch.cat(audio_clip_labels).cpu()
    audio_clip_acc = (audio_clip_pred == audio_clip_labels).float().mean()

    # vids = [j for sub in vids for j in sub]
    # np.save(
    #     "inf_output.npy",
    #     {"vids": vids, "embs_527": audio_outputs_all, "targets": audio_targets_all},
    # )

    stats = calculate_stats(audio_outputs_all, audio_targets_all)
    if args.audio_dataset == "audioset":
        mAP = np.mean([stat["AP"] for stat in stats])
    elif args.audio_dataset == "speechcommands":
        audio_acc = stats[0]["acc"]
        mAUC = np.mean([stat["auc"] for stat in stats])

    # gather the stats from all processes
    audio_metric_logger.synchronize_between_processes()

    if args.audio_dataset == "audioset":
        print_log(
            "* audio_mAP: {mAP:.6f} audio_clip_acc: {audio_clip_acc:.4f} ".format(
                mAP=mAP,
                audio_clip_acc=audio_clip_acc.item(),
            ),
            logger=logger,
        )
        dict_output = {}
        dict_output["mAP"] = mAP
    elif args.audio_dataset == "speechcommands":
        print_log(
            "* audio_acc: {audio_acc:.6f} audio_mAUC: {mAUC:.6f} audio_clip_acc: {audio_clip_acc:.4f} ".format(
                audio_acc=audio_acc,
                mAUC=mAUC,
                audio_clip_acc=audio_clip_acc.item(),
            ),
            logger=logger,
        )
        dict_output = {}
        dict_output["audio_acc"] = audio_acc
        dict_output["mAUC"] = mAUC

    return dict_output


@torch.no_grad()
def evaluate_point(
    pc_data_loader,
    model,
    device,
    args,
):
    logger = get_logger(args.log_name)
    pc_metric_logger = misc.MetricLogger(delimiter="  ", logger=logger)
    pc_header = "Point Cloud Task Test:"

    # switch to evaluation mode
    model.eval()

    npoints = args.pc_n_points

    pc_test_pred = []
    pc_test_label = []

    pc_clip_pred = []

    rank = get_rank()
    item_size = args.text_embed_dim * 2

    point_manager = TxtManager(args.point_text_feature_path, item_size, rank)

    point_labels_features = []
    for label in range(args.pc_nb_classes):
        label = str(label)
        bstr = point_manager.read(label)
        text_feature = np.frombuffer(bstr[:item_size], dtype=np.float16).copy()
        text_feature = torch.from_numpy(text_feature).to(
            device=device, dtype=torch.float16
        )
        point_labels_features.append(text_feature)
    point_labels_features = torch.stack(point_labels_features)

    for points, pc_targets in pc_metric_logger.log_every(pc_data_loader, 20, pc_header):
        points = points.to(device, non_blocking=True)
        pc_targets = pc_targets.to(device, non_blocking=True)

        points = misc.fps(points, npoints)

        with torch.cuda.amp.autocast():
            output = model(
                [{"point": points}],
                ["point"],
                ["point"],
                use_text=True,
            )

        pc_targets = pc_targets.view(-1)
        pc_pred = output["logits"]["point"].argmax(-1).view(-1)
        pc_test_pred.append(pc_pred.detach())
        pc_test_label.append(pc_targets.detach())

        point_features = output["features"]["point"]["point"]
        # point_features /= point_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * point_features @ point_labels_features.T).softmax(dim=-1)
        pc_clip_pred.append(similarity.argmax(dim=1).detach())
        # clip_accuracy = (similarity.argmax(dim=1) == pc_targets).float().mean()
        # pc_metric_logger.update(pc_clip_accuracy=clip_accuracy)

    # point cloud
    pc_test_pred = torch.cat(pc_test_pred, dim=0)
    pc_test_label = torch.cat(pc_test_label, dim=0)

    pc_clip_pred = torch.cat(pc_clip_pred, dim=0)

    if args.dist_eval:
        pc_test_pred = concat_all_gather(pc_test_pred)
        pc_test_label = concat_all_gather(pc_test_label)
        pc_clip_pred = concat_all_gather(pc_clip_pred)

    pc_acc = (
        (pc_test_pred == pc_test_label).sum() / float(pc_test_label.size(0)) * 100.0
    ).item()

    pc_clip_acc = (
        (pc_clip_pred == pc_test_label).sum() / float(pc_test_label.size(0)) * 100.0
    ).item()

    # gather the stats from all processes
    pc_metric_logger.synchronize_between_processes()

    print_log(
        "* pc_acc: {pc_acc:.4f} pc_clip_acc: {pc_clip_acc:.4f}".format(
            pc_acc=pc_acc,
            pc_clip_acc=pc_clip_acc,
        ),
        logger=logger,
    )
    dict_output = {}
    dict_output["pc_acc"] = pc_acc

    return dict_output


@torch.no_grad()
def evaluate_point_zeroshot(
    pc_data_loader,
    dataset,
    model,
    device,
    args,
):
    logger = get_logger(args.log_name)
    pc_metric_logger = misc.MetricLogger(delimiter="  ", logger=logger)
    pc_header = f"{dataset} Point Cloud Task Zero Shot Test:"

    # switch to evaluation mode
    model.eval()

    pc_test_label = []

    pc_clip_pred = []

    rank = get_rank()
    item_size = args.text_embed_dim * 2
    if dataset == "modelnet40":
        if args.text_embed_dim == 1536:
            point_manager = TxtManager(
                "text_features/Point_cloud/modelnet40_openai", item_size, rank
            )
        else:
            point_manager = TxtManager(
                "text_features/Point_cloud/modelnet40_openclip", item_size, rank
            )

        point_labels_features = []
        for label in range(40):
            label = str(label)
            bstr = point_manager.read(label)
            text_feature = np.frombuffer(bstr[:item_size], dtype=np.float16).copy()
            text_feature = torch.from_numpy(text_feature).to(
                device=device, dtype=torch.float16
            )
            point_labels_features.append(text_feature)
        point_labels_features = torch.stack(point_labels_features)
    else:
        if args.text_embed_dim == 1536:
            point_manager = TxtManager(
                "text_features/Point_cloud/shapenet55_openai", item_size, rank
            )
        else:
            point_manager = TxtManager(
                "text_features/Point_cloud/shapenet55_openclip", item_size, rank
            )
        point_labels_features = []
        for label in range(55):
            label = str(label)
            bstr = point_manager.read(label)
            text_feature = np.frombuffer(bstr[:item_size], dtype=np.float16).copy()
            text_feature = torch.from_numpy(text_feature).to(
                device=device, dtype=torch.float16
            )
            point_labels_features.append(text_feature)
        point_labels_features = torch.stack(point_labels_features)

    for points, pc_targets in pc_metric_logger.log_every(pc_data_loader, 20, pc_header):
        points = points.to(device, non_blocking=True)
        pc_targets = pc_targets.to(device, non_blocking=True)

        # points = misc.fps(points, npoints)

        with torch.cuda.amp.autocast():
            output = model(
                [{"point": points}],
                ["point"],
                ["point"],
                use_text=True,
            )

        pc_targets = pc_targets.view(-1)
        # pc_pred = output["logits"]["point"].argmax(-1).view(-1)
        # pc_test_pred.append(pc_pred.detach())
        pc_test_label.append(pc_targets.detach())

        point_features = output["features"]["point"]["point"]
        # point_features /= point_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * point_features @ point_labels_features.T).softmax(dim=-1)
        pc_clip_pred.append(similarity.argmax(dim=1).detach())
        # clip_accuracy = (similarity.argmax(dim=1) == pc_targets).float().mean()
        # pc_metric_logger.update(pc_clip_accuracy=clip_accuracy)

    # point cloud
    # pc_test_pred = torch.cat(pc_test_pred, dim=0)
    pc_test_label = torch.cat(pc_test_label, dim=0)

    pc_clip_pred = torch.cat(pc_clip_pred, dim=0)

    if args.dist_eval:
        # pc_test_pred = concat_all_gather(pc_test_pred)
        pc_test_label = concat_all_gather(pc_test_label)
        pc_clip_pred = concat_all_gather(pc_clip_pred)

    # pc_acc = (
    #     (pc_test_pred == pc_test_label).sum() / float(pc_test_label.size(0)) * 100.0
    # ).item()

    pc_clip_acc = (
        (pc_clip_pred == pc_test_label).sum() / float(pc_test_label.size(0)) * 100.0
    ).item()

    # gather the stats from all processes
    pc_metric_logger.synchronize_between_processes()

    print_log(
        "* pc_clip_acc: {pc_clip_acc:.4f}".format(
            pc_clip_acc=pc_clip_acc,
        ),
        logger=logger,
    )
    dict_output = {}
    dict_output["pc_acc"] = pc_clip_acc

    return dict_output


@torch.no_grad()
def evaluate_video(
    video_data_loader,
    model,
    device,
    args,
):
    criterion = torch.nn.CrossEntropyLoss()
    logger = get_logger(args.log_name)
    video_metric_logger = misc.MetricLogger(delimiter="  ", logger=logger)

    video_header = "Video Task Test:"

    # switch to evaluation mode
    model.eval()
    rank = get_rank()
    item_size = args.text_embed_dim * 2

    video_manager = TxtManager(args.video_text_feature_path, item_size, rank)

    video_labels_features = []
    for label in range(args.video_nb_classes):
        label = str(label)
        bstr = video_manager.read(label)
        text_feature = np.frombuffer(bstr[:item_size], dtype=np.float16).copy()
        text_feature = torch.from_numpy(text_feature).to(
            device=device, dtype=torch.float16
        )
        video_labels_features.append(text_feature)
    video_labels_features = torch.stack(video_labels_features)

    video_clip_pred = []
    video_clip_labels = []

    for batch in video_metric_logger.log_every(video_data_loader, 100, video_header):
        video_samples = batch[0]
        video_targets = batch[1]
        video_samples = video_samples.to(device, non_blocking=True)
        video_targets = video_targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            output = model(
                [{"video": video_samples}],
                ["video"],
                ["video"],
            )

            # loss = criterion(output["logits"]["video"], video_targets)
        # acc1, acc5 = accuracy(output["logits"]["video"], video_targets, topk=(1, 5))
        # batch_size = video_samples.shape[0]
        # video_metric_logger.update(loss=loss.item())
        # video_metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        # video_metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)

        video_features = output["features"]["video"]["video"]
        # video_features /= video_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * video_features @ video_labels_features.T).softmax(dim=-1)

        clip_pred = similarity.argmax(dim=1)
        clip_labels = video_targets

        if args.dist_eval:
            clip_pred = concat_all_gather(clip_pred)
            clip_labels = concat_all_gather(clip_labels)

        video_clip_pred.append(clip_pred)
        video_clip_labels.append(clip_labels)

    video_clip_pred = torch.cat(video_clip_pred).cpu()
    video_clip_labels = torch.cat(video_clip_labels).cpu()

    video_clip_acc = (video_clip_pred == video_clip_labels).float().mean()

    # gather the stats from all processes
    video_metric_logger.synchronize_between_processes()

    print_log(
        "* video_Acc@1 {top1.global_avg:.3f} video_Acc@5 {top5.global_avg:.3f} video_loss {losses.global_avg:.3f} video_clip_acc {video_clip_acc:.4f}".format(
            top1=video_metric_logger.acc1,
            top5=video_metric_logger.acc5,
            losses=video_metric_logger.loss,
            video_clip_acc=video_clip_acc.item(),
        ),
        logger=logger,
    )
    dict_output = {
        k: meter.global_avg for k, meter in video_metric_logger.meters.items()
    }

    return dict_output


@torch.no_grad()
def evaluate_rgbd(
    rgbd_data_loader,
    model,
    device,
    dataset,
    args,
):
    criterion = torch.nn.CrossEntropyLoss()
    logger = get_logger(args.log_name)
    rgbd_metric_logger = misc.MetricLogger(delimiter="  ", logger=logger)

    rgbd_header = f"{dataset} Task Test:"

    # switch to evaluation mode
    model.eval()
    # rank = get_rank()
    # item_size = args.text_embed_dim * 2
    # image_manager = TxtManager(args.img_text_feature_path, item_size, rank)
    # audio_manager = TxtManager(args.audio_text_feature_path, item_size, rank)
    # point_manager = TxtManager(args.point_text_feature_path, item_size, rank)
    # rgbd_manager = TxtManager(args.rgbd_val_text_feature_path, item_size, rank)

    rgbd_labels_features = []
    # for label in range(args.rgbd_val_nb_classes):
    #     label = str(label)
    #     bstr = rgbd_manager.read(label)
    #     text_feature = np.frombuffer(bstr[:item_size], dtype=np.float16).copy()
    #     text_feature = torch.from_numpy(text_feature).to(
    #         device=device, dtype=torch.float16
    #     )
    #     rgbd_labels_features.append(text_feature)
    if dataset == "sun-rgbd":
        rgbd_val_text_features = torch.load(args.rgbd_sunrgbd_val_text_feature_path)
    elif dataset == "nyu-depth-v2-val1":
        rgbd_val_text_features = torch.load(args.rgbd_nyu_val1_text_feature_path)
    elif dataset == "nyu-depth-v2-val2":
        rgbd_val_text_features = torch.load(args.rgbd_nyu_val2_text_feature_path)

    for label in rgbd_data_loader.dataset.idx2label:
        text_feature = rgbd_val_text_features[label].to(
            device=device, dtype=torch.float16
        )
        rgbd_labels_features.append(text_feature)
    rgbd_labels_features = torch.stack(rgbd_labels_features)

    test_dataset = rgbd_data_loader.dataset

    for batch in rgbd_metric_logger.log_every(rgbd_data_loader, 50, rgbd_header):
        rgbd_samples, rgbd_targets = batch
        if args.use_depth_only:
            rgbd_samples = rgbd_samples[:, 3:, ...]

        rgbd_samples = rgbd_samples.to(device, non_blocking=True)
        rgbd_targets = rgbd_targets.to(device, non_blocking=True)

        # rgbd_text_features = []
        # for label in rgbd_targets:
        #     text_feature=rgbd_labels_features[label]
        #     rgbd_text_features.append(text_feature)
        # rgbd_text_features = torch.stack(rgbd_text_features)

        with torch.cuda.amp.autocast():
            output = model(
                [{"rgbd": rgbd_samples}],
                ["rgbd"],
                ["rgbd"],
                use_text=True,
            )

            rgbd_features = output["features"]["rgbd"]["rgbd"]
            # rgbd_features /= rgbd_features.norm(dim=-1, keepdim=True)
            logits_per_depth = rgbd_features @ rgbd_labels_features.T

            loss = criterion(logits_per_depth, rgbd_targets)

        if hasattr(test_dataset, "other_idx"):
            merge_idx = test_dataset.other_idx
            mapping_indices = test_dataset.map_to_others_idx
            (acc1, acc5), correct = cond_acc(
                logits_per_depth, rgbd_targets, mapping_indices, merge_idx, topk=(1, 5)
            )
        else:
            acc1, acc5 = accuracy(logits_per_depth, rgbd_targets, topk=(1, 5))
        acc1, acc5 = scaled_all_reduce([acc1, acc5])
        batch_size = rgbd_samples.shape[0]
        rgbd_metric_logger.update(loss=loss.item())
        rgbd_metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        rgbd_metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)

        # similarity_labels = (100.0 * rgbd_features @ rgbd_labels_features.T).softmax(dim=-1)
        # rgbd_clip_pred.append(similarity_labels.argmax(dim=1))
        # rgbd_clip_labels.append(rgbd_targets)

    # rgbd_clip_pred = torch.cat(rgbd_clip_pred).cpu()
    # rgbd_clip_labels = torch.cat(rgbd_clip_labels).cpu()
    # rgbd_clip_acc = (rgbd_clip_pred == rgbd_clip_labels).float().mean()

    # gather the stats from all processes
    rgbd_metric_logger.synchronize_between_processes()

    print_log(
        "* {dataset} rgbd_Acc@1 {top1.global_avg:.3f} rgbd_Acc@5 {top5.global_avg:.3f} ".format(
            dataset=dataset,
            top1=rgbd_metric_logger.acc1,
            top5=rgbd_metric_logger.acc5,
        ),
        logger=logger,
    )
    dict_output = {
        k: meter.global_avg for k, meter in rgbd_metric_logger.meters.items()
    }

    return dict_output


def test_zeroshot_3d_core(
    testloaders, model, open_clip_text_model, tokenizer, args=None
):
    metrics = dict()
    if isinstance(testloaders, dict):
        for dname, dloader in testloaders.items():
            if dname == "scanobjectnn":
                m = test_scanobjectnn(
                    dloader, model, open_clip_text_model, tokenizer, dname, args
                )
                metrics.update({dname: m})
            else:
                m = test_zeroshot_3d_single(
                    dloader, model, open_clip_text_model, tokenizer, dname, args
                )
                metrics.update({dname: m})
    else:
        dname = args.point_val_data
        if dname == "scanobjectnn":
            m = test_scanobjectnn(
                testloaders, model, open_clip_text_model, tokenizer, dname, args
            )
            metrics.update({dname: m})
        else:
            m = test_zeroshot_3d_single(
                testloaders, model, open_clip_text_model, tokenizer, dname, args
            )
            metrics.update({dname: m})
    return metrics


def test_zeroshot_3d_single(
    test_loader, model, open_clip_text_model, tokenizer, dataset_name, args=None
):
    logger = get_logger(args.log_name)
    pc_metric_logger = misc.MetricLogger(delimiter="  ", logger=logger)
    pc_header = f"{dataset_name} Point Cloud Task Zero Shot Test:"

    # switch to evaluate mode
    # model = get_model(model)
    model.eval()

    with torch.no_grad():
        if args.text_embed_dim == 1536:
            rank = get_rank()
            item_size = args.text_embed_dim * 2
            if dataset_name == "modelnet40":
                if args.text_embed_dim == 1536:
                    point_manager = TxtManager(
                        "/home/zhoubo/farm/M2PT/Fuse/text_features/Point_cloud/modelnet40_openai",
                        item_size,
                        rank,
                    )
                else:
                    point_manager = TxtManager(
                        "/home/zhoubo/farm/M2PT/Fuse/text_features/Point_cloud/modelnet40_openclip",
                        item_size,
                        rank,
                    )

                text_features = []
                for label in range(40):
                    label = str(label)
                    bstr = point_manager.read(label)
                    text_feature = np.frombuffer(
                        bstr[:item_size], dtype=np.float16
                    ).copy()
                    text_feature = torch.from_numpy(text_feature).to(
                        device=args.device, non_blocking=True
                    )
                    text_features.append(text_feature)
                text_features = torch.stack(text_features)
            else:
                if args.text_embed_dim == 1536:
                    point_manager = TxtManager(
                        "text_features/Point_cloud/shapenet55_openai", item_size, rank
                    )
                else:
                    point_manager = TxtManager(
                        "text_features/Point_cloud/shapenet55_openclip", item_size, rank
                    )
                text_features = []
                for label in range(55):
                    label = str(label)
                    bstr = point_manager.read(label)
                    text_feature = np.frombuffer(
                        bstr[:item_size], dtype=np.float16
                    ).copy()
                    text_feature = torch.from_numpy(text_feature).to(
                        device=args.device, non_blocking=True
                    )
                    text_features.append(text_feature)
                text_features = torch.stack(text_features)
        else:
            print_log("=> encoding captions", logger=logger)
            if dataset_name == "modelnet40":
                with open(f"{PC_META_DATA_DIR}/templates.json") as f:
                    templates = json.load(f)[args.point_val_data_prompt]
            else:
                templates = ["{}.", "a {}.", "a phote of {}"]

            with open(f"{PC_META_DATA_DIR}/labels.json") as f:
                labels = json.load(f)[dataset_name]
            text_features = []
            for label in labels:
                texts = [t.format(label) for t in templates]
                texts = tokenizer(texts).cuda(args.device, non_blocking=True)
                if len(texts.shape) < 2:
                    texts = texts[None, ...]

                if args.distributed:
                    class_embeddings = open_clip_text_model.module.encode_text(texts)
                else:
                    class_embeddings = open_clip_text_model.encode_text(texts)
                class_embeddings = class_embeddings / class_embeddings.norm(
                    dim=-1, keepdim=True
                )
                class_embeddings = class_embeddings.mean(dim=0)
                class_embeddings = class_embeddings / class_embeddings.norm(
                    dim=-1, keepdim=True
                )
                text_features.append(class_embeddings)
            text_features = torch.stack(text_features, dim=0)

        # per_class_stats = collections.defaultdict(int)
        # per_class_correct_top1 = collections.defaultdict(int)
        # per_class_correct_top5 = collections.defaultdict(int)

        for batch in pc_metric_logger.log_every(test_loader, 20, pc_header):
            pc, target = batch["pc"], batch["label"]
            # target_name = batch["class_name"]
            # for name in target_name:
            #     per_class_stats[name] += 1

            pc = pc.to(args.device, non_blocking=True)
            if isinstance(target, list):
                target = torch.LongTensor(target)
            target = target.to(args.device, non_blocking=True)

            with torch.cuda.amp.autocast():
                output = model(
                    [{"point": pc}],
                    ["point"],
                    ["point"],
                )
                pc_features = output["features"]["point"]["point"]

                # cosine similarity as logits
                logits_per_pc = pc_features @ text_features.t()

            # measure accuracy and record loss
            (acc1, acc5), correct = acc(logits_per_pc, target, topk=(1, 5))
            # TODO: fix the all reduce for the correct variable, assuming only one process for evaluation!
            acc1, acc5 = scaled_all_reduce([acc1, acc5])
            pc_metric_logger.meters["acc1"].update(acc1.item(), pc.size(0))
            pc_metric_logger.meters["acc5"].update(acc5.item(), pc.size(0))

            # top1_accurate = correct[:1].squeeze()
            # top5_accurate = correct[:5].float().sum(0, keepdim=True).squeeze()
            # for idx, name in enumerate(target_name):
            #     if top1_accurate[idx].item():
            #         per_class_correct_top1[name] += 1
            #     if top5_accurate[idx].item():
            #         per_class_correct_top5[name] += 1

        # top1_accuracy_per_class = {}
        # top5_accuracy_per_class = {}
        # for name in per_class_stats.keys():
        #     top1_accuracy_per_class[name] = (
        #         per_class_correct_top1[name] / per_class_stats[name]
        #     )
        #     top5_accuracy_per_class[name] = (
        #         per_class_correct_top5[name] / per_class_stats[name]
        #     )

        # top1_accuracy_per_class = collections.OrderedDict(top1_accuracy_per_class)
        # top5_accuracy_per_class = collections.OrderedDict(top5_accuracy_per_class)
        # print_log(",".join(top1_accuracy_per_class.keys()), logger=logger)
        # print_log(
        #     ",".join([str(value) for value in top1_accuracy_per_class.values()]),
        #     logger=logger,
        # )
        # print_log(
        #     ",".join([str(value) for value in top5_accuracy_per_class.values()]),
        #     logger=logger,
        # )

    pc_metric_logger.synchronize_between_processes()

    print_log(
        f"Point Cloud 0-shot * Acc@1 {pc_metric_logger.acc1.global_avg:.3f} Acc@5 {pc_metric_logger.acc5.global_avg:.3f}",
        logger=logger,
    )
    return {
        "acc1": pc_metric_logger.acc1.global_avg,
        "acc5": pc_metric_logger.acc5.global_avg,
    }


def test_scanobjectnn(
    test_loader, model, open_clip_text_model, tokenizer, dataset_name, args=None
):
    logger = get_logger(args.log_name)
    pc_metric_logger = misc.MetricLogger(delimiter="  ", logger=logger)
    pc_header = f"{dataset_name} Point Cloud Task Zero Shot Test:"

    model.eval()

    # clip_text_feat = torch.from_numpy(test_loader.dataset.clip_cat_feat).to(args.device)

    # clip_text_feat = model.module.extract_text_embedding(clip_text_feat, 'point')
    templates = ["{}.", "a {}.", "a phote of {}"]
    with open(f"{PC_META_DATA_DIR}/labels.json") as f:
        labels = json.load(f)["scanobjectnn"]
    text_features = []
    for label in labels:
        texts = [t.format(label) for t in templates]
        texts = tokenizer(texts).cuda(args.device, non_blocking=True)
        if len(texts.shape) < 2:
            texts = texts[None, ...]

        if args.distributed:
            class_embeddings = open_clip_text_model.module.encode_text(texts)
        else:
            class_embeddings = open_clip_text_model.encode_text(texts)
        class_embeddings = class_embeddings / class_embeddings.norm(
            dim=-1, keepdim=True
        )
        class_embeddings = class_embeddings.mean(dim=0)
        class_embeddings = class_embeddings / class_embeddings.norm(
            dim=-1, keepdim=True
        )
        text_features.append(class_embeddings)
    text_features = torch.stack(text_features, dim=0)

    per_cat_correct = torch.zeros(15).to(args.device)
    per_cat_count = torch.zeros(15).to(args.device)
    # category2idx = test_loader.dataset.category2idx
    # idx2category = {v: k for k, v in category2idx.items()}

    # logits_all = []
    # labels_all = []
    with torch.no_grad():
        for batch in pc_metric_logger.log_every(test_loader, 20, pc_header):
            pc, target = batch["pc"], batch["label"]
            pc = pc.to(args.device, non_blocking=True)
            if isinstance(target, list):
                target = torch.LongTensor(target)
            target = target.to(args.device, non_blocking=True)

            with torch.cuda.amp.autocast():
                output = model(
                    [{"point": pc}],
                    ["point"],
                    ["point"],
                )
                pc_features = output["features"]["point"]["point"]

                logits = pc_features @ text_features.T

            # logits_all.append(target.detach())
            # labels_all.append(target)
            # calculate per class accuracy
            for i in range(15):
                idx = target == i
                if idx.sum() > 0:
                    per_cat_correct[i] += (
                        (logits[idx].argmax(dim=1) == target[idx]).float().sum()
                    )
                    per_cat_count[i] += idx.sum()

            (acc1, acc5), correct = acc(logits, target, topk=(1, 5))
            acc1, acc5 = scaled_all_reduce([acc1, acc5])
            pc_metric_logger.meters["acc1"].update(acc1.item(), pc.size(0))
            pc_metric_logger.meters["acc5"].update(acc5.item(), pc.size(0))
    # topk_acc, correct = acc(torch.cat(logits_all), torch.cat(labels_all), topk=(1,3,5,))

    overall_acc = per_cat_correct.sum() / per_cat_count.sum() * 100
    per_cat_acc = per_cat_correct / per_cat_count * 100

    print_log(
        "Test ScanObjectNN: overall acc: {0} class_acc: {1}".format(
            overall_acc, per_cat_acc.mean()
        ),
        logger=logger,
    )
    print_log(
        f"Point Cloud 0-shot * Acc@1 {pc_metric_logger.acc1.global_avg:.3f} Acc@5 {pc_metric_logger.acc5.global_avg:.3f}",
        logger=logger,
    )
    return {
        "acc1": pc_metric_logger.acc1.global_avg,
        "acc5": pc_metric_logger.acc5.global_avg,
    }

    # wandb.log({"test_scanobjectnn/epoch": self.epoch,
    #            "test_scanobjectnn/step": self.step,
    #            "test_scanobjectnn/overall_acc": overall_acc,
    #            "test_scanobjectnn/class_acc": per_cat_acc.mean(),
    #            "test_scanobjectnn/top3_acc": topk_acc[1],
    #            "test_scanobjectnn/top5_acc": topk_acc[2],})


def test_rgbd_cls_single(
    test_loader, model, open_clip_text_model, tokenizer, dataset_name, args=None
):
    logger = get_logger(args.log_name)
    rgbd_metric_logger = misc.MetricLogger(delimiter="  ", logger=logger)
    rgbd_header = f"{dataset_name}@ RGB-Depth Zero-shot Classification:"

    # switch to evaluate mode
    # model = get_model(model)
    model.eval()

    print_log("=> encoding captions w/ templates", logger=logger)
    templates = SCENE_CLS_TEMPLATE
    test_dataset = test_loader.dataset
    labels = test_dataset.idx2label

    with torch.no_grad():
        text_features = []
        if args.text_embed_dim == 1536:
            if dataset_name == "sun-rgbd":
                rgbd_val_text_features = torch.load(
                    args.rgbd_sunrgbd_val_text_feature_path
                )
            elif dataset_name == "nyu-depth-v2-val1":
                rgbd_val_text_features = torch.load(
                    args.rgbd_nyu_val1_text_feature_path
                )
            elif dataset_name == "nyu-depth-v2-val2":
                rgbd_val_text_features = torch.load(
                    args.rgbd_nyu_val2_text_feature_path
                )

            for label in labels:
                text_feature = rgbd_val_text_features[label].to(
                    args.device, non_blocking=True
                )
                text_features.append(text_feature)
            text_features = torch.stack(text_features)
        else:
            for label in labels:
                texts = [t(label) for t in templates]
                texts = tokenizer(texts).cuda(args.device, non_blocking=True)
                if len(texts.shape) < 2:
                    texts = texts[None, ...]

                if args.distributed:
                    class_embeddings = open_clip_text_model.module.encode_text(texts)
                else:
                    class_embeddings = open_clip_text_model.encode_text(texts)
                class_embeddings = class_embeddings / class_embeddings.norm(
                    dim=-1, keepdim=True
                )
                class_embeddings = class_embeddings.mean(dim=0)
                class_embeddings = class_embeddings / class_embeddings.norm(
                    dim=-1, keepdim=True
                )
                text_features.append(class_embeddings)
            text_features = torch.stack(text_features, dim=0)

        for batch in rgbd_metric_logger.log_every(test_loader, 50, rgbd_header):
            depth, target = (
                batch["depth"],
                batch["label"],
            )

            depth = depth.to(args.device, non_blocking=True)
            if isinstance(target, list):
                target = torch.LongTensor(target)
            target = target.to(args.device, non_blocking=True)

            # encode visual
            with torch.cuda.amp.autocast():
                output = model(
                    [{"rgbd": depth}],
                    ["rgbd"],
                    ["rgbd"],
                )

                depth_features = output["features"]["rgbd"]["rgbd"]
                # cosine similarity as logits
                logits_per_depth = depth_features @ text_features.t()

            # measure accuracy and record loss
            if hasattr(test_dataset, "other_idx"):
                merge_idx = test_dataset.other_idx
                mapping_indices = test_dataset.map_to_others_idx
                (acc1, acc5), correct = cond_acc(
                    logits_per_depth, target, mapping_indices, merge_idx, topk=(1, 5)
                )
            else:
                (acc1, acc5), correct = acc(logits_per_depth, target, topk=(1, 5))

            # TODO: fix the all reduce for the correct variable, assuming only one process for evaluation!
            acc1, acc5 = scaled_all_reduce([acc1, acc5])
            rgbd_metric_logger.meters["acc1"].update(acc1.item(), depth.size(0))
            rgbd_metric_logger.meters["acc5"].update(acc5.item(), depth.size(0))

            # top1_accurate = correct[:1].squeeze()
            # top5_accurate = correct[:5].float().sum(0, keepdim=True).squeeze()

    rgbd_metric_logger.synchronize_between_processes()

    if args.distributed:
        torch.distributed.barrier()

    print_log(
        f"[{dataset_name}] : 0-shot * Acc@1 {rgbd_metric_logger.acc1.global_avg:.3f} Acc@5 {rgbd_metric_logger.acc5.global_avg:.3f}",
        logger=logger,
    )
    return {
        "acc1": rgbd_metric_logger.acc1.global_avg,
        "acc5": rgbd_metric_logger.acc5.global_avg,
    }


def test_rgbd_cls_core(testloaders, model, open_clip_text_model, tokenizer, args=None):
    metrics = dict()
    if isinstance(testloaders, dict):
        for dname, dloader in testloaders.items():
            m = test_rgbd_cls_single(
                dloader, model, open_clip_text_model, tokenizer, dname, args
            )
            metrics.update({dname: m})
    else:
        m = test_rgbd_cls_single(
            testloaders,
            model,
            open_clip_text_model,
            tokenizer,
            "Eval RGBD-CLS Dataset",
            args,
        )
        metrics.update({"Single Eval": m})
    return metrics


def test_audio_single_map(
    testloader,
    model,
    open_clip_text_model,
    tokenizer,
    dataset_name="Eval Audio mAP",
    args=None,
):
    logger = get_logger(args.log_name)
    audio_metric_logger = misc.MetricLogger(delimiter="  ", logger=logger)
    audio_header = f"{dataset_name}@ Audio mAP Task Test:"

    model.eval()

    metric = MAP()
    metric.initialize()

    # zs_metric = MAP()
    # zs_metric.initialize()

    if args.text_embed_dim == 1536:
        rank = get_rank()
        item_size = args.text_embed_dim * 2

        audio_manager = TxtManager(args.audio_text_feature_path, item_size, rank)

    with torch.no_grad():
        text_features = []
        if args.text_embed_dim == 1536:
            for label in range(args.audio_nb_classes):
                label = str(label)
                bstr = audio_manager.read(label)
                text_feature = np.frombuffer(bstr[:item_size], dtype=np.float16).copy()
                text_feature = torch.from_numpy(text_feature).to(
                    args.device, non_blocking=True
                )
                text_features.append(text_feature)
            text_features = torch.stack(text_features)
        else:
            labels = testloader.dataset.idx2label
            for label in labels:
                texts = [t(label) for t in SOUND_AS_IMAGE_TEMPLATE]
                texts = tokenizer(texts).cuda(args.device, non_blocking=True)
                if len(texts.shape) < 2:
                    texts = texts[None, ...]

                if args.distributed:
                    class_embeddings = open_clip_text_model.module.encode_text(texts)
                else:
                    class_embeddings = open_clip_text_model.encode_text(texts)
                class_embeddings = class_embeddings / class_embeddings.norm(
                    dim=-1, keepdim=True
                )
                class_embeddings = class_embeddings.mean(dim=0)
                class_embeddings = class_embeddings / class_embeddings.norm(
                    dim=-1, keepdim=True
                )
                text_features.append(class_embeddings)
            text_features = torch.stack(text_features, dim=0)

        # audio forward
        for batch in audio_metric_logger.log_every(testloader, 100, audio_header):
            ids, audio, targets = batch["id"], batch["audio"], batch["target"]

            audio = audio.to(args.device, non_blocking=True)

            targets = targets.to(args.device, non_blocking=True)
            ids = torch.tensor(ids).to(args.device)

            with torch.cuda.amp.autocast():
                output = model(
                    [{"audio": audio}],
                    ["audio"],
                    ["audio"],
                )

                audio_features = output["features"]["audio"]["audio"]
                # audio_logits = output["logits"]["audio"]

                # cosine similarity as logits
                logits_per_audio = audio_features @ text_features.t()
                metric.compute(ids, logits_per_audio, targets)
                # zs_metric.compute(ids, logits_per_audio, targets)

        if args.distributed:
            torch.distributed.barrier()

        stats = metric.merge_results()
        # zs_stats = zs_metric.merge_results()

    # hack, `acc1` field for saving best checkpoint

    results = {
        "mAP": stats["map"],
        "acc1": stats["map"],
        # "zs_mAP": zs_stats["map"],
    }

    stats["acc1"] = stats["map"]

    print_log(
        f'[{dataset_name}] : mAP {stats["map"]} ',
        logger=logger,
    )

    # stats.update({"zs_map": zs_stats["map"]})
    return results


def test_audio_single_cls(
    testloader,
    model,
    open_clip_text_model,
    tokenizer,
    dataset_name="Eval Audio Cls",
    args=None,
):
    logger = get_logger(args.log_name)
    audio_metric_logger = misc.MetricLogger(delimiter="  ", logger=logger)
    audio_header = f"{dataset_name}@ Audio CLS Task Test:"

    model.eval()

    metric = Accuracy()
    metric.initialize()

    print_log("=> encoding captions w/ templates", logger=logger)
    test_dataset = testloader.dataset
    labels = test_dataset.idx2label

    if args.text_embed_dim == 1536:
        rank = get_rank()
        item_size = args.text_embed_dim * 2
        if dataset_name == "audiocaps":
            audio_manager = TxtManager(
                "/home/zhoubo/farm/open_clip/src/open_clip_train/text_features/Audio/audiocaps_test_openai",
                item_size,
                rank,
            )
        else:
            audio_manager = TxtManager(args.audio_text_feature_path, item_size, rank)

    with torch.no_grad():
        text_features = []
        if args.text_embed_dim == 1536:
            for label in range(args.audio_nb_classes):
                label = str(label)
                bstr = audio_manager.read(label)
                text_feature = np.frombuffer(bstr[:item_size], dtype=np.float16).copy()
                text_feature = torch.from_numpy(text_feature).to(
                    args.device, non_blocking=True
                )
                text_features.append(text_feature)
            text_features = torch.stack(text_features)
        else:
            for label in labels:
                texts = [t(label) for t in SOUND_AS_IMAGE_TEMPLATE]
                texts = tokenizer(texts).cuda(args.device, non_blocking=True)
                if len(texts.shape) < 2:
                    texts = texts[None, ...]

                if args.distributed:
                    class_embeddings = open_clip_text_model.module.encode_text(texts)
                else:
                    class_embeddings = open_clip_text_model.encode_text(texts)
                class_embeddings = class_embeddings / class_embeddings.norm(
                    dim=-1, keepdim=True
                )
                class_embeddings = class_embeddings.mean(dim=0)
                class_embeddings = class_embeddings / class_embeddings.norm(
                    dim=-1, keepdim=True
                )
                text_features.append(class_embeddings)
            text_features = torch.stack(text_features, dim=0)

        # audio forward
        for batch in audio_metric_logger.log_every(testloader, 20, audio_header):
            ids, audio, targets = batch["id"], batch["audio"], batch["label"]

            audio = audio.to(args.device, non_blocking=True)
            if isinstance(targets, list):
                targets = torch.LongTensor(targets)
            targets = targets.to(args.device, non_blocking=True)
            ids = torch.tensor(ids).to(args.device, non_blocking=True)

            # encode visual
            afeat = None
            audio_dim = audio.ndim
            if audio_dim == 4:
                # bsz x n_clip x tdim x fdim
                n_clip = audio.size(1)
                audio = einops.rearrange(audio, "b n ... -> (b n) ...")

            with torch.cuda.amp.autocast():
                output = model(
                    [{"audio": audio}],
                    ["audio"],
                    ["audio"],
                )
                afeat = output["features"]["audio"]["audio"]

                if audio_dim == 4:
                    afeat = einops.rearrange(afeat, "(b n) ... -> b n ...", n=n_clip)
                    afeat = torch.mean(afeat, dim=1)
                audio_features = afeat / afeat.norm(dim=-1, keepdim=True)
                # cosine similarity as logits
                logits_per_audio = audio_features @ text_features.t()
                metric.compute(ids, logits_per_audio, targets)

        stats = metric.merge_results()
    # hack: `acc1` field for saving best checkpoint
    stats["acc1"] = stats["accuracy"]

    print_log(
        f'[{dataset_name}] : Classification * Acc@1 {stats["accuracy"]}', logger=logger
    )
    return stats


def test_audio_single_ret(
    testloader,
    model,
    open_clip_text_model,
    tokenizer,
    dataset_name="Eval Audio Ret",
    args=None,
):
    logger = get_logger(args.log_name)
    audio_metric_logger = misc.MetricLogger(delimiter="  ", logger=logger)
    audio_header = f"{dataset_name}@ Audio retrival Task Test:"

    # model = get_model(model)
    model.eval()
    metric = Recall()

    dataset = testloader.dataset
    text_ids = dataset.text_ids
    texts = dataset.texts
    stats = {}

    if args.text_embed_dim == 1536:
        rank = get_rank()
        item_size = args.text_embed_dim * 2

        if "audiocaps" in args.audio_val_data:
            audio_manager = TxtManager(
                "/home/zhoubo/farm/open_clip/src/open_clip_train/text_features/Audio/audiocaps_test_openai",
                item_size,
                rank,
            )
        else:
            audio_manager = TxtManager(args.audio_text_feature_path, item_size, rank)

    with torch.no_grad():
        text_ids = torch.tensor(text_ids).cuda()

        if args.text_embed_dim == 1536:
            text_logits = []
            for label in text_ids:
                label = str(label.item())
                bstr = audio_manager.read(label)
                text_feature = np.frombuffer(bstr[:item_size], dtype=np.float16).copy()
                text_feature = torch.from_numpy(text_feature).to(
                    args.device, dtype=torch.float16, non_blocking=True
                )
                text_logits.append(text_feature)
            text_logits = torch.stack(text_logits)
        else:
            text_cnt = len(text_ids)

            if args.distributed:
                slice_id = dist.get_rank()
                slice_count = dist.get_world_size()
            else:
                slice_id = 0
                slice_count = 1
            batch_sampler = misc.new_islice(
                range(text_cnt), slice_id, text_cnt, slice_count
            )
            start_idx = batch_sampler[0]
            end_idx = batch_sampler[-1] + 1

            text_logits_list = []
            for i in range(start_idx, end_idx, 50):
                samples_list = []
                for text in texts[i : min(i + 50, end_idx)]:
                    # text = text --> seems no need for template for retrieval
                    samples_list.append(text)
                tokenized_captions = tokenizer(samples_list).cuda(
                    args.device, non_blocking=True
                )

                if args.distributed:
                    text_logits = open_clip_text_model.module.encode_text(
                        tokenized_captions, True
                    )
                else:
                    text_logits = open_clip_text_model.encode_text(
                        tokenized_captions, True
                    )
                text_logits_list.append(text_logits)

            text_logits = torch.cat(text_logits_list, dim=0).to(torch.float16)
            text_logits = (
                misc.all_gather(text_logits) if args.distributed else text_logits
            )

        metric.initialize(text_ids=text_ids, text_logits=text_logits)

        # forward audio
        for batch in audio_metric_logger.log_every(testloader, 50, audio_header):
            audio, audio_ids = batch["audio"], batch["uniq_id"]
            audio = audio.to(args.device, non_blocking=True)
            if isinstance(audio_ids, list):
                audio_ids = torch.tensor(audio_ids).to(args.device)

            afeat = None
            audio_dim = audio.ndim
            if audio_dim == 4:
                # bsz x n_clip x tdim x fdim
                n_clip = audio.size(1)
                audio = einops.rearrange(audio, "b n ... -> (b n) ...")

            with torch.cuda.amp.autocast():
                output = model(
                    [{"audio": audio}],
                    ["audio"],
                    ["audio"],
                )
                afeat = output["features"]["audio"]["audio"]
                if audio_dim == 4:
                    afeat = einops.rearrange(afeat, "(b n) ... -> b n ...", n=n_clip)
                    afeat = torch.mean(afeat, dim=1)
                audio_logits = afeat / afeat.norm(dim=-1, keepdim=True)
                audio_logits = audio_logits.to(torch.float16)
                metric.compute(audio_ids, audio_logits)

        stats = metric.merge_results()

    for key in list(stats.keys()):
        if key.startswith("img"):
            stats[key.replace("img", "audio")] = stats[key]
            del stats[key]
    # hack: use `acc1` field to save best checkpoint, retrieval scale up 100 for printing result
    stats["acc1"] = (
        stats["txt_r1"]
        + stats["txt_r5"]
        + stats["txt_r10"]
        + stats["audio_r1"]
        + stats["audio_r5"]
        + stats["audio_r10"]
    ) / (6 * 100.0)
    print_log(
        "**  %s ** Eval result = %s" % (dataset_name, json.dumps(stats)), logger=logger
    )

    return stats


def test_audiotasks_core(
    testloaders, model, open_clip_text_model, tokenizer, args=None
):
    metrics = dict()
    test_fn_mapping = {
        "map": test_audio_single_map,
        "acc": test_audio_single_cls,
        "recall": test_audio_single_ret,
    }

    if isinstance(testloaders, dict):
        for dname, dloader in testloaders.items():
            eval_metric_key = dloader.dataset.eval_metric.lower()
            m = test_fn_mapping[eval_metric_key](
                dloader, model, open_clip_text_model, tokenizer, dname, args
            )
            metrics.update({dname: m})
    else:
        eval_metric_key = testloaders.dataset.eval_metric.lower()
        m = test_fn_mapping[eval_metric_key](
            testloaders,
            model,
            open_clip_text_model,
            tokenizer,
            args.audio_val_data,
            args,
        )
        metrics.update({"Single Eval": m})
    return metrics


def test_vidret_single(
    testloader, model, open_clip_text_model, dataset_name="Eval set", args=None
):
    logger = get_logger(args.log_name)
    video_metric_logger = misc.MetricLogger(delimiter="  ", logger=logger)
    video_header = f"{dataset_name}@ video retrival Task Test:"

    model.eval()

    vis_feats = []
    text_feats = []
    image_ids = []

    # zs_mean_pool = args.vid_dire_mean_pool
    # n_frames = args.n_frames

    with torch.no_grad():
        for batch in video_metric_logger.log_every(testloader, 50, video_header):
            video, text, vid = batch["video"], batch["caption"], batch["vid"]

            video = video.to(args.device, non_blocking=True)

            with torch.cuda.amp.autocast():
                output = model(
                    [{"video": video}],
                    ["video"],
                    ["video"],
                )
                vfeat = output["features"]["video"]["video"]

            # if zs_mean_pool:
            #     vfeat = einops.rearrange(vfeat, "(b t) ... -> b t ...", t=n_frames)
            #     vfeat = torch.mean(vfeat, dim=1)
            # vfeat = vfeat / vfeat.norm(dim=-1, keepdim=True)
            if args.distributed:
                tfeat = open_clip_text_model.module.encode_text(text, True)
            else:
                tfeat = open_clip_text_model.encode_text(text, True)

            tfeat = tfeat.to(vfeat.dtype)
            
            vid = torch.LongTensor(vid).to(vfeat.device)

            vis_feats.append(vfeat)
            text_feats.append(tfeat)
            image_ids.append(vid)

    # ###
    visual_feats = {}
    for feats, ids in zip(vis_feats, image_ids):
        for i, _idx in enumerate(ids):
            idx = _idx.item()
            if idx not in visual_feats:
                visual_feats[idx] = feats[i]

    tiids = torch.cat(image_ids, dim=0)
    iids = []
    sorted_tensors = []
    for key in sorted(visual_feats.keys()):
        sorted_tensors.append(visual_feats[key].view(1, -1))
        iids.append(key)

    video_feats = torch.cat(sorted_tensors, dim=0)
    text_feats = torch.cat(text_feats, dim=0)
    iids = torch.LongTensor(iids).to(video_feats.device)

    if args.distributed:  # in get data, use distributed sampler
        torch.distributed.barrier()
        iids = concat_all_gather(iids)
        tiids = concat_all_gather(tiids)
        video_feats = concat_all_gather(video_feats)
        text_feats = concat_all_gather(text_feats)

    scores = video_feats @ text_feats.t()

    print("scores: {}".format(scores.size()))
    print("iids: {}".format(iids.size()))
    print("tiids: {}".format(tiids.size()))

    topk10 = scores.topk(10, dim=1)
    topk5 = scores.topk(5, dim=1)
    topk1 = scores.topk(1, dim=1)

    topk10_iids = tiids[topk10.indices]
    topk5_iids = tiids[topk5.indices]
    topk1_iids = tiids[topk1.indices]

    tr_r10 = (iids.unsqueeze(1) == topk10_iids).float().max(dim=1)[0].mean()
    tr_r5 = (iids.unsqueeze(1) == topk5_iids).float().max(dim=1)[0].mean()
    tr_r1 = (iids.unsqueeze(1) == topk1_iids).float().max(dim=1)[0].mean()

    topk10 = scores.topk(10, dim=0)
    topk5 = scores.topk(5, dim=0)
    topk1 = scores.topk(1, dim=0)
    topk10_iids = iids[topk10.indices]
    topk5_iids = iids[topk5.indices]
    topk1_iids = iids[topk1.indices]

    ir_r10 = (tiids.unsqueeze(0) == topk10_iids).float().max(dim=0)[0].mean()
    ir_r5 = (tiids.unsqueeze(0) == topk5_iids).float().max(dim=0)[0].mean()
    ir_r1 = (tiids.unsqueeze(0) == topk1_iids).float().max(dim=0)[0].mean()

    eval_result = {
        "tr_r10": tr_r10.item() * 100.0,
        "tr_r5": tr_r5.item() * 100.0,
        "tr_r1": tr_r1.item() * 100.0,
        "ir_r10": ir_r10.item() * 100.0,
        "ir_r5": ir_r5.item() * 100.0,
        "ir_r1": ir_r1.item() * 100.0,
        "acc1": 100.0 * (tr_r1 + tr_r5 + tr_r10 + ir_r1 + ir_r5 + ir_r10).item() / 6.0,
    }

    print_log(
        "**  %s ** Eval result = %s" % (dataset_name, json.dumps(eval_result)),
        logger=logger,
    )
    return eval_result


def test_vidret_core(testloaders, model, open_clip_text_model, tokenizer, args=None):
    metrics = dict()

    if isinstance(testloaders, dict):
        for dname, dloader in testloaders.items():
            m = test_vidret_single(dloader, model, open_clip_text_model, dname, args)
            metrics.update({dname: m})
    else:
        m = test_vidret_single(
            testloaders, model, open_clip_text_model, args.video_val_data, args
        )
        metrics.update({args.video_val_data: m})
    return metrics
