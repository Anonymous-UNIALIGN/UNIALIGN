# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import sys
import argparse

import json
import copy

import numpy as np
import os
import time
from pathlib import Path
import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import timm

from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

import util.lr_decay as lrd
import util.misc as misc

from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import src.models.vit_one_anchor as vit_one
from src.models.lora_module.lora import LoraConfig, LoraModel
from src.train.engine_pretrain_one_anchor import (
    train_one_epoch_concat,
    train_one_epoch_concat_use_all,
    evaluate_image,
    test_audiotasks_core,
    test_rgbd_cls_core,
    test_zeroshot_3d_core,
    test_vidret_core,
)

from util.loss import MultiModalUncertaintyWeightingStrategy

from util.logger import print_log, get_root_logger

from clip.simple_tokenizer import SimpleTokenizer

from datasets.data import get_joint_data

from datasets.imgnet_datasets import build_transform

from util.module_dfg import get_peft_cfg

from datetime import timedelta, datetime


def get_args_parser():
    parser = argparse.ArgumentParser(
        "MAE fine-tuning for image classification & Point cloud classification & audio classification ",
        add_help=False,
    )
    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus",
    )
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument(
        "--accum_iter",
        default=1,
        type=int,
        help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)",
    )

    parser.add_argument(
        "--model",
        default="vit_base_patch16",
        type=str,
        metavar="MODEL",
        help="Name of model to train",
    )

    parser.add_argument("--input_size", default=224, type=int, help="images input size")

    parser.add_argument(
        "--drop_path",
        type=float,
        default=0.1,
        metavar="PCT",
        help="Drop path rate (default: 0.1)",
    )

    parser.add_argument(
        "--clip_grad",
        type=float,
        default=None,
        metavar="NORM",
        help="Clip gradient norm (default: None, no clipping)",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.05, help="weight decay (default: 0.05)"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        metavar="LR",
        help="learning rate (absolute lr)",
    )
    parser.add_argument(
        "--blr",
        type=float,
        default=1e-3,
        metavar="LR",
        help="base learning rate: absolute_lr = base_lr * total_batch_size / 256",
    )
    parser.add_argument(
        "--base_batchsize", type=int, default=256, help="base batch size"
    )

    parser.add_argument(
        "--layer_decay",
        type=float,
        default=0.75,
        help="layer-wise lr decay from ELECTRA/BEiT",
    )

    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-6,
        metavar="LR",
        help="lower lr bound for cyclic schedulers that hit 0",
    )

    parser.add_argument(
        "--warmup_epochs", type=int, default=5, metavar="N", help="epochs to warmup LR"
    )

    parser.add_argument(
        "--color_jitter",
        type=float,
        default=None,
        metavar="PCT",
        help="Color jitter factor (enabled only when not using Auto/RandAug)",
    )
    (
        parser.add_argument(
            "--aa",
            type=str,
            default="rand-m9-mstd0.5-inc1",
            metavar="NAME",
            help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)',
        ),
    )
    parser.add_argument(
        "--smoothing", type=float, default=0.1, help="Label smoothing (default: 0.1)"
    )

    parser.add_argument(
        "--reprob",
        type=float,
        default=0.25,
        metavar="PCT",
        help="Random erase prob (default: 0.25)",
    )
    parser.add_argument(
        "--remode",
        type=str,
        default="pixel",
        help='Random erase mode (default: "pixel")',
    )
    parser.add_argument(
        "--recount", type=int, default=1, help="Random erase count (default: 1)"
    )
    parser.add_argument(
        "--resplit",
        action="store_true",
        default=False,
        help="Do not random erase first (clean) augmentation split",
    )

    parser.add_argument(
        "--mixup", type=float, default=0, help="mixup alpha, mixup enabled if > 0."
    )
    parser.add_argument(
        "--audio_mixup", type=float, default=0, help="mixup alpha for audio"
    )
    parser.add_argument(
        "--cutmix", type=float, default=0, help="cutmix alpha, cutmix enabled if > 0."
    )
    parser.add_argument(
        "--cutmix_minmax",
        type=float,
        nargs="+",
        default=None,
        help="cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)",
    )
    parser.add_argument(
        "--mixup_prob",
        type=float,
        default=1.0,
        help="Probability of performing mixup or cutmix when either/both is enabled",
    )
    parser.add_argument(
        "--mixup_switch_prob",
        type=float,
        default=0.5,
        help="Probability of switching to cutmix when both mixup and cutmix enabled",
    )
    parser.add_argument(
        "--mixup_mode",
        type=str,
        default="batch",
        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"',
    )

    parser.add_argument("--finetune", default="", help="finetune from checkpoint")
    parser.add_argument("--global_pool", action="store_true")
    parser.set_defaults(global_pool=False)
    parser.add_argument(
        "--cls_token",
        action="store_false",
        dest="global_pool",
        help="Use class token instead of global pool for classification",
    )
    parser.add_argument(
        "--audio_pretrained_path", type=str, help="pretrained weights for aux models"
    )

    parser.add_argument(
        "--point_pretrained_path", type=str, help="pretrained weights for aux models"
    )

    parser.add_argument(
        "--img_data_path",
        default="/datasets01/imagenet_full_size/061417/",
        type=str,
        help="dataset path",
    )
    parser.add_argument(
        "--image_nb_classes",
        default=1000,
        type=int,
        help="number of the classification types",
    )

    parser.add_argument(
        "--pc_train_dataset", type=str, default="modelnet40", help="point cloud dataset"
    )

    parser.add_argument(
        "--point_train_data",
        type=str,
        default=None,
        help="Path to point cloud training data",
    )
    parser.add_argument(
        "--point_val_data",
        type=str,
        default=None,
        help="Path to point cloud validation data",
    )

    parser.add_argument(
        "--pc_root_path",
        default="/datasets01/imagenet_full_size/061417/",
        type=str,
        help="dataset path",
    )
    parser.add_argument(
        "--pc_data_path", type=str, default=None, help="point cloud data path"
    )
    parser.add_argument(
        "--pc_image_data_path",
        type=str,
        default=None,
        help="point cloud image data path",
    )
    parser.add_argument(
        "--pc_text_data_path", type=str, default=None, help="point cloud text data path"
    )

    parser.add_argument(
        "--pc_whole_data", action="store_true", default=False, help="use whole data"
    )

    parser.add_argument(
        "--pc_nb_classes",
        default=40,
        type=int,
        help="number of the classification types",
    )
    parser.add_argument(
        "--pc_dataset_n_points",
        default=40,
        type=int,
        help="number of the points",
    )
    parser.add_argument(
        "--pc_n_points",
        default=40,
        type=int,
        help="number of the points",
    )
    parser.add_argument(
        "--pc_group_size", default=32, type=int, help="size of point cloud groups"
    )
    parser.add_argument(
        "--pc_num_group", default=64, type=int, help="number of point cloud groups"
    )
    parser.add_argument("--pc_logits_path", type=str, default="")
    parser.add_argument("--pc_logits_name", type=str, default=None)
    parser.add_argument("--pc_text_logits_name", type=str, default=None)
    parser.add_argument("--pc_image_logits_name", type=str, default=None)
    parser.add_argument("--pc_topk", type=int, default=40)

    parser.add_argument(
        "--point_train_data_prompt",
        default=None,
        type=str,
        help="Prompt template for training set, if any",
    )

    parser.add_argument(
        "--point_val_data_prompt",
        default=None,
        type=str,
        help="Prompt template for val set, if any",
    )

    parser.add_argument(
        "--pc_in_channel",
        type=int,
        default=3,
        choices=[3, 6],
        help="point cloud input channels",
    )
    parser.add_argument(
        "--pc_trans_dim",
        default=384,
        type=int,
        help="point cloud transformer dim, final dim of tokenizer",
    )
    parser.add_argument(
        "--pc_encoder_dims",
        default=256,
        type=int,
        help="point cloud tokenizer intermediate dim",
    )

    parser.add_argument(
        "--USE_NORMALS",
        default=False,
        type=bool,
        help="use normals",
    )

    parser.add_argument(
        "--output_dir",
        default="./output_dir",
        help="path where to save, empty for no saving",
    )
    parser.add_argument(
        "--log_dir", default="./output_dir", help="path where to tensorboard log"
    )
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")

    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
    parser.add_argument(
        "--dist_eval",
        action="store_true",
        default=False,
        help="Enabling distributed evaluation (recommended during training for faster monitor",
    )
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    parser.add_argument(
        "--distributed", action="store_true", help="Run distributed training"
    )
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )

    parser.add_argument(
        "--audio_data_train",
        type=str,
        default="/checkpoint/berniehuang/ast/egs/audioset/data/datafiles/train_video.json",
        help="training data json",
    )
    parser.add_argument(
        "--audio_data_eval",
        type=str,
        default="/checkpoint/berniehuang/ast/egs/audioset/data/datafiles/eval_video.json",
        help="validation data json",
    )
    parser.add_argument(
        "--audio_label_csv",
        type=str,
        default="/checkpoint/berniehuang/ast/egs/audioset/data/class_labels_indices.csv",
        help="csv with class labels",
    )
    parser.add_argument(
        "--audio_dataset",
        type=str,
        default="audioset",
        help="the dataset used",
    )
    parser.add_argument(
        "--audio_train_data", type=str, default=None, help="audio train data"
    )
    parser.add_argument(
        "--audio_val_data", type=str, default=None, help="audio val data"
    )
    parser.add_argument(
        "--audio_nb_classes", default=527, type=int, help="number of audio classes"
    )
    parser.add_argument("--audio_logits_path", type=str, default="")
    parser.add_argument("--audio_logits_name", type=str, default=None)
    parser.add_argument("--audio_text_logits_name", type=str, default=None)
    parser.add_argument("--audio_image_logits_name", type=str, default=None)
    
    parser.add_argument("--audio_topk", type=int, default=527)
    parser.add_argument("--use_fbank", type=bool, default=False)
    parser.add_argument(
        "--audio_clip_duration",
        type=float,
        default=5.0,
        help="length for audio (in seconds)",
    )
    parser.add_argument("--audio_stride", type=int, default=16)
    parser.add_argument("--use_soft", type=bool, default=False)
    parser.add_argument(
        "--freqm", help="frequency mask max length", type=int, default=48
    )
    parser.add_argument("--timem", help="time mask max length", type=int, default=192)
    parser.add_argument(
        "--fbank_dir",
        type=str,
        default="/checkpoint/berniehuang/ast/egs/esc50/data/ESC-50-master/fbank",
        help="fbank dir",
    )
    parser.add_argument(
        "--audio_load_vision",
        default=False,
        action="store_true",
        help="whether load vision (video or image) data for audio modality.",
    )
    parser.add_argument(
        "--use_custom_patch",
        action="store_true",
        default=False,
        help="use custom patch with overlapping and override timm PatchEmbed",
    )
    parser.add_argument(
        "--audio_sampling_rate",
        type=int,
        default=16000,
        help="Sampling rate (in Hz) for audio.",
    )
    parser.add_argument(
        "--audio_noise_aug",
        default=False,
        action="store_true",
        help="whether use noise augmentation.",
    )
    parser.add_argument(
        "--audio_mel_bins",
        type=int,
        default=128,
        help="mel bins for audio mel spectrogram",
    )
    parser.add_argument(
        "--audio_target_length",
        type=int,
        default=512,
        help="target length for audio mel spectrogram",
    )
    parser.add_argument(
        "--audio_n_frames", type=int, default=3, help="sample #frames for video input."
    )
    parser.add_argument(
        "--audio_mix_up",
        default=False,
        action="store_true",
        help="whether use mix_up for audio related training.",
    )
    parser.add_argument(
        "--audio_mix_up_p",
        type=float,
        default=0.3,
        help="Audio training mixup propability.",
    )
    parser.add_argument(
        "--roll_mag_aug", type=bool, default=False, help="use roll_mag_aug"
    )
    parser.add_argument(
        "--mask_t_prob",
        default=0.0,
        type=float,
        help="T masking ratio (percentage of removed patches).",
    )  #
    parser.add_argument(
        "--mask_f_prob",
        default=0.0,
        type=float,
        help="F masking ratio (percentage of removed patches).",
    )  #
    parser.add_argument(
        "--weight_sampler",
        action="store_true",
        default=False,
        help="use weight_sampler",
    )
    parser.add_argument(
        "--epoch_len",
        default=200000,
        type=int,
        help="num of samples/epoch with weight_sampler",
    )
    parser.add_argument(
        "--distributed_wrapper",
        type=bool,
        default=False,
        help="use distributedwrapper for weighted sampler",
    )
    parser.add_argument(
        "--replacement", action="store_true", default=False, help="use weight_sampler"
    )
    parser.add_argument("--mask_2d", type=bool, default=True, help="use 2d masking")
    parser.add_argument("--load_video", type=bool, default=False, help="load video")

    parser.add_argument(
        "--batch_mode",
        type=str,
        default="ratio",
        help="how to assign batch size",
    )
    parser.add_argument(
        "--save_logits", action="store_true", default=False, help="save logits"
    )
    parser.add_argument("--use_adapter", action="store_true", help="use adapter")
    parser.add_argument("--fuse", action="store_true", help="fuse")
    parser.add_argument("--distill", action="store_true", default=False, help="distill")
    parser.add_argument("--img_weight", type=float, default=1.0, help="image weight")
    parser.add_argument("--audio_weight", type=float, default=1.0, help="audio weight")
    parser.add_argument(
        "--pc_weight", type=float, default=1.0, help="point cloud weight"
    )
    parser.add_argument(
        "--audio_global_pool",
        default=False,
        action="store_true",
        help="audio global pool",
    )
    parser.add_argument(
        "--img_rep_w", type=float, default=1.0, help="img repeat weight"
    )
    parser.add_argument(
        "--audio_rep_w", type=float, default=1.0, help="audio repeat weight"
    )
    parser.add_argument("--pc_rep_w", type=float, default=1.0, help="pc repeat weight")
    parser.add_argument("--concat", action="store_true", help="concatenate datasets")
    parser.add_argument("--use_flash_attn", action="store_true", help="use flash attn")
    parser.add_argument(
        "--frozen_backbone", action="store_true", default=False, help="frozen backbone"
    )
    parser.add_argument(
        "--audio_weight_csv",
        type=str,
        default="/checkpoint/berniehuang/mae/data/audioset/weight_train_all.csv",
        help="weight file",
    )
    parser.add_argument("--grad_norm", action="store_true", help="use grad norm")
    parser.add_argument(
        "--grad_norm_alpha", type=float, default=0.12, help="grad norm alpha"
    )
    parser.add_argument("--modal_nums", type=int, default=3, help="modal nums")
    parser.add_argument("--log_name", type=str, default="main", help="log name")
    parser.add_argument("--use_loramoe", action="store_true", help="use lora moe")

    # lora
    parser.add_argument(
        "--lora_trainable", type=str, default="model", help="trainable modules"
    )
    parser.add_argument("--lora_rank", type=int, default=8, help="lora rank")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="lora dropout")
    parser.add_argument("--lora_alpha", type=float, default=32.0, help="lora alpha")
    parser.add_argument(
        "--modules_to_save", type=str, default=None, help="modules to save"
    )
    parser.add_argument("--lora_nums", type=int, default=2, help="lora nums")

    parser.add_argument(
        "--audio_distill_dim", type=int, default=1536, help="audio distill dim"
    )
    parser.add_argument(
        "--point_distill_dim", type=int, default=512, help="point distill dim"
    )
    parser.add_argument(
        "--rgbd_distill_dim", type=int, default=768, help="rgbd distill dim"
    )

    parser.add_argument("--use_text", action="store_true", help="use text")

    parser.add_argument(
        "--use_text_template", action="store_true", help="use text template"
    )

    parser.add_argument(
        "--img_text_feature_path", type=str, default=None, help="img text feature path"
    )
    parser.add_argument(
        "--audio_text_feature_path",
        type=str,
        default=None,
        help="audio text feature path",
    )
    parser.add_argument(
        "--point_text_feature_path",
        type=str,
        default=None,
        help="point text feature path",
    )

    parser.add_argument("--train_modal_list", nargs="+")
    parser.add_argument("--eval_modal_list", nargs="+")
    parser.add_argument("--model_modal_list", nargs="+")

    parser.add_argument("--cross_align", action="store_true", help="cross align")
    parser.add_argument("--uni_align", action="store_true", help="uni align")
    parser.add_argument("--temperature", type=float, default=4.0, help="temperature")
    parser.add_argument(
        "--text_embed_dim", type=int, default=1536, help="text embed adim"
    )
    parser.add_argument(
        "--rgbd_train_data", type=str, default="sun-rgbd", help="rgbd train dataset"
    )
    parser.add_argument(
        "--rgbd_val_data", type=str, default="sun-rgbd", help="rgbd val dataset"
    )
    parser.add_argument(
        "--rgbd_logits_path", type=str, default="", help="rgbd logits path"
    )
    parser.add_argument("--rgbd_logits_name", type=str, default=None)
    parser.add_argument("--rgbd_text_logits_name", type=str, default=None)
    parser.add_argument("--rgbd_image_logits_name", type=str, default=None)

    parser.add_argument("--rgbd_topk", type=int, default=768, help="rgbd topk")
    parser.add_argument("--rgbd_rep_w", type=float, default=1.0, help="rgbd rep weight")
    parser.add_argument("--rgbd_weight", type=float, default=1.0, help="rgbd weight")
    parser.add_argument(
        "--use_openclip_transform",
        action="store_true",
        default=False,
        help="whether use openclip transform in training.",
    )
    parser.add_argument(
        "--rgbd_train_text_feature_path",
        type=str,
        default=None,
        help="rgbd train text feature path",
    )
    parser.add_argument(
        "--rgbd_sunrgbd_val_text_feature_path",
        type=str,
        default=None,
        help="rgbd val text feature path",
    )
    parser.add_argument(
        "--rgbd_nyu_val1_text_feature_path",
        type=str,
        default=None,
        help="rgbd val text feature path",
    )
    parser.add_argument(
        "--rgbd_nyu_val2_text_feature_path",
        type=str,
        default=None,
        help="rgbd val text feature path",
    )

    parser.add_argument(
        "--rgbd_pretrained_path", type=str, default=None, help="rgbd pretrained path"
    )
    parser.add_argument(
        "--use_depth_only", action="store_true", default=False, help="use depth only"
    )

    # Video
    parser.add_argument(
        "--video_dataset",
        default="Kinetics-400",
        
        type=str,
        help="dataset",
    )
    parser.add_argument("--video_data_path", type=str, default="")
    parser.add_argument("--video_data_root", type=str, default="")
    parser.add_argument("--video_nb_classes", type=int, default=400)
    parser.add_argument("--video_input_size", type=int, default=224)
    parser.add_argument(
        "--video_fname_tmpl",
        default="img_{:05}.jpg",
        type=str,
        help="filename_tmpl for rawframe dataset",
    )
    parser.add_argument(
        "--video_start_idx", default=1, type=int, help="start_idx for rwaframe dataset"
    )
    parser.add_argument("--video_num_segments", type=int, default=1)
    parser.add_argument("--video_num_frames", type=int, default=16)
    parser.add_argument("--video_sampling_rate", type=int, default=4)
    parser.add_argument("--video_sparse_sample", default=False, action="store_true")
    parser.add_argument("--video_tubelet_size", type=int, default=2)
    parser.add_argument("--video_crop_pct", type=float, default=None)
    parser.add_argument("--video_short_side_size", type=int, default=224)
    parser.add_argument("--video_test_num_segment", type=int, default=10)
    parser.add_argument("--video_test_num_crop", type=int, default=3)

    parser.add_argument(
        "--video_color_jitter",
        type=float,
        default=0.4,
        metavar="PCT",
        help="Color jitter factor (default: 0.4)",
    )
    parser.add_argument(
        "--video_num_sample", type=int, default=1, help="Repeated_aug (default: 1)"
    )
    (
        parser.add_argument(
            "--video_aa",
            type=str,
            default="rand-m7-n4-mstd0.5-inc1",
            metavar="NAME",
            help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m7-n4-mstd0.5-inc1)',
        ),
    )
    parser.add_argument(
        "--video_smoothing",
        type=float,
        default=0.1,
        help="Label smoothing (default: 0.1)",
    )
    parser.add_argument(
        "--video_train_interpolation",
        type=str,
        default="bicubic",
        help='Training interpolation (random, bilinear, bicubic default: "bicubic")',
    )

    # Video Random Erase params
    parser.add_argument(
        "--video_reprob",
        type=float,
        default=0.25,
        metavar="PCT",
        help="Random erase prob (default: 0.25)",
    )
    parser.add_argument(
        "--video_remode",
        type=str,
        default="pixel",
        help='Random erase mode (default: "pixel")',
    )
    parser.add_argument(
        "--video_recount", type=int, default=1, help="Random erase count (default: 1)"
    )
    parser.add_argument(
        "--video_resplit",
        action="store_true",
        default=False,
        help="Do not random erase first (clean) augmentation split",
    )

    parser.add_argument("--video_2d_patch", action="store_true", help="use 2d patch")

    parser.add_argument(
        "--video_text_feature_path",
        type=str,
        default=None,
        help="video text feature path",
    )
    parser.add_argument(
        "--video_pretrained_path", type=str, default=None, help="video pretrained path"
    )
    parser.add_argument(
        "--video_rep_w", type=float, default=1.0, help="video rep weight"
    )
    parser.add_argument("--video_weight", type=float, default=1.0, help="video weight")

    parser.add_argument(
        "--video_train_csv",
        type=str,
        default="data/MSR-VTT/anns/MSRVTT_train.9k.csv",
        help="",
    )
    parser.add_argument(
        "--video_val_csv",
        type=str,
        default="data/MSR-VTT/anns/MSRVTT_JSFUSION_test.csv",
        help="",
    )
    parser.add_argument(
        "--video_features_path",
        type=str,
        default="s3://video_pub/MSR-VTT/videos",
        help="feature path",
    )
    parser.add_argument("--video_max_words", type=int, default=77, help="")
    parser.add_argument("--video_feature_framerate", type=int, default=1, help="")

    parser.add_argument(
        "--loose_type",
        action="store_true",
        help="Default using tight type for retrieval.",
    )
    parser.add_argument("--expand_msrvtt_sentences", action="store_true", help="")

    parser.add_argument(
        "--video_train_frame_order",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="Frame order, 0: ordinary order; 1: reverse order; 2: random order.",
    )
    parser.add_argument(
        "--video_eval_frame_order",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="Frame order, 0: ordinary order; 1: reverse order; 2: random order.",
    )

    parser.add_argument(
        "--video_slice_framepos",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.",
    )

    parser.add_argument("--video_topk", type=int, default=768, help="video")
    parser.add_argument(
        "--video_distill_dim", type=int, default=768, help="video distill dim"
    )
    parser.add_argument(
        "--video_logits_path", type=str, default="", help="video logits path"
    )
    parser.add_argument("--video_logits_name", type=str, default=None)
    parser.add_argument('--video_text_logits_name', type=str, default=None)
    parser.add_argument('--video_image_logits_name', type=str, default=None)

    parser.add_argument(
        "--video_train_data", type=str, default=None, help="video train data"
    )
    parser.add_argument(
        "--video_val_data", type=str, default=None, help="video val data"
    )

    parser.add_argument("--save_best", action="store_true", help="save best model")

    parser.add_argument(
        "--audio_text_template_path",
        type=str,
        default=None,
        help="audio text template path",
    )
    parser.add_argument(
        "--point_text_template_path",
        type=str,
        default=None,
        help="point text template path",
    )
    parser.add_argument(
        "--video_text_template_path",
        type=str,
        default=None,
        help="video text template path",
    )
    parser.add_argument(
        "--rgbd_text_template_path",
        type=str,
        default=None,
        help="rgbd text template path",
    )

    parser.add_argument("--use_peft", action="store_true")
    parser.add_argument("--moe_type", type=str, default=None, help="peft type")
    parser.add_argument("--use_moe_loss", action="store_true")

    parser.add_argument("--local_loss", action="store_true")
    parser.add_argument("--gather_with_grad", action="store_true")

    parser.add_argument(
        "--patch_drop_rate", type=float, default=0.0, help="patch drop rate"
    )

    parser.add_argument("--anchor_align", action="store_true")

    parser.add_argument("--use_sigliploss", action="store_true")
    parser.add_argument("--use_clip_loss", action="store_true")
    parser.add_argument(
        "--task_balancer",
        type=str,
        default="none",
        help="Task balancing scheme. One out of [uncertainty, none] (default: %(default)s)",
    )
    parser.add_argument(
        "--continue_train", type=str, default=None, help="continue training"
    )
    parser.add_argument(
        "--continue_train_path", type=str, default=None, help="continue training path"
    )
    parser.add_argument("--use_pc_image", action="store_true")

    parser.add_argument("--use_modality_adapter", action="store_true")
    parser.add_argument(
        "--align_train",
        type=str,
        default=None,
    )
    parser.add_argument("--use_orthogonal_loss", action="store_true")
    parser.add_argument("--use_aux_cls_loss", action="store_true")
    parser.add_argument(
        "--use_text_branch", action="store_true", default=False, help="use text encoder"
    )

    parser.add_argument(
        "--multi_modal_distill",
        action="store_true",
        default=False,
        help="Whether to use multi-modal distillation",
    )
    parser.add_argument(
        "--multi_modal_distill_modal_list", nargs="+", help="Which modal to distill"
    )

    parser.add_argument(
        "--alpha_ckd_loss", type=float, default=0.0, help="CRD loss weight"
    )
    parser.add_argument(
        "--alpha_icl_loss", type=float, default=0.0, help="ICL_loss weight"
    )
    parser.add_argument(
        "--alpha_cross_kd_loss", type=float, default=0.0, help="cross_kd_loss weight"
    )
    parser.add_argument(
        "--alpha_fd_loss", type=float, default=2000.0, help="FD_loss weight"
    )
    parser.add_argument(
        "--alpha_gd_loss", type=float, default=0.0, help="gd_loss weight"
    )
    parser.add_argument(
        "--alpha_afd_loss", type=float, default=0.0, help="AFD_loss weight"
    )
    parser.add_argument(
        "--visual_stat_flops", action="store_true", help="visualize flops"
    )

    parser.add_argument("--depth_channel", type=int, default=1)

    parser.add_argument("--moe_topk", type=int, default=1)

    parser.add_argument("--expert_nums", type=int, default=1)

    return parser


def main(args):
    if args.distributed:
        misc.init_distributed_mode(args)
    # logger
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = os.path.join(args.log_dir, f"{timestamp}.log")
    logger = get_root_logger(log_file=log_file, name=args.log_name)

    print_log(
        "job dir: {}".format(os.path.dirname(os.path.realpath(__file__))), logger=logger
    )
    print_log("{}".format(args).replace(", ", ",\n"), logger=logger)

    device = torch.device(args.device)

    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    global_rank = misc.get_rank()

    if global_rank == 0 and args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    start_epoch = 0
    tokenizer = SimpleTokenizer()
    preprocess_train = build_transform(True, args)
    preprocess_val = build_transform(False, args)
    data, eff_batch_size, real_batch_size = get_joint_data(
        args,
        epoch=start_epoch,
        tokenizer=tokenizer,
        image_transform=(preprocess_train, preprocess_val),
        logger=logger,
    )
    data_loaders_train = data["train"]

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0.0 or args.cutmix_minmax is not None
    if mixup_active:
        print_log("Mixup is activated!", logger=logger)
        mixup_fn = Mixup(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            mode=args.mixup_mode,
            label_smoothing=args.smoothing,
            num_classes=args.image_nb_classes,
        )

    if "clip" in args.finetune:
        model_type = "clip"
    else:
        model_type = "mae"

    train_modal_list = args.train_modal_list
    eval_modal_list = args.eval_modal_list
    model_modal_list = args.model_modal_list

    global_pool_dict = {
        "image": False,
        "audio": args.audio_global_pool,
        "point": False,
        "rgbd": False if args.depth_channel == 3 else True,
        "video": False,
    }

    distill_feature_dict = {
        "audio": args.audio_distill_dim,
        "point": args.point_distill_dim,
        "rgbd": args.rgbd_distill_dim,
        "audio": args.audio_distill_dim,
        "video": args.audio_distill_dim,
    }

    has_cls_head = {
        "image": True if "image" in args.train_modal_list else False,
        "audio": False,
        "point": False,
        "rgbd": False,
        "video": False,
    }

    model = vit_one.__dict__[args.model](
        has_cls_head=has_cls_head,
        global_pool_dict=global_pool_dict,
        distill_feature_dict=distill_feature_dict,
        audio_length=args.audio_target_length,
        num_classes=args.image_nb_classes,
        drop_path_rate=args.drop_path,
        mask_2d=args.mask_2d,
        use_custom_patch=args.use_custom_patch,
        model_type=model_type,
        modals=model_modal_list,
        args=args,
    )

    open_clip_text_model = None
    if args.use_text_branch:
        import open_clip

        open_clip_text_model, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-16", pretrained="laion2b_s34b_b88k"
        )
        open_clip_text_model.eval()
        open_clip_text_model.visual = None
        open_clip_text_model.to(device)

    if args.finetune:
        if "clip" in args.finetune:
            checkpoint_model = timm.create_model(
                args.finetune, pretrained=True
            ).state_dict()
        else:
            checkpoint_model = torch.load(args.finetune, map_location="cpu")["model"]

            print_log(
                "Load pre-trained checkpoint from: %s" % args.finetune, logger=logger
            )
            # checkpoint_model = checkpoint.state_dict()
            state_dict = model.state_dict()
            if not args.eval:
                for k in ["head.weight", "head.bias"]:
                    if (
                        k in checkpoint_model
                        and checkpoint_model[k].shape != state_dict[k].shape
                    ):
                        print_log(
                            f"Removing key {k} from pretrained checkpoint",
                            logger=logger,
                        )
                        del checkpoint_model[k]
            # interpolate position embedding
            interpolate_pos_embed(model, checkpoint_model)

        replace_list = [
            "patch_embed",
            "pos_embed",
            "norm.",
            "head",
            "cls_token",
        ]

        for k, v in list(checkpoint_model.items()):
            for i in replace_list:
                if i in k:
                    if i == "norm.":
                        new_k = k.replace(i, i + "image.")
                        checkpoint_model[new_k] = v
                        del checkpoint_model[k]

                    elif i == "head":
                        if "ft" in args.finetune:
                            new_k = k.replace(i, i + ".image")
                            checkpoint_model[new_k] = v

                        del checkpoint_model[k]
                    else:
                        new_k = k.replace(i, i + ".image")
                        checkpoint_model[new_k] = v
                        del checkpoint_model[k]

        if args.use_flash_attn:
            flash_dict = {}
            for k, v in checkpoint_model.items():
                if "attn.qkv" in k:
                    flash_k = k.replace("attn.qkv", "attn.Wqkv")
                    flash_dict[flash_k] = v
                elif "attn.proj" in k:
                    flash_k = k.replace("attn.proj", "attn.out_proj")
                    flash_dict[flash_k] = v
                else:
                    flash_dict[k] = v
            msg = model.load_state_dict(flash_dict, strict=False)
            print_log(msg, logger=logger)
        else:
            
            msg = model.load_state_dict(checkpoint_model, strict=False)
            print_log(msg, logger=logger)

    if args.use_peft:
        peft_cfg = get_peft_cfg(args)
        lora_cfg = LoraConfig(**peft_cfg)
        model = LoraModel(lora_cfg, model)
        for key, value in peft_cfg.items():
            print_log(f"{key}: {value}", logger=logger)

    # =========================================== Load Pretrained Models
    if "audio" in model_modal_list:
        audio_pretrained_model = torch.load(args.audio_pretrained_path)["model"]
        import collections

        if isinstance(audio_pretrained_model, collections.OrderedDict):
            if "model" in audio_pretrained_model:
                audio_aux_state_dict = audio_pretrained_model["model"]
            else:
                audio_aux_state_dict = audio_pretrained_model

        elif isinstance(audio_pretrained_model, nn.Module):
            audio_aux_state_dict = audio_pretrained_model.state_dict()

        
    if "point" in model_modal_list:
        point_pretrained_model = torch.load(args.point_pretrained_path)["base_model"]
        import collections

        if isinstance(point_pretrained_model, collections.OrderedDict):
            if "base_model" in point_pretrained_model:
                point_aux_state_dict = point_pretrained_model["base_model"]

            else:
                point_aux_state_dict = point_pretrained_model
        elif isinstance(point_pretrained_model, nn.Module):
            point_aux_state_dict = point_pretrained_model.state_dict()

    if "rgbd" in model_modal_list:
        if args.depth_channel == 3:
            if "openai" in args.finetune:
                rgbd_pretrained_model = timm.create_model(
                    "vit_base_patch16_clip_224.openai", pretrained=True
                ).state_dict()
            else:
                rgbd_pretrained_model = timm.create_model(
                    "vit_base_patch16_clip_224.laion2b", pretrained=True
                ).state_dict()
            rgbd_aux_state_dict = rgbd_pretrained_model
        else:
            rgbd_aux_state_dict = None

    #     rgbd_pretrained_model = torch.load(args.rgbd_pretrained_path)["model"]
    #     import collections

    #     if isinstance(rgbd_pretrained_model, collections.OrderedDict):
    #         if "model" in rgbd_pretrained_model:
    #             rgbd_aux_state_dict = rgbd_pretrained_model["model"]
    #         else:
    #             rgbd_aux_state_dict = rgbd_pretrained_model

    #     elif isinstance(rgbd_pretrained_model, nn.Module):
    #         rgbd_aux_state_dict = rgbd_pretrained_model.state_dict()

    if "video" in model_modal_list:
        if args.video_2d_patch:
            if "openai" in args.finetune:
                video_pretrained_model = timm.create_model(
                    "vit_base_patch16_clip_224.openai", pretrained=True
                ).state_dict()
            else:
                video_pretrained_model = timm.create_model(
                    "vit_base_patch16_clip_224.laion2b", pretrained=True
                ).state_dict()
            video_aux_state_dict = video_pretrained_model
        else:
            video_pretrained_model = torch.load(args.video_pretrained_path)["module"]
            import collections

            if isinstance(video_pretrained_model, collections.OrderedDict):
                
                video_aux_state_dict = video_pretrained_model

            elif isinstance(video_pretrained_model, nn.Module):
                video_aux_state_dict = video_pretrained_model.state_dict()

    if "openai" in args.finetune:
        clip_modal = timm.create_model(
            "vit_base_patch16_clip_224.openai", pretrained=True
        ).state_dict()
    else:
        clip_modal = timm.create_model(
            "vit_base_patch16_clip_224.laion2b", pretrained=True
        ).state_dict()
    clip_proj_dict = {}

    for k, v in clip_modal.items():
        if "head" in k:
            for m in model_modal_list:
                if args.use_peft:
                    if args.text_embed_dim == 1536:
                        new_k = k.replace("head", f"model.text_projector.{m}.clip_proj")
                    else:
                        new_k = k.replace("head", f"model.text_projector.{m}")
                else:
                    if args.text_embed_dim == 1536:
                        new_k = k.replace("head", f"text_projector.{m}.clip_proj")
                    else:
                        new_k = k.replace("head", f"text_projector.{m}")
            
                clip_proj_dict[new_k] = v

    state_dict = model.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():
        if "audio" in k:
            if args.text_embed_dim == 1536:
                if "clip_proj" in k:
                    new_state_dict[k] = clip_proj_dict[k]
                    print_log(f"Init {k}", logger=logger)
                    continue
            else:
                if "text" in k:
                    new_state_dict[k] = clip_proj_dict[k]
                    print_log(f"Init {k}", logger=logger)
                    continue

            ori_k = k.replace("model.", "")
            ori_k = ori_k.replace(".audio", "")
            if (
                ori_k in audio_aux_state_dict
                and state_dict[k].shape == audio_aux_state_dict[ori_k].shape
            ):
                new_state_dict[k] = audio_aux_state_dict[ori_k]
                print_log(f"Init {k}", logger=logger)
            else:
                if "pos_embed" in k:
                    num_patches = state_dict[k].shape[1] - 1
                    new_pos_embed = (
                        audio_aux_state_dict[ori_k][:, 1:, :]
                        .transpose(1, 2)
                        .reshape(1, 768, 8, 64)
                    )
                    t_dim = 8 if args.audio_dataset == "speechcommands" else 32
                    # f_dim = 8
                    new_pos_embed = new_pos_embed[
                        :, :, :, 32 - int(t_dim / 2) : 32 - int(t_dim / 2) + t_dim
                    ]
                    new_pos_embed = new_pos_embed.reshape(
                        1, 768, num_patches
                    ).transpose(1, 2)
                    new_state_dict[k] = torch.cat(
                        [
                            audio_aux_state_dict[ori_k][:, 0, :].unsqueeze(1),
                            new_pos_embed,
                        ],
                        dim=1,
                    )
                    assert state_dict[k].shape == new_state_dict[k].shape
                    print_log("cut positional embedding", logger=logger)

                    
                else:
                    new_state_dict[k] = v
                    if "lora" not in k:
                        print_log(
                            f"Key {k} not found in audio_aux_state_dict",
                            logger=logger,
                        )
            
            

        elif "point" in k:
            if args.text_embed_dim == 1536:
                if "clip_proj" in k:
                    new_state_dict[k] = clip_proj_dict[k]
                    print_log(f"Init {k}", logger=logger)
                    continue
            else:
                if "text" in k:
                    new_state_dict[k] = clip_proj_dict[k]
                    print_log(f"Init {k}", logger=logger)
                    continue
            ori_k = k.replace("model.", "")
            if "patch_embed" in k:
                ori_k = ori_k.replace("patch_embed.point", "module.model.embed")
            
            elif "pos_embed" in k:
                ori_k = ori_k.replace("pos_embed.point", "module.model.pos_embed")
            elif "norm" in k:
                ori_k = ori_k.replace("norm.point", "module.model.norm")
            else:
                ori_k = ori_k.replace(".point", "")

            if (
                ori_k in point_aux_state_dict
                and state_dict[k].shape == point_aux_state_dict[ori_k].shape
            ):
                new_state_dict[k] = point_aux_state_dict[ori_k]
                print_log(f"Init {k}", logger=logger)
            else:
                if "patch_embed" in k:
                    new_state_dict[k] = point_aux_state_dict[ori_k][:, :3, :]
                else:
                    new_state_dict[k] = v
                    if "lora" not in k:
                        print_log(
                            f"Key {k} not found in point_aux_state_dict",
                            logger=logger,
                        )
        elif "video" in k:
            if args.text_embed_dim == 1536:
                if "clip_proj" in k:
                    new_state_dict[k] = clip_proj_dict[k]
                    print_log(f"Init {k}", logger=logger)
                    continue
            else:
                if "text" in k:
                    new_state_dict[k] = clip_proj_dict[k]
                    print_log(f"Init {k}", logger=logger)
                    continue
            ori_k = k.replace("model.", "")
            ori_k = ori_k.replace(".video", "")
            if (
                ori_k in video_aux_state_dict
                and state_dict[k].shape == video_aux_state_dict[ori_k].shape
            ):
                new_state_dict[k] = video_aux_state_dict[ori_k]
                print_log(f"Init {k}", logger=logger)
            else:
                new_state_dict[k] = v
                if "lora" not in k:
                    print_log(
                        f"Key {k} not found in video_aux_state_dict", logger=logger
                    )
        elif "rgbd" in k:
            if args.text_embed_dim == 1536:
                if "clip_proj" in k:
                    new_state_dict[k] = clip_proj_dict[k]
                    print_log(f"Init {k}", logger=logger)
                    continue
                else:
                    new_state_dict[k] = v
            else:
                if "text" in k:
                    new_state_dict[k] = clip_proj_dict[k]
                    print_log(f"Init {k}", logger=logger)
                    continue
                else:
                    new_state_dict[k] = v
            if rgbd_aux_state_dict is not None:
                ori_k = k.replace("model.", "")
                ori_k = ori_k.replace(".rgbd", "")
                if (
                    ori_k in rgbd_aux_state_dict
                    and state_dict[k].shape == rgbd_aux_state_dict[ori_k].shape
                ):
                    new_state_dict[k] = rgbd_aux_state_dict[ori_k]
                    print_log(f"Init {k}", logger=logger)
                else:
                    new_state_dict[k] = v
                    if "lora" not in k:
                        print_log(
                            f"Key {k} not found in rgbd_aux_state_dict", logger=logger
                        )

        elif "image" in k:
            if args.text_embed_dim == 1536:
                if "clip_proj" in k:
                    new_state_dict[k] = clip_proj_dict[k]
                    print_log(f"Init {k}", logger=logger)
                    continue
                else:
                    new_state_dict[k] = v
            else:
                if "text" in k:
                    new_state_dict[k] = clip_proj_dict[k]
                    print_log(f"Init {k}", logger=logger)
                    continue
                else:
                    new_state_dict[k] = v
        else:
            if "text" in k or "clip_proj" in k:
                print_log(f"Init {k}", logger=logger)
                new_state_dict[k] = clip_proj_dict[k]
            else:
                new_state_dict[k] = v

    state_dict.update(new_state_dict)

    msg = model.load_state_dict(new_state_dict, strict=True)
    print_log(msg, logger=logger)

    if args.frozen_backbone:
        for name, param in model.named_parameters():
            param.requires_grad = True

        for name, param in model.named_parameters():
            if (
                "audio" in name
                or "point" in name
                or "lora_" in name
                or "text" in name
                or "video" in name
                or "rgbd" in name
                or "logit_scale" in name
                
            ):
                if "text" in name and args.use_text is False:
                    print_log(f"freeze text_proj {name}", logger=logger)
                    param.requires_grad = False
                elif "text" in name and "clip_proj" in name:
                    print_log(f"freeze text_proj {name}", logger=logger)
                    param.requires_grad = False
                elif "text" in name and "image" in name:
                    print_log(f"freeze text_proj {name}", logger=logger)
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            else:
                param.requires_grad = False
    else:
        for name, param in model.named_parameters():
            if "image" in name:
                param.requires_grad = False

    for name, param in model.named_parameters():
        print_log(f"{name}: {param.requires_grad}", logger=logger)

    model.to(device)

    if args.visual_stat_flops:
        from ptflops import get_model_complexity_info

        dset = data["train"]
        visual_inp, modal_inp = next(dset)
        resolution = ()

        if modal_inp == "image":
            
            resolution += visual_inp["image"].shape[1:]
            
        elif modal_inp == "audio":
            
            resolution += visual_inp["audio"].shape[1:]
        
        elif modal_inp == "point":
            
            resolution += visual_inp["pc"].shape[1:]
            resolution += visual_inp["image"].shape[1:]
        
        elif modal_inp == "rgbd":
            
            resolution += visual_inp["depth"].shape[1:]
            resolution += visual_inp["image"].shape[1:]
        
        resolution += (modal_inp,)

        def prepare_input(resolution):
            cur_modal = resolution[-1]
            if cur_modal == "image":
                image = torch.FloatTensor(1, *resolution[:-1]).to(
                    device, dtype=torch.float16
                )
                input = {"image": image}
                return {
                    "x_list": [input],
                    "modal_list": ["image"],
                    "anchor_list": ["image"],
                }
            elif cur_modal == "audio":
                audio = torch.FloatTensor(1, *resolution[:-1]).to(
                    device, dtype=torch.float16
                )
                input = {"audio": audio}
                return {
                    "x_list": [input],
                    "modal_list": ["audio"],
                    "anchor_list": ["audio"],
                }
            elif cur_modal == "point":
                point = torch.FloatTensor(1, *resolution[0]).to(
                    device, dtype=torch.float16
                )
                image = torch.FloatTensor(1, *resolution[1]).to(
                    device, dtype=torch.float16
                )
                input = {"point": point, "image": image}
                return {
                    "x_list": [input],
                    "modal_list": ["point", "image"],
                    "anchor_list": ["point", "image"],
                }
            elif cur_modal == "rgbd":
                depth = torch.FloatTensor(1, *resolution[0]).to(
                    device, dtype=torch.float16
                )
                image = torch.FloatTensor(1, *resolution[1]).to(
                    device, dtype=torch.float16
                )
                input = {"rgbd": depth, "image": image}
                return {
                    "x_list": [input],
                    "modal_list": ["rgbd", "image"],
                    "anchor_list": ["rgbd", "image"],
                }

        with torch.cuda.amp.autocast():
            macs_cnt, params_cnt = get_model_complexity_info(
                model,
                resolution,
                input_constructor=prepare_input,
                as_strings=True,
                print_per_layer_stat=False,
            )
        print_log(
            "{:<30}  {:<8}".format("Computational complexity: ", macs_cnt),
            logger=logger,
        )
        flops_cnt = f"{float(macs_cnt[:-5]) * 2} GFLOPS"  # Convert GMACS to GFLOPS assuming 'macs_cnt' ends with ' GMACS'
        print_log(
            "{:<30}  {:<8}".format("Computational complexity: ", flops_cnt),
            logger=logger,
        )
        print_log(
            "{:<30}  {:<8}".format("Number of parameters: ", params_cnt), logger=logger
        )
        del visual_inp
        # sys.exit(1)

    model_without_ddp = model

    loss_balancer = MultiModalUncertaintyWeightingStrategy(args)

    loss_balancer.to(device)
    loss_balancer_without_ddp = loss_balancer

    print_log("Model = %s" % str(model_without_ddp), logger=logger)
    num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_total_param = sum(p.numel() for p in model.parameters())
    print_log(
        f"Number of total parameters: {num_total_param/ 1.0e6}, tunable parameters: {num_param/ 1.0e6}",
        logger=logger,
    )

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / args.base_batchsize

    print_log(
        "base lr: %.2e" % (args.lr * args.base_batchsize / eff_batch_size),
        logger=logger,
    )
    print_log("actual lr: %.2e" % args.lr, logger=logger)

    print_log("accumulate grad iterations: %d" % args.accum_iter, logger=logger)
    print_log("effective batch size: %d" % eff_batch_size, logger=logger)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        
        model_without_ddp = model.module

        if args.use_text_branch:
            open_clip_text_model = torch.nn.parallel.DistributedDataParallel(
                open_clip_text_model, device_ids=[args.gpu], find_unused_parameters=True
            )

        if args.task_balancer != "none":
            loss_balancer = torch.nn.parallel.DistributedDataParallel(
                loss_balancer, device_ids=[args.gpu], find_unused_parameters=True
            )
            loss_balancer_without_ddp = loss_balancer.module

    # build optimizer with layer-wise lr decay (lrd)
    param_groups = lrd.param_groups_multimodal(
        model_without_ddp,
        train_modal_list,
        eff_batch_size / args.accum_iter,
        real_batch_size,
        loss_balancer_without_ddp if args.task_balancer != "none" else None,
        args.weight_decay,
        no_weight_decay_list=model_without_ddp.no_weight_decay(),
    )
    
    optimizer = torch.optim.AdamW(
        param_groups, lr=args.lr, betas=(0.9, 0.98), eps=1.0e-6
    )
    
    loss_scaler = NativeScaler()

    criterion = {}
    if mixup_fn is not None:
        # smoothing is handled with mixup label transform

        for m in train_modal_list:
            if m == "image":
                criterion["image"] = SoftTargetCrossEntropy()
                print_log("img_criterion = %s" % str(criterion["image"]), logger=logger)
            if m == "audio":
                criterion["audio"] = nn.BCEWithLogitsLoss()
                print_log(
                    "audio_criterion = %s" % str(criterion["audio"]), logger=logger
                )
            if m == "point":
                criterion["point"] = torch.nn.CrossEntropyLoss()
                print_log("pc_criterion = %s" % str(criterion["point"]), logger=logger)
            if m == "video":
                criterion["video"] = torch.nn.CrossEntropyLoss()
                print_log(
                    "video_criterion = %s" % str(criterion["video"]), logger=logger
                )
            
    else:
        for m in train_modal_list:
            if m == "image":
                criterion["image"] = torch.nn.CrossEntropyLoss()
                print_log("img_criterion = %s" % str(criterion["image"]), logger=logger)
            if m == "audio":
                criterion["audio"] = nn.BCEWithLogitsLoss()
                print_log(
                    "audio_criterion = %s" % str(criterion["audio"]), logger=logger
                )
            # if m == 'point':
            #     # criterion['point'] = torch.nn.CrossEntropyLoss()
            #     criterion['point'] = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
            #     print_log("pc_criterion = %s" % str(criterion['point']), logger=logger)
            # if m == "video":
            #     criterion["video"] = torch.nn.CrossEntropyLoss()
            #     print_log(
            #         "video_criterion = %s" % str(criterion["video"]), logger=logger
            #     )
            
    best_acc = 0
    best_metric = {}
    if "image" in args.eval_modal_list:
        best_metric["image"] = {"imagenet": {"acc1": 0.0, "acc5": 0.0}}
    if "audio" in args.eval_modal_list:
        best_metric["audio"] = {}
        if "audioset" in args.audio_val_data:
            best_metric["audio"].update({"audioset": {"mAP": 0.0}})
        if "speechcommands" in args.audio_val_data:
            best_metric["audio"].update({"speechcommands": {"acc1": 0.0, "mAUC": 0.0}})
        if "audiocaps" in args.audio_val_data:
            best_metric["audio"].update(
                {
                    "audiocaps": {
                        "txt_r1": 0.0,
                        "txt_r5": 0.0,
                        "txt_r10": 0.0,
                        "audio_r1": 0.0,
                        "audio_r5": 0.0,
                        "audio_r10": 0.0,
                    }
                }
            )
        if "esc50" in args.audio_val_data:
            best_metric["audio"].update(
                {
                    "esc50": {
                        "acc1": 0.0,
                    }
                }
            )
        if "clotho" in args.audio_val_data:
            best_metric["audio"].update(
                {
                    "clotho": {
                        "txt_r1": 0.0,
                        "txt_r5": 0.0,
                        "txt_r10": 0.0,
                        "audio_r1": 0.0,
                        "audio_r5": 0.0,
                        "audio_r10": 0.0,
                    }
                }
            )
    if "point" in args.eval_modal_list:
        best_metric["point"] = {}
        if "modelnet40" in args.point_val_data:
            best_metric["point"].update(
                {
                    "modelnet40": {
                        "acc1": 0.0,
                        "acc5": 0.0,
                    }
                }
            )
        if "scanobjectnn" in args.point_val_data:
            best_metric["point"].update(
                {
                    "scanobjectnn": {
                        "acc1": 0.0,
                        "acc5": 0.0,
                    }
                }
            )
    if "video" in args.eval_modal_list:
        best_metric["video"] = {}
        if "msrvtt" in args.video_val_data:
            best_metric["video"].update(
                {
                    "msrvtt": {
                        "tr_r10": 0.0,
                        "tr_r5": 0.0,
                        "tr_r1": 0.0,
                        "ir_r10": 0.0,
                        "ir_r5": 0.0,
                        "ir_r1": 0.0,
                    }
                }
            )
        if "ucf101" in args.video_val_data:
            best_metric["video"].update({"ucf101": {"acc1": 0.0, "acc5": 0.0}})
        
    if "rgbd" in args.eval_modal_list:
        best_metric["rgbd"] = {
            "sun-rgbd": {"acc1": 0.0, "acc5": 0.0},
            "nyu-depth-v2-val2": {"acc1": 0.0, "acc5": 0.0},
        }

    max_metric = copy.deepcopy(best_metric)

    resume_metric = misc.auto_resume_model(
        args=args,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
    )
    if resume_metric is not None:
        best_metric = resume_metric

    if args.eval:
        for modal in eval_modal_list:
            epoch = 0
            if modal == "image":
                test_image_stats = evaluate_image(
                    data["val"]["image"],
                    model,
                    open_clip_text_model,
                    tokenizer,
                    device,
                    args,
                )
                print_log(
                    f"Accuracy of the network on the {len(data['val']['image'])} test images: {test_image_stats['acc1']:.3f}%",
                    logger=logger,
                )
                
                    
            elif modal == "audio":
                test_audio_metrics = test_audiotasks_core(
                    data["val"]["audio"], model, open_clip_text_model, tokenizer, args
                )
                
            elif modal == "point":
                test_point_metrics = test_zeroshot_3d_core(
                    data["val"]["point"], model, open_clip_text_model, tokenizer, args
                )
                
            elif modal == "rgbd":
                test_rgbd_metrics = test_rgbd_cls_core(
                    data["val"]["rgbd"], model, open_clip_text_model, tokenizer, args
                )
                
            elif modal == "video":
                test_video_metrics = test_vidret_core(
                    data["val"]["video"], model, open_clip_text_model, tokenizer, args
                )
                

        exit(0)

    print_log(f"Start training for {args.epochs} epochs", logger=logger)
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        
        if args.distributed:
            if args.batch_mode == "use_all":
                data_loaders_train.sampler.set_epoch(epoch)
            else:
                data_loaders_train.set_epoch(epoch)

        if args.concat:
            if args.batch_mode == "use_all":
                train_stats = train_one_epoch_concat_use_all(
                    model,
                    criterion,
                    data_loaders_train,
                    optimizer,
                    device,
                    epoch,
                    loss_scaler,
                    args.clip_grad,
                    mixup_fn,
                    log_writer=log_writer,
                    args=args,
                )
            else:
                train_stats = train_one_epoch_concat(
                    model,
                    open_clip_text_model,
                    loss_balancer,
                    criterion,
                    data_loaders_train,
                    optimizer,
                    device,
                    epoch,
                    loss_scaler,
                    args.clip_grad,
                    mixup_fn,
                    log_writer=log_writer,
                    args=args,
                )

        metric_ep = 0.0
        for modal in eval_modal_list:
            if modal == "image":
                test_image_stats = evaluate_image(
                    data["val"]["image"],
                    model,
                    open_clip_text_model,
                    tokenizer,
                    device,
                    args,
                )

                metric_ep += test_image_stats["acc1"]

                print_log(
                    f"Accuracy of the network on the {len(data['val']['image'])} test images: {test_image_stats['acc1']:.3f}%",
                    logger=logger,
                )

                best_metric["image"]["imagenet"]["acc1"] = max(
                    best_metric["image"]["imagenet"]["acc1"], test_image_stats["acc1"]
                )
                best_metric["image"]["imagenet"]["acc5"] = max(
                    best_metric["image"]["imagenet"]["acc5"], test_image_stats["acc5"]
                )
                print_log(
                    f"Max Imgae acc1: {best_metric['image']['imagenet']['acc1']:.2f}%",
                    logger=logger,
                )

            elif modal == "audio":
                test_audio_metrics = test_audiotasks_core(
                    data["val"]["audio"], model, open_clip_text_model, tokenizer, args
                )
                if isinstance(data["val"]["audio"], dict):
                    for dname in max_metric["audio"]:
                        for k in max_metric["audio"][dname]:
                            max_metric["audio"][dname][k] = max(
                                max_metric["audio"][dname][k],
                                test_audio_metrics[dname][k],
                            )

                metric_ep += sum(
                    [test_audio_metrics[k]["acc1"] for k in test_audio_metrics]
                )
                print_log(f"Max audio metrics: {max_metric['audio']}", logger=logger)

            elif modal == "point":
                test_point_metrics = test_zeroshot_3d_core(
                    data["val"]["point"], model, open_clip_text_model, tokenizer, args
                )
                if isinstance(data["val"]["point"], dict):
                    for dname in max_metric["point"]:
                        for k in max_metric["point"][dname]:
                            max_metric["point"][dname][k] = max(
                                max_metric["point"][dname][k],
                                test_point_metrics[dname][k],
                            )

                    metric_ep += sum(
                        [test_point_metrics[k]["acc1"] for k in test_point_metrics]
                    )
                else:
                    for k in max_metric["point"]["modelnet40"]:
                        max_metric["point"]["modelnet40"][k] = max(
                            max_metric["point"]["modelnet40"][k],
                            test_point_metrics["modelnet40"][k],
                        )

                    metric_ep += test_point_metrics["modelnet40"]["acc1"]
                print_log(f"Max point metrics: {max_metric['point']}", logger=logger)

            elif modal == "rgbd":
                test_rgbd_metrics = test_rgbd_cls_core(
                    data["val"]["rgbd"], model, open_clip_text_model, tokenizer, args
                )
                for dname in max_metric["rgbd"]:
                    for k in max_metric["rgbd"][dname]:
                        max_metric["rgbd"][dname][k] = max(
                            max_metric["rgbd"][dname][k], test_rgbd_metrics[dname][k]
                        )

                metric_ep += sum(
                    [test_rgbd_metrics[k]["acc1"] for k in test_rgbd_metrics]
                )
                print_log(f"Max rgbd metrics: {max_metric['rgbd']}", logger=logger)

            elif modal == "video":
                test_video_metrics = test_vidret_core(
                    data["val"]["video"], model, open_clip_text_model, tokenizer, args
                )
                for dname in max_metric["video"]:
                    for k in max_metric["video"][dname]:
                        max_metric["video"][dname][k] = max(
                            max_metric["video"][dname][k], test_video_metrics[dname][k]
                        )
                metric_ep += sum(
                    [test_video_metrics[k]["acc1"] for k in test_video_metrics]
                )
                print_log(f"Max video metrics: {max_metric['video']}", logger=logger)

        better = metric_ep > best_acc
        if better:
            best_acc = metric_ep
            if "image" in args.eval_modal_list:
                for k in best_metric["image"]["imagenet"]:
                    best_metric["image"]["imagenet"][k] = test_image_stats[k]

            if "audio" in args.eval_modal_list:
                for dname in best_metric["audio"]:
                    for k in best_metric["audio"][dname]:
                        best_metric["audio"][dname][k] = test_audio_metrics[dname][k]

            if "point" in args.eval_modal_list:
                for dname in best_metric["point"]:
                    for k in best_metric["point"][dname]:
                        best_metric["point"][dname][k] = test_point_metrics[dname][k]

            if "rgbd" in args.eval_modal_list:
                for dname in best_metric["rgbd"]:
                    for k in best_metric["rgbd"][dname]:
                        best_metric["rgbd"][dname][k] = test_rgbd_metrics[dname][k]

            if "video" in args.eval_modal_list:
                for dname in best_metric["video"]:
                    for k in best_metric["video"][dname]:
                        best_metric["video"][dname][k] = test_video_metrics[dname][k]

        if better and args.output_dir and args.save_best:
            prefix = "best"
            print_log(f"Best Metric: {best_metric}", logger=logger)
            misc.save_checkpoint(
                args,
                epoch,
                model,
                model_without_ddp,
                optimizer,
                loss_scaler,
                prefix,
                best_metric,
                logger=logger,
            )

        if args.output_dir:
            prefix = "last"
            misc.save_checkpoint(
                args,
                epoch,
                model,
                model_without_ddp,
                optimizer,
                loss_scaler,
                prefix,
                best_metric,
                logger=logger,
            )

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            "epoch": epoch,
            "n_parameters": num_param,
        }
        for modal in eval_modal_list:
            if modal == "image":
                log_stats.update(
                    **{f"test_{k}": v for k, v in test_image_stats.items()}
                )
            elif modal == "audio":
                log_stats.update(
                    **{f"test_{k}": v for k, v in test_audio_metrics.items()}
                )
            elif modal == "point":
                log_stats.update(
                    **{f"test_{k}": v for k, v in test_point_metrics.items()}
                )
            elif modal == "rgbd":
                log_stats.update(
                    **{f"test_{k}": v for k, v in test_rgbd_metrics.items()}
                )
        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(
                os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8"
            ) as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(timedelta(seconds=int(total_time)))
    print_log("Training time {}".format(total_time_str), logger=logger)


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
