# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL

from torchvision import datasets, transforms
from torch.utils.data import Dataset
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import numpy as np


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    root = os.path.join(args.img_data_path, "train" if is_train else "val")
    dataset = datasets.ImageFolder(root, transform=transform)

    # if args.img_subset is not None and is_train:
    #     # random.seed(args.random_seed)  # Set the random seed
    #     # indices = list(range(len(dataset)))
    #     # random.shuffle(indices)
    #     # indices = indices[0:args.img_subset]
    #     # dataset = torch.utils.data.Subset(dataset, indices)

    #     samples_weight = torch.from_numpy(
    #         np.loadtxt(args.img_weight_csv, delimiter=",")
    #     )

    #     subsample_balanced_indicies = torch.multinomial(
    #         samples_weight, args.img_subset, False
    #     )

    #     dataset = torch.utils.data.Subset(dataset, subsample_balanced_indicies)

    print(dataset)
    print(len(dataset))
    # dataset.classes 
    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation="bicubic",
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(
            size, interpolation=PIL.Image.BICUBIC
        ),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)

class ImageNet(Dataset):
    def __init__(self, is_train, args):
        super().__init__()
        transform = build_transform(is_train, args)
        root = os.path.join(args.img_data_path, "train" if is_train else "val")
        
        self.dataset = datasets.ImageFolder(root, transform=transform)
        
    
    def __getitem__(self, index):
        image_data, label = self.dataset[index]
        
        return image_data, label

    def __len__(self):
        return len(self.dataset)
