import os
import random
import json
import logging

from typing import Iterable
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset
from omegaconf import OmegaConf
from easydict import EasyDict as edict

from datasets.Sample import Sample, SampleCollator
from datasets.constants import DEPTH_META_DATA_DIR, DEPTH_DATA_DIR
from util.logger import print_log
from zmq import device
from .data.scene_cls_template import SCENE_CLS_TEMPLATE
from .processors.vt_processor import (
    BlipCaptionProcessor,
    RGBD_Processor_Train,
    RGBD_Processor_Eval,
)

import torch

# from open_clip_train.distributed import is_master

# def load_annotation(filename):
#     if filename.endswith(".json"):
#         anno = json.load(open(filename, "r"))
#     elif filename.endswith(".tsv"):
#         anno = pd.read_csv(filename, sep="\t", header=None)
#     else:
#         raise NotImplementedError
#     return anno


def concat_datasets(datasets):
    if isinstance(datasets, dict):
        dataset_list = [d for d in datasets.values()]
    elif isinstance(datasets, list):
        dataset_list = datasets
    else:
        NotImplemented

    concat_dataset = ConcatDataset(dataset_list)
    return concat_dataset


class BaseDataset(Dataset):
    def __init__(
        self,
        vis_processor=None,
        text_processor=None,
        vis_root=None,
        anno_path={},
        args=None,
        split=None,
        tokenizer=None,
        **kwargs,
    ):
        self.vis_root = vis_root
        self.anno_path = anno_path
        self.annotation = None

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self.args = args
        self.split = split

        self.tokenizer = tokenizer

        self.data_type = None

    def __len__(self):
        return len(self.annotation)

    def collater(self, samples):
        return SampleCollator(self, samples)

    def set_processors(self, vis_processor, text_processor):
        self.vis_processor = vis_processor
        self.text_processor = text_processor


class RGBDBaseDataset(Dataset):
    def __init__(
        self,
        vis_processor=None,
        text_processor=None,
        vis_root=None,
        anno_path={},
        args=None,
        split=None,
        tokenizer=None,
        image_transform=None,
        **kwargs,
    ):
        self.vis_root = vis_root
        self.anno_path = anno_path
        self.annotation = None

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self.args = args
        self.split = split
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        
        if split == "train" and not self.args.use_text_branch:
            self.rgbd_text_features =None
        
        self.device = torch.device(args.device) if not args.pin_mem else None

    def __len__(self):
        return len(self.annotation)

    def collater(self, samples):
        return SampleCollator(self, samples)

    def set_processors(self, vis_processor, text_processor):
        self.vis_processor = vis_processor
        self.text_processor = text_processor

    def __getitem__(self, index):
        rtn = dict()
        ann = self.annotation[index]
        img_path = os.path.join(DEPTH_DATA_DIR, ann["image_path"])
        disp_path = os.path.join(DEPTH_DATA_DIR, ann["disparity_path"])
        rgb, depth = self.vis_processor(img_path=img_path, depth_path=disp_path, device=self.device)
        cleaned_label = ann["cleaned_label"]
        benchmark_label = ann["benchmark_label"] if "benchmark_label" in ann else None

        # todo: add caption: prompt template w/ label
        caption = self.text_processor(random.choice(SCENE_CLS_TEMPLATE)(cleaned_label))
        tokenized_caption = self.tokenizer([caption])[0]

        if self.args.use_openclip_transform:
            rgb = Image.open(img_path).convert("RGB")
            rgb = self.image_transform(rgb)
        
        if self.args.depth_channel==3:
            depth = depth.repeat(3, 1, 1)
        
        rtn = {
            "image": rgb,
            "depth": depth,
            "cleaned_label": self.text_processor(cleaned_label),
            "benchmark_label": self.text_processor(benchmark_label),
            "id": f"{index}_{img_path}",
            "label": self.label2idx[cleaned_label],
            "caption": tokenized_caption,
        }
        
        if self.split == "train" and not self.args.use_text_branch:
            template_idx = random.randint(0,18)
            text_feature = self.rgbd_text_features[cleaned_label][template_idx].to(dtype=torch.float16)
            rtn.update({"text_feature": text_feature})
        
        return Sample(rtn)


class SUNRGBDDataset(RGBDBaseDataset):
    def __init__(
        self,
        vis_processor=None,
        text_processor=None,
        vis_root=None,
        anno_path=dict(
            train=f"{DEPTH_META_DATA_DIR}/SUN-RGBD_train.json",
            val=f"{DEPTH_META_DATA_DIR}/SUN-RGBD_val.json",
            test=f"{DEPTH_META_DATA_DIR}/SUN-RGBD_val.json",
        ),
        args=None,
        split=None,
        tokenizer=None,
        image_transform=None,
        n_repeat_train=1,
        **kwargs,
    ):
        super().__init__(
            vis_processor,
            text_processor,
            vis_root,
            anno_path,
            args,
            split,
            tokenizer,
            image_transform,
            **kwargs,
        )

        self.annotation = json.load(open(anno_path[split], "r"))
        if split == "train" and n_repeat_train > 1:
            self.annotation = self.annotation * n_repeat_train

        self.init_labels()
        
        print_log(
            f"[SUN-RGBD-{split}]: Transform: {self.vis_processor}. # Samples: {len(self.annotation)}",'SUN-RGBD'
        )
        
        if split == "train" and not self.args.use_text_branch:
            self.rgbd_text_features = torch.load(args.rgbd_text_template_path)

    def init_labels(self):
        labelset = set()
        for ann in self.annotation:
            cleaned_label = ann["cleaned_label"]
            labelset.add(cleaned_label)

        self.idx2label = list(labelset)
        self.label2idx = {self.idx2label[i]: i for i in range(len(self.idx2label))}
        
        print_log(f"[SUN-RGBD] idx2label: {self.idx2label}.",'SUN-RGBD')
        print_log(f"[SUN-RGBD] label2idx: {self.label2idx}.",'SUN-RGBD')


class NYUDepthV2Dataset(RGBDBaseDataset):
    def __init__(
        self,
        vis_processor=None,
        text_processor=None,
        vis_root=None,
        anno_path=dict(
            train=f"{DEPTH_META_DATA_DIR}/NYU-Depth-v2_train.json",
            val1=f"{DEPTH_META_DATA_DIR}/NYU-Depth-v2_train.json",
            val2=f"{DEPTH_META_DATA_DIR}/NYU-Depth-v2_val.json",
        ),
        args=None,
        split=None,
        tokenizer=None,
        image_transform=None,
        **kwargs,
    ):
        super().__init__(
            vis_processor,
            text_processor,
            vis_root,
            anno_path,
            args,
            split,
            tokenizer,
            image_transform,
            **kwargs,
        )

        self.annotation = json.load(open(anno_path[split], "r"))
        self.init_labels()
        
        print_log(
            f"[NYU-Depthv2-{split}]: Transform: {self.vis_processor}. # Samples: {len(self.annotation)}",'NYU-Depth-v2'
        )

    def init_labels(self):
        labelset = set()
        map_to_others = set()
        for ann in self.annotation:
            cleaned_label = ann["cleaned_label"]
            benchmark_label = ann["benchmark_label"]
            labelset.add(cleaned_label)
            if cleaned_label != benchmark_label:
                map_to_others.add(cleaned_label)

        self.idx2label = list(labelset)
        self.label2idx = {self.idx2label[i]: i for i in range(len(self.idx2label))}

        self.other_idx = 100
        self.map_to_others = map_to_others
        self.map_to_others_idx = [
            self.label2idx[label] for label in self.label2idx if label in map_to_others
        ]
        
        print_log(f"[NYU-depth-v2] idx2label: {self.idx2label}.",'NYU-depth-v2')
        print_log(f"[NYU-depth-v2] label2idx: {self.label2idx}.",'NYU-depth-v2')
        print_log(f"[NYU-depth-v2] map_to_others: {map_to_others}.",'NYU-depth-v2')
        print_log(f"[NYU-depth-v2] map_to_others_idx: {self.map_to_others_idx}",'NYU-depth-v2')


name2dataset = {
    "sun-rgbd": SUNRGBDDataset,
    "nyu-depth-v2": NYUDepthV2Dataset,
    "nyu-depth-v2-val1": NYUDepthV2Dataset,
    "nyu-depth-v2-val2": NYUDepthV2Dataset,
}


def create_rgbd_dataset(args, is_train, tokenizer, image_transform=None):
    dataset_names = args.rgbd_train_data if is_train else args.rgbd_val_data
    dataset_names = dataset_names.split("::")

    datasets = dict()

    for name in dataset_names:
        dataset_cls = name2dataset[name]
        vis_proc_cls = RGBD_Processor_Train if is_train else RGBD_Processor_Eval
        vis_processor = vis_proc_cls.from_config()
        text_processor = BlipCaptionProcessor.from_config()

        split = None
        if name == "nyu-depth-v2-val1":
            split = "val1"
        elif name == "nyu-depth-v2-val2":
            split = "val2"
        else:
            if is_train:
                split = "train"
            else:
                split = "val"

        dataset = dataset_cls(
            vis_processor=vis_processor,
            text_processor=text_processor,
            args=args,
            split=split,
            tokenizer=tokenizer,
            image_transform=image_transform,
        )

        datasets[name] = dataset

    datasets_list = [v for _, v in datasets.items()]
    if len(datasets) > 1 and is_train:
        return concat_datasets(datasets_list)
    elif len(datasets) == 1 and is_train:
        return datasets_list[0]
    elif len(datasets) == 1 and not is_train:
        return datasets_list[0]
    else:
        return datasets
