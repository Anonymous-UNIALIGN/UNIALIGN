import os

import random
import h5py
import copy
import torch
import numpy as np

import torch.utils.data as data
# from open_clip.modal_3d.visual_imgs import ImageToAscii

import yaml
from easydict import EasyDict

from .io import IO

# from open_clip.util.logger import *
from datasets.Sample import Sample, SampleCollator
from datasets.util.build import DATASETS, build_dataset_from_cfg
from datasets.constants import OBJAVERSE_DATA_DIR

import json
from tqdm import tqdm
import pickle
from PIL import Image

from datasets.constants import PC_DATA_DIR, PC_META_DATA_DIR
import logging
from datasets.aug_random import np_random
from util.logger import print_log

pc_data_config = {
    "shapenet": {
        "config": f"{PC_META_DATA_DIR}/ShapeNet-55.yaml",
        "train": "train",
        "test": "test",
        "usage": "train",
    },
    "modelnet40": {
        "config": f"{PC_META_DATA_DIR}/ModelNet40.yaml",
        "train": "train",
        "test": "test",
        "usage": "test",
    },
    "objverse": {
        "config": f"{PC_META_DATA_DIR}/Objverse.yaml",
        "train": "train",
        "test": "test",
        "usage": "train",
    },
    "scanobjectnn": {
        "config": f"{PC_META_DATA_DIR}/ScanObjectNN.yaml",
        "test": "test",
        "usage": "test",
    },
}


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def normalize_pc(pc):
    # normalize pc to [-1, 1]
    pc = pc - np.mean(pc, axis=0)
    if np.max(np.linalg.norm(pc, axis=1)) < 1e-6:
        pc = np.zeros_like(pc)
    else:
        pc = pc / np.max(np.linalg.norm(pc, axis=1))
    return pc


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


def rotate_point_cloud(batch_data):
    """Randomly rotate the point clouds to augument the dataset
    rotation is per shape based along up direction
    Input:
      BxNx3 array, original batch of point clouds
    Return:
      BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array(
            [[cosval, 0, sinval], [0, 1, 0], [-sinval, 0, cosval]]
        )
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def rotate_point_cloud_distill(batch_data):
    """Randomly rotate the point clouds to augument the dataset
    rotation is per shape based along up direction
    Input:
      BxNx3 array, original batch of point clouds
    Return:
      BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np_random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array(
            [[cosval, 0, sinval], [0, 1, 0], [-sinval, 0, cosval]]
        )
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def random_point_dropout(batch_pc, max_dropout_ratio=0.875):
    """batch_pc: BxNx3"""
    for b in range(batch_pc.shape[0]):
        dropout_ratio = np.random.random() * max_dropout_ratio  # 0~0.875
        drop_idx = np.where(np.random.random((batch_pc.shape[1])) <= dropout_ratio)[0]
        if len(drop_idx) > 0:
            batch_pc[b, drop_idx, :] = batch_pc[b, 0, :]  # set to the first point
    return batch_pc


def random_point_dropout_distill(batch_pc, max_dropout_ratio=0.875):
    """batch_pc: BxNx3"""
    for b in range(batch_pc.shape[0]):
        dropout_ratio = np_random.rand() * max_dropout_ratio  # 0~0.875
        drop_idx = np.where(np_random.rand((batch_pc.shape[1])) <= dropout_ratio)[0]
        if len(drop_idx) > 0:
            batch_pc[b, drop_idx, :] = batch_pc[b, 0, :]  # set to the first point
    return batch_pc


def random_scale_point_cloud(batch_data, scale_low=0.8, scale_high=1.25):
    """Randomly scale the point cloud. Scale is per point cloud.
    Input:
        BxNx3 array, original batch of point clouds
    Return:
        BxNx3 array, scaled batch of point clouds
    """
    B, N, C = batch_data.shape
    scales = np.random.uniform(scale_low, scale_high, B)
    for batch_index in range(B):
        batch_data[batch_index, :, :] *= scales[batch_index]
    return batch_data


def random_scale_point_cloud_distill(batch_data, scale_low=0.8, scale_high=1.25):
    """Randomly scale the point cloud. Scale is per point cloud.
    Input:
        BxNx3 array, original batch of point clouds
    Return:
        BxNx3 array, scaled batch of point clouds
    """
    B, N, C = batch_data.shape
    scales = np_random.uniform(scale_low, scale_high, B)
    for batch_index in range(B):
        batch_data[batch_index, :, :] *= scales[batch_index]
    return batch_data


def shift_point_cloud(batch_data, shift_range=0.1):
    """Randomly shift point cloud. Shift is per point cloud.
    Input:
      BxNx3 array, original batch of point clouds
    Return:
      BxNx3 array, shifted batch of point clouds
    """
    B, N, C = batch_data.shape
    shifts = np.random.uniform(-shift_range, shift_range, (B, 3))
    for batch_index in range(B):
        batch_data[batch_index, :, :] += shifts[batch_index, :]
    return batch_data


def shift_point_cloud_distill(batch_data, shift_range=0.1):
    """Randomly shift point cloud. Shift is per point cloud.
    Input:
      BxNx3 array, original batch of point clouds
    Return:
      BxNx3 array, shifted batch of point clouds
    """
    B, N, C = batch_data.shape
    shifts = np_random.uniform(-shift_range, shift_range, (B, 3))
    for batch_index in range(B):
        batch_data[batch_index, :, :] += shifts[batch_index, :]
    return batch_data


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """Randomly jitter points. jittering is per point.
    Input:
      BxNx3 array, original batch of point clouds
    Return:
      BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert clip > 0
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1 * clip, clip)
    jittered_data += batch_data
    return jittered_data


def rotate_perturbation_point_cloud(batch_data, angle_sigma=0.06, angle_clip=0.18):
    """Randomly perturb the point clouds by small rotations
    Input:
      BxNx3 array, original batch of point clouds
    Return:
      BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        angles = np.clip(angle_sigma * np.random.randn(3), -angle_clip, angle_clip)
        Rx = np.array(
            [
                [1, 0, 0],
                [0, np.cos(angles[0]), -np.sin(angles[0])],
                [0, np.sin(angles[0]), np.cos(angles[0])],
            ]
        )
        Ry = np.array(
            [
                [np.cos(angles[1]), 0, np.sin(angles[1])],
                [0, 1, 0],
                [-np.sin(angles[1]), 0, np.cos(angles[1])],
            ]
        )
        Rz = np.array(
            [
                [np.cos(angles[2]), -np.sin(angles[2]), 0],
                [np.sin(angles[2]), np.cos(angles[2]), 0],
                [0, 0, 1],
            ]
        )
        R = np.dot(Rz, np.dot(Ry, Rx))
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), R)
    return rotated_data


def rotate_perturbation_point_cloud_distill(
    batch_data, angle_sigma=0.06, angle_clip=0.18
):
    """Randomly perturb the point clouds by small rotations
    Input:
      BxNx3 array, original batch of point clouds
    Return:
      BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        angles = np.clip(angle_sigma * np_random.randn(3), -angle_clip, angle_clip)
        Rx = np.array(
            [
                [1, 0, 0],
                [0, np.cos(angles[0]), -np.sin(angles[0])],
                [0, np.sin(angles[0]), np.cos(angles[0])],
            ]
        )
        Ry = np.array(
            [
                [np.cos(angles[1]), 0, np.sin(angles[1])],
                [0, 1, 0],
                [-np.sin(angles[1]), 0, np.cos(angles[1])],
            ]
        )
        Rz = np.array(
            [
                [np.cos(angles[2]), -np.sin(angles[2]), 0],
                [np.sin(angles[2]), np.cos(angles[2]), 0],
                [0, 0, 1],
            ]
        )
        R = np.dot(Rz, np.dot(Ry, Rx))
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), R)
    return rotated_data


import os, sys, h5py

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)


@DATASETS.register_module()
class ModelNet(data.Dataset):
    def __init__(self, config):
        self.root = os.path.join(PC_DATA_DIR, config.DATA_PATH)
        self.npoints = config.npoints
        self.use_normals = config.USE_NORMALS
        self.num_category = config.NUM_CATEGORY
        self.process_data = True
        self.uniform = True
        self.generate_from_raw_data = False
        split = config.subset
        self.subset = config.subset

        self.use_10k_pc = config.use_10k_pc
        self.use_colored_pc = config.with_color

        if self.num_category == 10:
            self.catfile = os.path.join(self.root, "modelnet10_shape_names.txt")
        else:
            self.catfile = os.path.join(self.root, "modelnet40_shape_names.txt")

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        shape_ids = {}
        if self.num_category == 10:
            shape_ids["train"] = [
                line.rstrip()
                for line in open(os.path.join(self.root, "modelnet10_train.txt"))
            ]
            shape_ids["test"] = [
                line.rstrip()
                for line in open(os.path.join(self.root, "modelnet10_test.txt"))
            ]
        else:
            shape_ids["train"] = [
                line.rstrip()
                for line in open(os.path.join(self.root, "modelnet40_train.txt"))
            ]
            shape_ids["test"] = [
                line.rstrip()
                for line in open(os.path.join(self.root, "modelnet40_test.txt"))
            ]

        assert split == "train" or split == "test"
        shape_names = ["_".join(x.split("_")[0:-1]) for x in shape_ids[split]]
        self.datapath = [
            (
                shape_names[i],
                os.path.join(self.root, shape_names[i], shape_ids[split][i]) + ".txt",
            )
            for i in range(len(shape_ids[split]))
        ]
        print_log("The size of %s data is %d" % (split, len(self.datapath)), "ModelNet")

        if self.uniform:
            self.save_path = os.path.join(
                self.root,
                "modelnet%d_%s_%dpts_fps.dat"
                % (self.num_category, split, self.npoints),
            )
        else:
            self.save_path = os.path.join(
                self.root,
                "modelnet%d_%s_%dpts.dat" % (self.num_category, split, self.npoints),
            )

        if self.process_data:
            if not os.path.exists(self.save_path):
                # make sure you have raw data in the path before you enable generate_from_raw_data=True.
                if self.generate_from_raw_data:
                    print_log(
                        "Processing data %s (only running in the first time)..."
                        % self.save_path,
                        "ModelNet",
                    )
                    self.list_of_points = [None] * len(self.datapath)
                    self.list_of_labels = [None] * len(self.datapath)

                    for index in tqdm(
                        range(len(self.datapath)), total=len(self.datapath)
                    ):
                        fn = self.datapath[index]
                        cls = self.classes[self.datapath[index][0]]
                        cls = np.array([cls]).astype(np.int32)
                        point_set = np.loadtxt(fn[1], delimiter=",").astype(np.float32)

                        if self.uniform:
                            point_set = farthest_point_sample(point_set, self.npoints)
                            print_log(
                                "uniformly sampled out {} points".format(self.npoints),
                                "ModelNet",
                            )
                        else:
                            point_set = point_set[0 : self.npoints, :]

                        self.list_of_points[index] = point_set
                        self.list_of_labels[index] = cls

                    with open(self.save_path, "wb") as f:
                        pickle.dump([self.list_of_points, self.list_of_labels], f)
                else:
                    # no pre-processed dataset found and no raw data found, then load 8192 points dataset then do fps after.
                    self.save_path = os.path.join(
                        self.root,
                        "modelnet%d_%s_%dpts_fps.dat"
                        % (self.num_category, split, 8192),
                    )
                    print_log(
                        "Load processed data from %s..." % self.save_path, "ModelNet"
                    )
                    if not self.use_10k_pc:
                        print_log(
                            "since no exact points pre-processed dataset found and no raw data found, load 8192 pointd dataset first, then do fps to {} after, the speed is excepted to be slower due to fps...".format(
                                self.npoints
                            ),
                            "ModelNet",
                        )
                    with open(self.save_path, "rb") as f:
                        self.list_of_points, self.list_of_labels = pickle.load(f)

            else:
                print_log("Load processed data from %s..." % self.save_path, "ModelNet")
                with open(self.save_path, "rb") as f:
                    self.list_of_points, self.list_of_labels = pickle.load(f)

        self.shape_names_addr = os.path.join(self.root, "modelnet40_shape_names.txt")
        with open(self.shape_names_addr) as file:
            lines = file.readlines()
            lines = [line.rstrip() for line in lines]
        self.shape_names = lines

        # TODO: disable for backbones except for PointNEXT!!!
        self.use_height = config.use_height

        if self.use_10k_pc and self.use_colored_pc:
            self.modelnet_10k_colored_pc_file = "/home/zhoubo/farm/open_clip/data/ModelNet/modelnet40_colored_10k_pc.npy"
            self.modelnet_10k_rgb_data = np.load(
                self.modelnet_10k_colored_pc_file, allow_pickle=True
            )
            with open(
                "/home/zhoubo/farm/open_clip/data/ModelNet/modelnet40_test_split_10k_colored.json",
                "r",
            ) as f:
                self.cat_name = json.load(f)

        if self.subset == "train":
            self.pc_text_features = torch.load(config.args.point_text_template_path)

    def __len__(self):
        return len(self.list_of_labels)

    def _get_item(self, index):
        if self.process_data:
            point_set, label = self.list_of_points[index], self.list_of_labels[index]
        else:
            fn = self.datapath[index]
            cls = self.classes[self.datapath[index][0]]
            label = np.array([cls]).astype(np.int32)
            point_set = np.loadtxt(fn[1], delimiter=",").astype(np.float32)

            if self.uniform:
                point_set = farthest_point_sample(point_set, self.npoints)
            else:
                point_set = point_set[0 : self.npoints, :]

        if self.npoints < point_set.shape[0]:
            point_set = farthest_point_sample(point_set, self.npoints)

        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        if not self.use_normals:
            point_set = point_set[:, 0:3]

        if self.use_height:
            self.gravity_dim = 1
            height_array = (
                point_set[:, self.gravity_dim : self.gravity_dim + 1]
                - point_set[:, self.gravity_dim : self.gravity_dim + 1].min()
            )
            point_set = np.concatenate((point_set, height_array), axis=1)

        if self.use_10k_pc and self.use_colored_pc:
            point_set = self.modelnet_10k_rgb_data[index]["xyz"]
            rgb_data = np.ones_like(point_set) * 0.4
            point_set = np.concatenate([point_set, rgb_data], axis=1)
            cat_name = self.cat_name[index]["category"]
            label = [self.shape_names.index(cat_name)]
        elif self.use_colored_pc:
            rgb_data = np.ones_like(point_set) * 0.4
            point_set = np.concatenate([point_set, rgb_data], axis=1)

        return point_set, label[0]

    def collater(self, samples):
        return SampleCollator(self, samples)

    def __getitem__(self, index):
        rtn = dict()

        points, label = self._get_item(index)
        pt_idxs = np.arange(0, points.shape[0])  # 2048
        if self.subset == "train":
            np.random.shuffle(pt_idxs)
        current_points = points[pt_idxs].copy()
        current_points = torch.from_numpy(current_points).float()
        label_name = self.shape_names[int(label)]

        rtn["pc"] = current_points
        rtn["label"] = label
        rtn["class_name"] = label_name

        if self.subset == "train":
            template_idx = random.randint(0, 59)
            text_feature = self.pc_text_features[label][template_idx].to(
                dtype=torch.float16
            )

            rtn["text_feature"] = text_feature

        return Sample(rtn)


# @DATASETS.register_module()
# class Objverse(data.Dataset):
#     def __init__(self, config):
#         self.ROOT_DIR = (
#             os.path.join(PC_DATA_DIR, config.DATA_PATH)
#             if config is not None
#             else OBJAVERSE_DATA_DIR
#         )
#         self.BUCKETS = os.listdir(self.ROOT_DIR)
#         self.BUCKETS.sort(key=lambda x: int(x.split(".")[0].split("_")[1]))
#         self.data_manual = {}

#         for bucket in self.BUCKETS:
#             env = lmdb.open(
#                 os.path.join(self.ROOT_DIR, bucket), readonly=True, lock=False
#             )
#             with env.begin() as txn:
#                 self.data_manual[bucket] = txn.stat()["entries"]

#         # calculate dataset entries
#         self.len = sum(self.data_manual.values())
#         self.dbs = [
#             lmdb.open(os.path.join(self.ROOT_DIR, bucket), readonly=True, lock=False)
#             for bucket in self.BUCKETS
#         ]
#         self.cumulative_bucket_scale = np.cumsum(list(self.data_manual.values()))

#         # print some info
#         print_log(
#             f"<Objverse>: Hi, I have discovered {sum(self.data_manual.values())} entries from {len(self.BUCKETS)} buckets."
#         )

#         # self.tokenizer = (
#         #     config.tokenizer if config is not None else get_tokenizer("ViT-B-16")
#         # )
#         self.train_transform = (
#             config.train_transform
#             if config is not None
#             else image_transform(image_size=224, is_train=True, mean=0, std=1)
#         )
#         self.augment = True

#     def __len__(self):
#         return self.len

#     def pc_norm(self, pc):
#         """pc: NxC, return NxC"""
#         centroid = np.mean(pc, axis=0)
#         pc = pc - centroid
#         m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
#         pc = pc / m
#         return pc

#     def find_bucket(self, id, bucket_scale):
#         bid = 0
#         for scale in bucket_scale:
#             if id >= scale:
#                 bid += 1
#             else:
#                 return bid

#     def __getitem__(self, index):
#         bucket_idx = self.find_bucket(index, self.cumulative_bucket_scale)
#         initial_idx = (
#             0 if bucket_idx == 0 else self.cumulative_bucket_scale[bucket_idx - 1]
#         )
#         with self.dbs[bucket_idx].begin() as txn:
#             # load the data
#             try:
#                 pc, imgs, texts = pickle.loads(
#                     txn.get(str(index - initial_idx).encode("ascii"))
#                 )
#             except:
#                 print(
#                     f"Error loading {index} as {index - initial_idx} from bucket {bucket_idx}"
#                 )

#             pc = self.pc_norm(pc)

#             if self.augment:
#                 pc = random_point_dropout(pc[None, ...])
#                 pc = random_scale_point_cloud(pc)
#                 pc = shift_point_cloud(pc)
#                 pc = rotate_perturbation_point_cloud(pc)
#                 pc = rotate_point_cloud(pc)
#                 pc = pc.squeeze()

#             pc = torch.from_numpy(pc)
#             # load images from bytes to tensors
#             img_idx = np.random.randint(0, len(imgs))
#             img = Image.open(io.BytesIO(imgs[img_idx]))
#             img = self.train_transform(img)

#             # tokenize the captions
#             textlist = texts[img_idx]
#             # caption = np.random.choice(textlist)
#             # tokenized_caption = self.tokenizer([caption])[0]
#             tokenized_caption=None

#             return Sample({"pc": pc, "image": img, "caption": tokenized_caption})


@DATASETS.register_module()
class Objaverse_Lvis_Colored(data.Dataset):
    def __init__(self, config):
        self.npoints = 10000
        self.tokenizer = config.tokenizer
        self.train_transform = config.train_transform

        self.lvis_list_addr = "data/objaverse-lvis/ulip-2/objaverse-lvis/lvis.json"
        self.lvis_metadata_addr = (
            "data/objaverse-lvis/ulip-2/objaverse-lvis/objaverse_lvis_metadata.json"
        )

        with open(self.lvis_list_addr, "r") as f:
            self.npy_file_map = json.load(f)

        self.file_list = list(self.npy_file_map.keys())

        with open(self.lvis_metadata_addr, "r") as f:
            self.lvis_metadata = json.load(f)

        self.prompt_template_addr = "src/open_clip/modal_3d/data/templates.json"
        with open(self.prompt_template_addr) as f:
            self.templates = json.load(f)[config.pretrain_dataset_prompt]

        self.sample_points_num = self.npoints

        print_log(
            f"Objaverse lvis {len(self.file_list)} instances were loaded",
            "Objaverse_Lvis_Colored",
        )

        self.permutation = np.arange(self.npoints)

        # =================================================
        # TODO: disable for backbones except for PointNEXT!!!
        self.use_height = False
        self.use_color = True

        self.objaverse_lvis_path = "data/objaverse-lvis"

        if self.use_color:
            print("use color")
        else:
            print("don't use color")

        self.augment = True

        self.subset = config.subset
        if self.subset == "train" and config.args.text_embed_dim == 1536:
            self.pc_text_features = torch.load(config.args.point_text_template_path)

    def pc_norm(self, pc):
        """pc: NxC, return NxC"""
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc

    def random_sample(self, pc, num):
        np.random.shuffle(self.permutation)
        pc = pc[self.permutation[:num]]
        return pc

    def collater(self, samples):
        return SampleCollator(self, samples)

    def __getitem__(self, idx):
        sample = self.file_list[idx]
        pc_addr = self.npy_file_map[sample]
        pc_addr = os.path.join(self.objaverse_lvis_path, self.npy_file_map[sample])
        data = np.load(pc_addr, allow_pickle=True)
        dict_data = data.item()
        xyz_data = dict_data["xyz"]
        rgb_data = dict_data["rgb"]

        rgb_data = self.train_transform(rgb_data)

        data = self.pc_norm(xyz_data)

        if self.augment:
            pc = random_point_dropout(data[None, ...])
            pc = random_scale_point_cloud(pc)
            pc = shift_point_cloud(pc)
            pc = rotate_perturbation_point_cloud(pc)
            pc = rotate_point_cloud(pc)
            pc = pc.squeeze()

        # if self.use_color:
        #     data = np.concatenate([data, rgb_data], axis=1)

        if self.use_height:
            self.gravity_dim = 1
            height_array = (
                data[:, self.gravity_dim : self.gravity_dim + 1]
                - data[:, self.gravity_dim : self.gravity_dim + 1].min()
            )
            data = np.concatenate((data, height_array), axis=1)
            data = torch.from_numpy(data).float()
        else:
            data = torch.from_numpy(data).float()

        data = data.contiguous()

        name = self.lvis_metadata["value_to_key_mapping"][sample]
        label = self.lvis_metadata["key_to_id"][name]

        text_feature = None
        if self.subset == "train":
            template_idx = random.randint(0, 59)
            if isinstance(self.pc_text_features[label], dict):
                text_feature = random.choice(self.pc_text_features[label])[
                    template_idx
                ].to(dtype=torch.float16)

            else:
                text_feature = self.pc_text_features[label][template_idx].to(
                    dtype=torch.float16
                )

        tokenized_caption = self.tokenizer([name])[0]

        return Sample(
            {
                "pc": data,
                "image": rgb_data,
                "caption": tokenized_caption,
                "label": label,
                "text_feature": text_feature,
            }
        )
        # return data, label, name

    def __len__(self):
        return len(self.file_list)


@DATASETS.register_module()
class ScanObjectNN(data.Dataset):
    def __init__(self, config):
        self.data_root = os.path.join(PC_DATA_DIR, config.DATA_PATH)
        self.subset = config.subset
        self.npoints = config.npoints
        self.tokenizer = config.tokenizer
        self.train_transform = config.train_transform

        self.test_set_name = "test_objectdataset.h5"
        self.splits = [
            "main_split_nobg",
            # "split1_nobg",
            # "split2_nobg",
            # "split3_nobg",
            # "split4_nobg",
        ]

        self.data = []
        self.label = []

        for split in self.splits:
            # fetch the h5 files
            test_h5 = h5py.File(
                os.path.join(self.data_root, split, self.test_set_name), "r"
            )

            # print some info
            print_log(
                f"<ScanObjectNN>: Hi, I have discovered {len(test_h5['data'])} entries from {self.data_root}/{split}.",
                "ScanObjectNN",
            )
            data = test_h5["data"][:]
            label = test_h5["label"][:]

            self.data.append(data)
            self.label.append(label)

        # cat the nd arrays
        self.data = np.concatenate(self.data, axis=0)
        self.label = np.concatenate(self.label, axis=0)

        self.semantic_classes = [
            "bag",
            "bin",
            "box",
            "cabinet",
            "chair",
            "desk",
            "display",
            "door",
            "shelf",
            "table",
            "bed",
            "pillow",
            "sink",
            "sofa",
            "toilet",
        ]
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        
        pc = self.data[item]
        label = self.label[item]

        pc = torch.from_numpy(pc)
        caption = self.tokenizer(self.semantic_classes[label])
        

        # TODO: Method to upsample the point cloud to 8192 points

        # pc = pc_normalize(pc)

        return Sample({"pc": pc, "caption": caption, "label": label})


@DATASETS.register_module()
class ShapeNet(data.Dataset):
    def __init__(self, config):
        self.data_root = os.path.join(PC_DATA_DIR, config.DATA_PATH)
        self.pc_path = config.PC_PATH
        self.subset = config.subset
        self.npoints = config.npoints
        self.tokenizer = config.tokenizer
        self.train_transform = config.train_transform
        self.id_map_addr = os.path.join(self.data_root, "taxonomy.json")
        self.rendered_image_addr = os.path.join(self.data_root, config.IMAGE_PATH)
        self.picked_image_type = ["", "_depth0001"]
        self.picked_rotation_degrees = list(range(0, 360, 12))
        self.picked_rotation_degrees = [
            (3 - len(str(degree))) * "0" + str(degree)
            if len(str(degree)) < 3
            else str(degree)
            for degree in self.picked_rotation_degrees
        ]

        with open(self.id_map_addr, "r") as f:
            self.id_map = json.load(f)

        self.prompt_template_addr = f"{PC_META_DATA_DIR}/templates.json"
        with open(self.prompt_template_addr) as f:
            self.templates = json.load(f)[config.train_data_prompt]

        self.synset_id_map = {}
        self.index_list = {}
        for i, id_dict in enumerate(self.id_map):
            synset_id = id_dict["synsetId"]
            self.synset_id_map[synset_id] = id_dict
            self.index_list[synset_id] = i

        self.data_list_file = os.path.join(self.data_root, f"{self.subset}.txt")
        test_data_list_file = os.path.join(self.data_root, "test.txt")

        self.sample_points_num = self.npoints
        self.whole = config.get("whole")

        print_log(f"[DATASET] sample out {self.sample_points_num} points", "ShapeNet")
        print_log(f"[DATASET] Open file {self.data_list_file}", "ShapeNet")
        with open(self.data_list_file, "r") as f:
            lines = f.readlines()
        if self.whole:
            with open(test_data_list_file, "r") as f:
                test_lines = f.readlines()
            print_log(f"[DATASET] Open file {test_data_list_file}", "ShapeNet")
            lines = test_lines + lines
        self.file_list = []
        for line in lines:
            line = line.strip()
            taxonomy_id = line.split("-")[0]
            model_id = line[len(taxonomy_id) + 1 :].split(".")[0]
            self.file_list.append(
                {"taxonomy_id": taxonomy_id, "model_id": model_id, "file_path": line}
            )
        print_log(f"[DATASET] {len(self.file_list)} instances were loaded", "ShapeNet")

        self.permutation = np.arange(self.npoints)

        self.uniform = True
        self.augment = True
        self.use_caption_templates = False
        # =================================================
        # TODO: disable for backbones except for PointNEXT!!!
        self.use_height = config.use_height
        # =================================================

        if self.augment:
            print("using augmented point clouds.")

        if self.subset == "train" and config.args.text_embed_dim == 1536:
            self.pc_text_features_path = "/home/zhoubo/farm/open_clip/src/open_clip_train/text_features/Point/shapenet_ulip"

        self.distill = False
        if self.subset == "train" and config.args.multi_modal_distill:
            self.distill = True

        self.with_color = config.with_color
        self.config = config
        
        self.device = torch.device(config.args.device) if not config.args.pin_mem else None

    def pc_norm(self, pc):
        """pc: NxC, return NxC"""
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc

    def random_sample(self, pc, num):
        if self.distill:
            np_random.shuffle(self.permutation)
        else:
            np.random.shuffle(self.permutation)
        pc = pc[self.permutation[:num]]
        return pc

    def collater(self, samples):
        return SampleCollator(self, samples)

    def __getitem__(self, idx):
        rtn = None
        while rtn is None:
            sample = self.file_list[idx]

            data = IO.get(
                os.path.join(self.data_root, self.pc_path, sample["file_path"])
            ).astype(np.float32)

            if self.uniform and self.sample_points_num < data.shape[0]:
                data = farthest_point_sample(data, self.sample_points_num)
            else:
                data = self.random_sample(data, self.sample_points_num)
            data = self.pc_norm(data)

            if self.augment:
                if self.distill:
                    data = random_point_dropout_distill(data[None, ...])
                    data = random_scale_point_cloud_distill(data)
                    data = shift_point_cloud_distill(data)
                    data = rotate_perturbation_point_cloud_distill(data)
                    data = rotate_point_cloud(data)
                    data = data.squeeze()
                else:
                    data = random_point_dropout(data[None, ...])
                    data = random_scale_point_cloud(data)
                    data = shift_point_cloud(data)
                    data = rotate_perturbation_point_cloud(data)
                    data = rotate_point_cloud(data)
                    data = data.squeeze()

            if self.with_color:
                color = np.ones(data.shape) * 0.4
                data = np.concatenate([data, color], axis=-1)

            if self.use_height:
                self.gravity_dim = 1
                height_array = (
                    data[:, self.gravity_dim : self.gravity_dim + 1]
                    - data[:, self.gravity_dim : self.gravity_dim + 1].min()
                )
                data = np.concatenate((data, height_array), axis=1)
                data = torch.from_numpy(data).float()
            else:
                data = torch.from_numpy(data).float()

            index = self.index_list[sample["taxonomy_id"]]
            label = torch.tensor(index)

            captions = self.synset_id_map[sample["taxonomy_id"]]["name"]
            captions = [
                caption.strip() for caption in captions.split(",") if caption.strip()
            ]
            caption = random.choice(captions)

            template = random.choice(self.templates)
            caption = template.format(caption)
            tokenized_caption = self.tokenizer([caption])[0]

            
            picked_model_rendered_image_addr = (
                self.rendered_image_addr + "/"
                
            )
            picked_image_name = (
                sample["taxonomy_id"]
                + "-"
                + sample["model_id"]
                + "_r_"
                + str(random.choice(self.picked_rotation_degrees))
                + random.choice(self.picked_image_type)
                + ".png"
            )
            picked_image_addr = picked_model_rendered_image_addr + picked_image_name

            text_feature = None
            if self.subset == "train" and self.config.args.text_embed_dim == 1536:
                npy_text_features = np.load(
                    self.pc_text_features_path + "/" + sample["taxonomy_id"] + ".npy"
                )

                template_idx = random.randint(0, npy_text_features.shape[0] - 1)
                text_feature = torch.from_numpy(npy_text_features[template_idx]).to(
                    dtype=torch.float16
                )

            try:
                image = pil_loader(picked_image_addr)
                image = self.train_transform(image)
                rtn = Sample(
                    {
                        "taxonomy_id": sample["taxonomy_id"],
                        "model_id": sample["model_id"],
                        "caption": tokenized_caption,
                        "pc": data,
                        "image": image,
                        "target": label,
                        "text_feature": text_feature,
                    }
                )
            except:
                print_log(
                    "image is corrupted: {}".format(picked_image_addr), "ShapeNet"
                )
                idx = random.randint(0, len(self.file_list) - 1)
        return rtn

    def __len__(self):
        return len(self.file_list)


import collections.abc as container_abcs

int_classes = int
string_classes = str

import re

default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}"
)
np_str_obj_array_pattern = re.compile(r"[SaUO]")


def customized_collate_fn(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""
    elem = batch[0]
    elem_type = type(elem)

    if isinstance(batch, list):
        batch = [example for example in batch if example[4] is not None]

    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif (
        elem_type.__module__ == "numpy"
        and elem_type.__name__ != "str_"
        and elem_type.__name__ != "string_"
    ):
        if elem_type.__name__ == "ndarray" or elem_type.__name__ == "memmap":
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return customized_collate_fn([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        return {key: customized_collate_fn([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
        return elem_type(*(customized_collate_fn(samples) for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError("each element in list of batch should be of equal size")
        transposed = zip(*batch)
        return [customized_collate_fn(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))


def merge_new_config(config, new_config):
    for key, val in new_config.items():
        if not isinstance(val, dict):
            if key == "_base_":
                with open(new_config["_base_"], "r") as f:
                    try:
                        val = yaml.load(f, Loader=yaml.FullLoader)
                    except:
                        val = yaml.load(f)
                config[key] = EasyDict()
                merge_new_config(config[key], val)
            else:
                config[key] = val
                continue
        if key not in config:
            config[key] = EasyDict()
        merge_new_config(config[key], val)
    return config


def cfg_from_yaml_file(cfg_file):
    config = EasyDict()
    with open(cfg_file, "r") as f:
        new_config = yaml.load(f, Loader=yaml.FullLoader)
    merge_new_config(config=config, new_config=new_config)
    return config


class Dataset_3D:
    def __init__(self, args, tokenizer, dataset_name, train_transform=None):
        
        self.dataset_catalog = pc_data_config
        self.dataset_usage = self.dataset_catalog[self.dataset_name]["usage"]
        self.dataset_split = self.dataset_catalog[self.dataset_name][self.dataset_usage]
        self.dataset_config_dir = self.dataset_catalog[self.dataset_name]["config"]
        self.tokenizer = tokenizer
        self.train_transform = train_transform
        self.train_data_prompt = args.point_train_data_prompt
        self.val_data_prompt = args.point_val_data_prompt
        if args.pc_dataset_n_points == 10000:
            self.use_10k_pc = True
        else:
            self.use_10k_pc = False
        self.build_3d_dataset(args, self.dataset_config_dir)

    def build_3d_dataset(self, args, config):
        config = cfg_from_yaml_file(config)
        config.tokenizer = self.tokenizer
        config.train_transform = self.train_transform
        config.train_data_prompt = self.train_data_prompt
        config.val_data_prompt = self.val_data_prompt
        config.args = args
        config.use_height = False  
        config.npoints = args.pc_n_points
        config.with_color = True if args.pc_in_channel == 6 else False
        config.use_10k_pc = self.use_10k_pc
        config_others = EasyDict({"subset": self.dataset_split, "whole": True})
        self.dataset = build_dataset_from_cfg(config, config_others)

    def __len__(self):
        return len(self.dataset)
