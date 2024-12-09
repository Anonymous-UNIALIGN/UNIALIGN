import os
from pydoc import text
from typing import Iterable, List, Optional
import multiprocessing
import torch.distributed as dist
import numpy as np
from datasets.aug_random import AugRandomContext
from datasets.manager import TxtManager
from torch.utils.data import Dataset
import bisect
import warnings
import random
import itertools
import torch
import math
from datasets.Sample import Sample

def get_rank():
    if not dist.is_available() or not dist.is_initialized():
        return 0
    return dist.get_rank()


class DatasetWrapper(Dataset):
    def __init__(self, dataset, logits_path, topk, write, logits_name=None):
        super().__init__()
        self.dataset = dataset
        self.logits_path = logits_path
        self.logits_name = logits_name
        self.epoch = multiprocessing.Value("i", 0)
        self.topk = topk
        self.write_mode = write
        self.keys = self._get_keys()
        self._manager = (None, None)

    def __getitem__(self, index: int):
        if self.write_mode:
            return self.__getitem_for_write(index)
        return self.__getitem_for_read(index)

    def __getitem_for_write(self, index: int):
        # get an augmentation seed
        key = self.keys[index]
        seed = np.int32(np.random.randint(0, 1 << 31))
        with AugRandomContext(seed=int(seed)):
            item = self.dataset[index]
        return (item, (key, seed))

    def __getitem_for_read(self, index: int):
        try:
            key = self.keys[index]
        except IndexError:
            print(index)
            exit()
        seed, logits_value = self._get_saved_logits(key)
        with AugRandomContext(seed=seed):
            item = self.dataset[index]
        return (item, (logits_value, np.int32(seed)))

    def _get_saved_logits(self, key: str):
        manager = self.get_manager()
        bstr: bytes = manager.read(key)
        # parse the augmentation seed
        seed = int(np.frombuffer(bstr[:4], dtype=np.int32))
        # parse the logits index and value
        # copy logits_index and logits_value to avoid warning of written flag from PyTorch
        bstr = bstr[4:]
        logits_value = np.frombuffer(bstr[: self.topk * 2], dtype=np.float16).copy()
        return seed, logits_value

    def _build_manager(self, logits_path: str):
        # topk * [idx, value] * 2 bytes  for logits + 4 bytes for seed
        item_size = self.topk * 2 + 4
        rank = get_rank()
        return TxtManager(logits_path, item_size, rank)

    def set_epoch(self, epoch: int):
        self.epoch.value = epoch
        self._manager = (None, None)

    def get_manager(self):
        epoch = self.epoch.value
        if epoch != self._manager[0]:
            if self.logits_name is None:
                # logits_path = os.path.join(
                #     self.logits_path, f"logits_top{self.topk}_epoch{self.epoch.value}"
                # )
                logits_path = os.path.join(self.logits_path, f"epoch{self.epoch.value}")
                self._manager = (epoch, self._build_manager(logits_path))
            else:
                logits_path = os.path.join(self.logits_path, self.logits_name)
                self._manager = (epoch, self._build_manager(logits_path))
        return self._manager[1]

    def __len__(self):
        return len(self.dataset)

    def _get_keys(self):
        if hasattr(self.dataset, "get_keys"):
            keys = self.dataset.get_keys()
            if self.write_mode:
                # we only check key unique in the write mode
                assert len(keys) == len(set(keys)), "keys must be unique"
            return keys
        return [str(i) for i in range(len(self))]

class DatasetSampleWrapper(Dataset):
    def __init__(self, dataset, logits_path, topk, write, logits_name=None):
        super().__init__()
        self.dataset = dataset
        self.logits_path = logits_path
        self.logits_name = logits_name
        self.epoch = multiprocessing.Value("i", 0)
        self.topk = topk
        self.write_mode = write
        self.keys = self._get_keys()
        self._manager = (None, None)

    def __getitem__(self, index: int):
        if self.write_mode:
            return self.__getitem_for_write(index)
        return self.__getitem_for_read(index)

    def __getitem_for_write(self, index: int):
        # get an augmentation seed
        key = self.keys[index]
        seed = np.int32(np.random.randint(0, 1 << 31))
        with AugRandomContext(seed=int(seed)):
            item = self.dataset[index]
        
        rtn = {**item, 'seed': seed, 'key': key}
        
        return Sample(rtn)
    def __getitem_for_read(self, index: int):
        try:
            key = self.keys[index]
        except IndexError:
            print(index)
            exit()
        seed, logits_value = self._get_saved_logits(key)
        with AugRandomContext(seed=seed):
            item = self.dataset[index]
        
        logits_value = torch.from_numpy(logits_value)
        
        rtn = {**item, 'seed': np.int32(seed), 'key': logits_value}
        
        return Sample(rtn)

    def _get_saved_logits(self, key: str):
        manager = self.get_manager()
        bstr: bytes = manager.read(key)
        # parse the augmentation seed
        seed = int(np.frombuffer(bstr[:4], dtype=np.int32))
        # parse the logits index and value
        # copy logits_index and logits_value to avoid warning of written flag from PyTorch
        bstr = bstr[4:]
        logits_value = np.frombuffer(bstr[: self.topk * 2], dtype=np.float16).copy()
        return seed, logits_value

    def _build_manager(self, logits_path: str):
        # topk * [idx, value] * 2 bytes  for logits + 4 bytes for seed
        item_size = self.topk * 2 + 4
        rank = get_rank()
        return TxtManager(logits_path, item_size, rank)

    def set_epoch(self, epoch: int):
        self.epoch.value = epoch
        self._manager = (None, None)

    def get_manager(self):
        epoch = self.epoch.value
        if epoch != self._manager[0]:
            if self.logits_name is None:
                # logits_path = os.path.join(
                #     self.logits_path, f"logits_top{self.topk}_epoch{self.epoch.value}"
                # )
                logits_path = os.path.join(self.logits_path, f"epoch{self.epoch.value}")
                self._manager = (epoch, self._build_manager(logits_path))
            else:
                logits_path = os.path.join(self.logits_path, self.logits_name)
                self._manager = (epoch, self._build_manager(logits_path))
        return self._manager[1]

    def __len__(self):
        return len(self.dataset)

    def _get_keys(self):
        if hasattr(self.dataset, "get_keys"):
            keys = self.dataset.get_keys()
            if self.write_mode:
                # we only check key unique in the write mode
                assert len(keys) == len(set(keys)), "keys must be unique"
            return keys
        return [str(i) for i in range(len(self))]


class SoftmaxlogitsDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, logits_path, topk, write):
        super().__init__()
        self.dataset = dataset
        self.logits_path = logits_path
        self.epoch = multiprocessing.Value("i", 0)
        self.topk = topk
        self.write_mode = write
        self.keys = self._get_keys()
        self._manager = (None, None)

    def __getitem__(self, index: int):
        if self.write_mode:
            return self.__getitem_for_write(index)
        return self.__getitem_for_read(index)

    def __getitem_for_write(self, index: int):
        # get an augmentation seed
        key = self.keys[index]
        seed = np.int32(np.random.randint(0, 1 << 31))
        with AugRandomContext(seed=int(seed)):
            item = self.dataset[index]
        return (item, (key, seed))

    def __getitem_for_read(self, index: int):
        key = self.keys[index]
        seed, logits_index, logits_value = self._get_saved_logits(key)
        with AugRandomContext(seed=seed):
            item = self.dataset[index]
        return (item, (logits_index, logits_value, np.int32(seed)))

    def _get_saved_logits(self, key: str):
        manager = self.get_manager()
        bstr: bytes = manager.read(key)
        # parse the augmentation seed
        seed = int(np.frombuffer(bstr[:4], dtype=np.int32))
        # parse the logits index and value
        # copy logits_index and logits_value to avoid warning of written flag from PyTorch
        bstr = bstr[4:]
        logits_index = np.frombuffer(bstr[: self.topk * 2], dtype=np.int16).copy()
        bstr = bstr[self.topk * 2 :]
        logits_value = np.frombuffer(bstr[: self.topk * 2], dtype=np.float16).copy()
        return seed, logits_index, logits_value

    def _build_manager(self, logits_path: str):
        # topk * [idx, value] * 2 bytes  for logits + 4 bytes for seed
        item_size = self.topk * 2 * 2 + 4
        rank = get_rank()
        return TxtManager(logits_path, item_size, rank)

    def set_epoch(self, epoch: int):
        self.epoch.value = epoch
        self._manager = (None, None)

    def get_manager(self):
        epoch = self.epoch.value
        if epoch != self._manager[0]:
            logits_path = os.path.join(
                self.logits_path, f"logits_top{self.topk}_epoch{self.epoch.value}"
            )
            self._manager = (epoch, self._build_manager(logits_path))
        return self._manager[1]

    def __len__(self):
        return len(self.dataset)

    def _get_keys(self):
        if hasattr(self.dataset, "get_keys"):
            keys = self.dataset.get_keys()
            if self.write_mode:
                # we only check key unique in the write mode
                assert len(keys) == len(set(keys)), "keys must be unique"
            return keys
        return [str(i) for i in range(len(self))]


class TriDistillationDatasetWrapper(Dataset):
    def __init__(self, dataset, logits_path, topk, write, logits_name=None, text_logits_name=None, image_logits_name=None):
        super().__init__()
        self.dataset = dataset
        self.logits_path = logits_path
        self.logits_name = logits_name
        self.text_logits_name = text_logits_name
        self.image_logits_name = image_logits_name
        self.epoch = multiprocessing.Value("i", 0)
        self.topk = topk
        self.write_mode = write
        self.keys = self._get_keys()
        
        self.num_manager = 1
        if text_logits_name is not None:
            self.num_manager += 1
        
        if image_logits_name is not None:
            self.num_manager += 1
        
        self._manager = (None, {})

    def __getitem__(self, index: int):
        if self.write_mode:
            return self.__getitem_for_write(index)
        return self.__getitem_for_read(index)

    def __getitem_for_write(self, index: int):
        # get an augmentation seed
        key = self.keys[index]
        # print(key)
        seed = np.int32(np.random.randint(0, 1 << 31))
        with AugRandomContext(seed=int(seed)):
            item = self.dataset[index]
        
        return item + (key, seed)

    def __getitem_for_read(self, index: int):
        try:
            key = self.keys[index]

            # print(f'index: {index} key: {key}')
        except IndexError:
            print(index)
            exit()
        seed, value_dcit = self._get_saved_logits(key)
        with AugRandomContext(seed=seed):
            item = self.dataset[index]
        
        rtn = {**item, 'seed': np.int32(seed), **value_dcit}
        
        return Sample(rtn)


    def _get_saved_logits(self, key: str):
        manager_dict = self.get_manager()
        value_dcit = {}
        
        bstr: bytes = manager_dict['logits'].read(key)
        # parse the augmentation seed
        seed = int(np.frombuffer(bstr[:4], dtype=np.int32))
        
        # parse the logits index and value
        # copy logits_index and logits_value to avoid warning of written flag from PyTorch
        bstr = bstr[4:]
        logits_value = np.frombuffer(bstr[: self.topk * 2], dtype=np.float16).copy()
        
        value_dcit['key'] = torch.from_numpy(logits_value)
        
        if self.text_logits_name is not None:
            text_bstr: bytes = manager_dict['text_logits'].read(key)
            seed_text = int(np.frombuffer(text_bstr[:4], dtype=np.int32))
            assert seed == seed_text 
            
            text_bstr = text_bstr[4:]
            text_logits_value = np.frombuffer(text_bstr[: self.topk * 2], dtype=np.float16).copy()
            value_dcit['key_text'] = torch.from_numpy(text_logits_value)
            
        if self.image_logits_name is not None:
            img_bstr: bytes = manager_dict['image_logits'].read(key)
            seed_img = int(np.frombuffer(img_bstr[:4], dtype=np.int32))
            assert seed == seed_img
            img_bstr = img_bstr[4:]
            image_logits_value = np.frombuffer(img_bstr[: self.topk * 2], dtype=np.float16).copy()
            value_dcit['key_image'] = torch.from_numpy(image_logits_value)
        
        return seed, value_dcit

    def _build_manager(self, logits_path: str):
        # topk * [idx, value] * 2 bytes  for logits + 4 bytes for seed
        item_size = self.topk * 2 + 4
        rank = get_rank()
        return TxtManager(logits_path, item_size, rank)

    def set_epoch(self, epoch: int):
        self.epoch.value = epoch
        self._manager = (None, {})

    def get_manager(self):
        epoch = self.epoch.value
        if epoch != self._manager[0]:
            # if self.logits_name is None:
            #     # logits_path = os.path.join(
            #     #     self.logits_path, f"logits_top{self.topk}_epoch{self.epoch.value}"
            #     # )
            #     logits_path = os.path.join(self.logits_path, f"epoch{self.epoch.value}")
            #     self._manager = (epoch, self._build_manager(logits_path))
            # else:
            multi_manger = {}
            if self.logits_name is not None:
                
                logits_path = os.path.join(self.logits_path, self.logits_name)
                multi_manger['logits'] = self._build_manager(logits_path)
                
            if self.text_logits_name is not None:
                
                text_logits_path = os.path.join(self.logits_path, self.text_logits_name)
                multi_manger['text_logits'] = self._build_manager(text_logits_path)
                
            if self.image_logits_name is not None:
                
                image_logits_path = os.path.join(self.logits_path, self.image_logits_name)
                multi_manger['image_logits'] = self._build_manager(image_logits_path)
            
            self._manager = (epoch, multi_manger)
                
            # self._manager = (epoch, self._build_manager(logits_path), self._build_manager(text_logits_path), self._build_manager(image_logits_path))
                
        return self._manager[1]

    def __len__(self):
        return len(self.dataset)

    def _get_keys(self):
        # if hasattr(self.dataset, "get_keys"):
        #     keys = self.dataset.get_keys()
        #     if self.write_mode:
        #         # we only check key unique in the write mode
        #         assert len(keys) == len(set(keys)), "keys must be unique"
        #     return keys
        return [str(i) for i in range(len(self))]


class ConcatDataset(Dataset):
    datasets: List[Dataset]
    modality_list: List[str]
    cumulative_sizes: List[int]

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0

        for _, e in enumerate(sequence):
            length = len(e)

            r.append(length + s)
            s += length
        return r

    def __init__(
        self,
        datasets: Iterable[Dataset],
        modality_list,
    ) -> None:
        super().__init__()
        self.datasets = list(datasets)
        assert len(self.datasets) > 0, "datasets should not be an empty iterable"  # type: ignore[arg-type]
        # for d in self.datasets:
        # assert not isinstance(d, IterableDataset), "ConcatDataset does not support IterableDataset"
        self.cumulative_sizes = self.cumsum(self.datasets)
        self.modality_list = modality_list

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    "absolute value of index should not exceed dataset length"
                )
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        modal = self.modality_list[dataset_idx]

        return self.datasets[dataset_idx][sample_idx], modal

    @property
    def cummulative_sizes(self):
        warnings.warn(
            "cummulative_sizes attribute is renamed to " "cumulative_sizes",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.cumulative_sizes


class ConcatDatasetSampler(torch.utils.data.Sampler):
    def __init__(
        self,
        datasets,
        batch_size,
        subdataset_lens,
        real_len,
        batching_strategy="use_all",
        repeat_factors=None,
        sample_weights=None,
        world_size=1,
        rank_id=0,
        seed=0,
        drop_last=0
    ):
        self.datasets = datasets
        self.batch_size = batch_size
        self.world_size = world_size
        self.rank_id = rank_id
        self.epoch = 0
        self.seed = seed
        self.real_len = real_len
        self.batching_strategy = batching_strategy
        self.repeat_factors = repeat_factors or [
            1 for _ in range(len(self.datasets.datasets))
        ]
        self.sample_weights = sample_weights or [
            None for _ in range(len(self.datasets.datasets))
        ]

        self.subdataset_lens = subdataset_lens
        self.num_samples = 0

        self.real_batch_size = []
        for i in range(len(self.datasets.datasets)):
            
            # real_batch_size = int(self.batch_size / len(self.datasets.datasets)) * self.world_size
            real_ratio_batch = int(
                self.batch_size
                * self.world_size
                * self.subdataset_lens[i]
                / self.real_len
            )
            # if real_ratio_batch % 2 != 0:
            #     real_ratio_batch = real_ratio_batch - 1
            
            self.num_samples += int(self.subdataset_lens[i] / self.world_size)
            
            self.real_batch_size.append(real_ratio_batch)
        
        # if real_len % self.world_size != 0:  # type: ignore[arg-type]
        #     # Split to nearest available length that is evenly divisible.
        #     # This is to ensure each rank receives the same amount of data when
        #     # using this Sampler.
        #     self.num_samples = math.ceil(
        #         (real_len - self.world_size) / self.world_size  # type: ignore[arg-type]
        #     )
        # else:
        #     self.num_samples = math.ceil(real_len / self.world_size)  # type: ignore[arg-type]
        # self.num_samples = int((real_len / self.world_size))

    @staticmethod
    def resample_list(train_indice, repeat_factor, sample_weight=None):
        # 如果repeat_factor大于1，那么我们将扩大列表
        # 如果repeat_factor小于1，那么我们将缩小列表
        # 如果repeat_factor等于1，那么列表保持不变
        new_size = int(len(train_indice) * repeat_factor)
        new_train_indice = random.choices(
            train_indice, weights=sample_weight, k=new_size
        )
        return new_train_indice

    def __iter__(self):
        random.seed(self.seed + self.epoch)

        batch_train_indices = []
        num_batches = []
        for i in range(len(self.datasets.datasets)):
            if self.repeat_factors[i] > 0:
                # get train_indices
                start_idx = self.datasets.cumulative_sizes[i - 1] if i > 0 else 0
                end_idx = self.datasets.cumulative_sizes[i]
                train_indice = list(range(start_idx, end_idx))
                # random.shuffle(train_indice)

                resample_train_indice = self.resample_list(
                    train_indice, self.repeat_factors[i], self.sample_weights[i]
                )

                num_batch = int(len(resample_train_indice) / self.real_batch_size[i])

                num_batches.append(num_batch)
                # get batch indices for each rank
                batch_train_indice = [
                    resample_train_indice[
                        batch * self.real_batch_size[i] : (batch + 1)
                        * self.real_batch_size[i]
                    ][self.rank_id :: self.world_size]
                    for batch in range(num_batch)
                ]
                batch_train_indices.append(batch_train_indice)
            else:
                assert (
                    self.repeat_factors[i] == -1
                ), "repetition factor must be > 0 or -1"
        # min_num_batch = min(num_batches)
        # train_indices_min = []

        # for batch in range(min_num_batch):
        #     for i in range(len(self.datasets.datasets)):
        #         train_indices_min.extend(batch_train_indices[i][batch])

        train_indices = []
        # for dataset_iters in itertools.zip_longest(*batch_train_indices):
        #     for dataset_iter in dataset_iters:
        #         if dataset_iter is not None:
        #             train_indices.extend(dataset_iter)
        for dataset_iters in zip(*batch_train_indices):
            for dataset_iter in dataset_iters:
                if dataset_iter is not None:
                    train_indices.extend(dataset_iter)

        return iter(train_indices)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        # 返回数据集中样本的数量
        return self.num_samples


class SubDatasetSampler(torch.utils.data.Sampler):
    def __init__(
        self,
        dataset,
        real_len,
        repeat_factors=1,
        sample_weights=None,
        world_size=1,
        rank_id=0,
        seed=0,
        drop_last=True
    ):
        self.dataset = dataset
        self.world_size = world_size
        self.rank_id = rank_id
        self.epoch = 0
        self.seed = seed
        self.repeat_factors = repeat_factors
        self.sample_weights = sample_weights
        self.real_len = real_len
        self.drop_last = drop_last
        
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and real_len % self.world_size != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (real_len - self.world_size) / self.world_size  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(real_len / self.world_size)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.world_size
        

    @staticmethod
    def resample_list(train_indice, repeat_factor, sample_weight=None):
        # 如果repeat_factor大于1，那么我们将扩大列表
        # 如果repeat_factor小于1，那么我们将缩小列表
        # 如果repeat_factor等于1，那么列表保持不变
        new_size = int(len(train_indice) * repeat_factor)
        new_train_indice = random.choices(
            train_indice, weights=sample_weight, k=new_size
        )
        return new_train_indice

    def __iter__(self):
        random.seed(self.seed + self.epoch)
        train_indice = list(range(len(self.dataset)))

        indices = self.resample_list(
            train_indice, self.repeat_factors, self.sample_weights
        )

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank_id:self.total_size:self.world_size]
        assert len(indices) == self.num_samples

        return iter(indices)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.num_samples


class concat_collater:
    def __init__(self):
        pass

    def __call__(self, batch):
        # batch = [(sample, label), (sample, label), ..., ((sample, label),(value, seeds)), ((sample, label),(value, seeds)), ...,]
        # grouped_batch = [batch[i:i+self.group_size[i]] for i in range(0, len(batch), self.group_size)]
        cur_index = 0
        modal_length = []
        cur_modal = batch[0][1]
        for i in range(len(batch)):
            if batch[i][1] != cur_modal:
                modal_length.append(i - cur_index)
                cur_modal = batch[i][1]
                cur_index = i
            if i == len(batch) - 1:
                modal_length.append(i - cur_index + 1)

        grouped_batch = []
        idx = 0
        for i in range(len(modal_length)):
            grouped_batch.append(batch[idx : idx + modal_length[i]])
            idx += modal_length[i]

        grouped_tensor = ()  # Initialize grouped_tensor as an empty tuple

        for group in grouped_batch:
            real_group = torch.utils.data.default_collate(group)
            grouped_tensor += (real_group,)

        return grouped_tensor  # Return grouped_tensor at the end of the function
