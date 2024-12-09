import ast
import json
import logging
import math
import os
import random
# from re import template
import sys
import braceexpand
from dataclasses import dataclass
from multiprocessing import Value

import numpy as np
import pandas as pd
import torch
import torchvision.datasets as datasets
from dataset_wrapper import DatasetSampleWrapper, SubDatasetSampler,TriDistillationDatasetWrapper
import webdataset as wds
from PIL import Image
from torch.utils.data import (
    Dataset,
    DataLoader,
    SubsetRandomSampler,
    IterableDataset,
    get_worker_info,
)
from torch.utils.data.distributed import DistributedSampler
from webdataset.filters import _shuffle
from webdataset.tariterators import (
    base_plus_ext,
    url_opener,
    tar_file_expander,
    valid_sample,
)

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None
from easydict import EasyDict as edict
from datasets.Sample import BatchCollator, Sample, SampleList, SampleCollator
from datasets.modal_audio.datasets import create_audio_datasets
from datasets.modal_3d.datasets import Dataset_3D
from datasets.modal_depth.datasets import create_rgbd_dataset
from datasets import build_video_dataset
from datasets.modal_video.dataloader_msrvtt_retrieval import dataloader_msrvtt_train, dataloader_msrvtt_test

from util.misc import get_rank, get_world_size
from util.logger import print_log
from datasets.zero_shot_metadata import OPENAI_IMAGENET_TEMPLATES, IMAGENET_CLASSNAMES



class ImageFolder_Sampler(datasets.ImageFolder):
    
    # 重载 __getitem__ 函数来包含文件路径
    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        rtn ={}

        rtn["image"] = image
        rtn["target"] = target
        caption = random.choice(IMAGENET_CLASSNAMES)
        template = random.choice(OPENAI_IMAGENET_TEMPLATES)
        caption =  template(caption)
        rtn['caption'] = caption

        return Sample(rtn)

    def collater(self, samples):
        return SampleCollator(self, samples)



class PrefetchLoader(object):
    """
    Modified from https://github.com/ChenRocks/UNITER.

    overlap compute and cuda data transfer
    (copied and then modified from nvidia apex)
    """

    def __init__(self, loader):
        self.loader = loader
        self.stream = torch.cuda.Stream()

    def __iter__(self):
        loader_it = iter(self.loader)
        self.preload(loader_it)
        batch = self.next(loader_it)
        while batch is not None:
            is_tuple = isinstance(batch, tuple)
            if is_tuple:
                task, batch = batch

            if is_tuple:
                yield task, batch
            else:
                yield batch
            batch = self.next(loader_it)

    def __len__(self):
        return len(self.loader)

    def preload(self, it):
        try:
            self.batch = next(it)
        except StopIteration:
            self.batch = None
            return
        # if record_stream() doesn't work, another option is to make sure
        # device inputs are created on the main stream.
        # self.next_input_gpu = torch.empty_like(self.next_input,
        #                                        device='cuda')
        # self.next_target_gpu = torch.empty_like(self.next_target,
        #                                         device='cuda')
        # Need to make sure the memory allocated for next_* is not still in use
        # by the main stream at the time we start copying to next_*:
        # self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self.batch = move_to_cuda(self.batch)
            # more code for the alternative if record_stream() doesn't work:
            # copy_ will record the use of the pinned source tensor in this
            # side stream.
            # self.next_input_gpu.copy_(self.next_input, non_blocking=True)
            # self.next_target_gpu.copy_(self.next_target, non_blocking=True)
            # self.next_input = self.next_input_gpu
            # self.next_target = self.next_target_gpu

    def next(self, it):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is not None:
            record_cuda_stream(batch)
        self.preload(it)
        return batch

    def __getattr__(self, name):
        method = self.loader.__getattribute__(name)
        return method


def record_cuda_stream(batch):
    if isinstance(batch, torch.Tensor):
        batch.record_stream(torch.cuda.current_stream())
    elif isinstance(batch, list) or isinstance(batch, tuple):
        for t in batch:
            record_cuda_stream(t)
    elif isinstance(batch, dict):
        for t in batch.values():
            record_cuda_stream(t)
    else:
        pass


def move_to_cuda(sample):
    def _move_to_cuda(tensor):
        return tensor.cuda()

    return apply_to_sample(_move_to_cuda, sample)


def apply_to_sample(f, sample):
    if len(sample) == 0:
        return {}

    def _apply(x):
        if torch.is_tensor(x):
            return f(x)
        elif isinstance(x, dict):
            rtn = {key: _apply(value) for key, value in x.items()}
            if isinstance(x, (SampleList, Sample)):
                rtn = x.__class__(rtn)
            return rtn
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        else:
            return x

    return _apply(sample)


class CsvDataset(Dataset):
    def __init__(
        self, input_filename, transforms, img_key, caption_key, sep="\t", tokenizer=None
    ):
        logging.debug(f"Loading csv data from {input_filename}.")
        df = pd.read_csv(input_filename, sep=sep)

        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        self.transforms = transforms
        logging.debug("Done loading data.")

        self.tokenize = tokenizer

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        images = self.transforms(Image.open(str(self.images[idx])))
        texts = self.tokenize([str(self.captions[idx])])[0]
        return images, texts


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value("i", epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)
    


def expand_urls(urls, weights=None):
    if weights is None:
        expanded_urls = wds.shardlists.expand_urls(urls)
        return expanded_urls, None
    if isinstance(urls, str):
        urllist = urls.split("::")
        weights = weights.split("::")
        assert (
            len(weights) == len(urllist)
        ), f"Expected the number of data components ({len(urllist)}) and weights({len(weights)}) to match."
        weights = [float(weight) for weight in weights]
        all_urls, all_weights = [], []
        for url, weight in zip(urllist, weights):
            expanded_url = list(braceexpand.braceexpand(url))
            expanded_weights = [weight for _ in expanded_url]
            all_urls.extend(expanded_url)
            all_weights.extend(expanded_weights)
        return all_urls, all_weights
    else:
        all_urls = list(urls)
        return all_urls, weights


def get_dataset_size(shards):
    shards_list, _ = expand_urls(shards)
    dir_path = os.path.dirname(shards_list[0])
    sizes_filename = os.path.join(dir_path, "sizes.json")
    len_filename = os.path.join(dir_path, "__len__")
    if os.path.exists(sizes_filename):
        sizes = json.load(open(sizes_filename, "r"))
        total_size = sum([int(sizes[os.path.basename(shard)]) for shard in shards_list])
    elif os.path.exists(len_filename):
        # FIXME this used to be eval(open(...)) but that seemed rather unsafe
        total_size = ast.literal_eval(open(len_filename, "r").read())
    else:
        total_size = None  # num samples undefined
        # some common dataset sizes (at time of authors last download)
        # CC3M (train): 2905954
        # CC12M: 10968539
        # LAION-400M: 407332084
        # LAION-2B (english): 2170337258
    num_shards = len(shards_list)
    return total_size, num_shards


def get_imagenet(args, preprocess_fns, split):
    assert split in ["train", "val", "v2"]
    is_train = split == "train"
    preprocess_train, preprocess_val = preprocess_fns

    if split == "v2":
        from imagenetv2_pytorch import ImageNetV2Dataset

        dataset = ImageNetV2Dataset(location=args.imagenet_v2, transform=preprocess_val)
    else:
        if is_train:
            data_path = args.imagenet_train
            preprocess_fn = preprocess_train
        else:
            data_path = args.imagenet_val
            preprocess_fn = preprocess_val
        assert data_path

        dataset = datasets.ImageFolder(data_path, transform=preprocess_fn)

    if is_train:
        idxs = np.zeros(len(dataset.targets))
        target_array = np.array(dataset.targets)
        k = 50
        for c in range(1000):
            m = target_array == c
            n = len(idxs[m])
            arr = np.zeros(n)
            arr[:k] = 1
            np.random.shuffle(arr)
            idxs[m] = arr

        idxs = idxs.astype("int")
        sampler = SubsetRandomSampler(np.where(idxs)[0])
    else:
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampler=sampler,
    )

    return DataInfo(dataloader=dataloader, sampler=sampler)


def count_samples(dataloader):
    os.environ["WDS_EPOCH"] = "0"
    n_elements, n_batches = 0, 0
    for images, texts in dataloader:
        n_batches += 1
        n_elements += len(images)
        assert len(images) == len(texts)
    return n_elements, n_batches


def filter_no_caption_or_no_image(sample):
    has_caption = "txt" in sample
    has_image = (
        "png" in sample or "jpg" in sample or "jpeg" in sample or "webp" in sample
    )
    return has_caption and has_image


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    logging.warning(f"Handling webdataset error ({repr(exn)}). Ignoring.")
    return True


def group_by_keys_nothrow(
    data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None
):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if (
            current_sample is None
            or prefix != current_sample["__key__"]
            or suffix in current_sample
        ):
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


def tarfile_to_samples_nothrow(src, handler=log_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples


def pytorch_worker_seed(increment=0):
    """get dataloader worker seed from pytorch"""
    worker_info = get_worker_info()
    if worker_info is not None:
        # favour using the seed already created for pytorch dataloader workers if it exists
        seed = worker_info.seed
        if increment:
            # space out seed increments so they can't overlap across workers in different iterations
            seed += increment * max(1, worker_info.num_workers)
        return seed
    # fallback to wds rank based seed
    return wds.utils.pytorch_worker_seed()


_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000


class detshuffle2(wds.PipelineStage):
    def __init__(
        self,
        bufsize=1000,
        initial=100,
        seed=0,
        epoch=-1,
    ):
        self.bufsize = bufsize
        self.initial = initial
        self.seed = seed
        self.epoch = epoch

    def run(self, src):
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        rng = random.Random()
        if self.seed < 0:
            # If seed is negative, we use the worker's seed, this will be different across all nodes/workers
            seed = pytorch_worker_seed(epoch)
        else:
            # This seed to be deterministic AND the same across all nodes/workers in each epoch
            seed = self.seed + epoch
        rng.seed(seed)
        return _shuffle(src, self.bufsize, self.initial, rng)


class ResampledShards2(IterableDataset):
    """An iterable dataset yielding a list of urls."""

    def __init__(
        self,
        urls,
        weights=None,
        nshards=sys.maxsize,
        worker_seed=None,
        deterministic=False,
        epoch=-1,
    ):
        """Sample shards from the shard list with replacement.

        :param urls: a list of URLs as a Python list or brace notation string
        """
        super().__init__()
        urls, weights = expand_urls(urls, weights)
        self.urls = urls
        self.weights = weights
        if self.weights is not None:
            assert (
                len(self.urls) == len(self.weights)
            ), f"Number of urls {len(self.urls)} and weights {len(self.weights)} should match."
        assert isinstance(self.urls[0], str)
        self.nshards = nshards
        self.rng = random.Random()
        self.worker_seed = worker_seed
        self.deterministic = deterministic
        self.epoch = epoch

    def __iter__(self):
        """Return an iterator over the shards."""
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        if self.deterministic:
            # reset seed w/ epoch if deterministic
            if self.worker_seed is None:
                # pytorch worker seed should be deterministic due to being init by arg.seed + rank + worker id
                seed = pytorch_worker_seed(epoch)
            else:
                seed = self.worker_seed() + epoch
            self.rng.seed(seed)
        for _ in range(self.nshards):
            if self.weights is None:
                yield dict(url=self.rng.choice(self.urls))
            else:
                yield dict(
                    url=self.rng.choices(self.urls, weights=self.weights, k=1)[0]
                )


def get_wds_dataset(
    args, preprocess_img, is_train, epoch=0, floor=False, tokenizer=None
):
    input_shards = args.train_data if is_train else args.val_data
    assert input_shards is not None
    resampled = getattr(args, "dataset_resampled", False) and is_train

    num_shards = None
    if is_train:
        if args.train_num_samples is not None:
            num_samples = args.train_num_samples
        else:
            num_samples, num_shards = get_dataset_size(input_shards)
            if not num_samples:
                raise RuntimeError(
                    "Currently, the number of dataset samples must be specified for the training dataset. "
                    "Please specify it via `--train-num-samples` if no dataset length info is present."
                )
    else:
        # Eval will just exhaust the iterator if the size is not specified.
        num_samples = args.val_num_samples or 0

    shared_epoch = SharedEpoch(
        epoch=epoch
    )  # create a shared epoch store to sync epoch to dataloader worker proc

    if is_train and args.train_data_upsampling_factors is not None:
        assert resampled, "--train_data_upsampling_factors is only supported when sampling with replacement (with --dataset-resampled)."

    if resampled:
        pipeline = [
            ResampledShards2(
                input_shards,
                weights=args.train_data_upsampling_factors,
                deterministic=True,
                epoch=shared_epoch,
            )
        ]
    else:
        pipeline = [wds.SimpleShardList(input_shards)]

    # at this point we have an iterator over all the shards
    if is_train:
        if not resampled:
            pipeline.extend(
                [
                    detshuffle2(
                        bufsize=_SHARD_SHUFFLE_SIZE,
                        initial=_SHARD_SHUFFLE_INITIAL,
                        seed=args.seed,
                        epoch=shared_epoch,
                    ),
                    wds.split_by_node,
                    wds.split_by_worker,
                ]
            )
        pipeline.extend(
            [
                # at this point, we have an iterator over the shards assigned to each worker at each node
                tarfile_to_samples_nothrow,  # wds.tarfile_to_samples(handler=log_and_continue),
                wds.shuffle(
                    bufsize=_SAMPLE_SHUFFLE_SIZE,
                    initial=_SAMPLE_SHUFFLE_INITIAL,
                ),
            ]
        )
    else:
        pipeline.extend(
            [
                wds.split_by_worker,
                # at this point, we have an iterator over the shards assigned to each worker
                wds.tarfile_to_samples(handler=log_and_continue),
            ]
        )
    pipeline.extend(
        [
            wds.select(filter_no_caption_or_no_image),
            wds.decode("pilrgb", handler=log_and_continue),
            wds.rename(image="jpg;png;jpeg;webp", text="txt"),
            wds.map_dict(image=preprocess_img, text=lambda text: tokenizer(text)[0]),
            wds.to_tuple("image", "text"),
            wds.batched(args.batch_size, partial=not is_train),
        ]
    )

    dataset = wds.DataPipeline(*pipeline)

    if is_train:
        if not resampled:
            num_shards = num_shards or len(expand_urls(input_shards)[0])
            assert (
                num_shards >= args.num_workers * args.world_size
            ), "number of shards must be >= total workers"
        # roll over and repeat a few samples to get same number of full batches on each node
        round_fn = math.floor if floor else math.ceil
        global_batch_size = args.batch_size * args.world_size
        num_batches = round_fn(num_samples / global_batch_size)
        num_workers = max(1, args.num_workers)
        num_worker_batches = round_fn(
            num_batches / num_workers
        )  # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size
        dataset = dataset.with_epoch(
            num_worker_batches
        )  # each worker is iterating over this
    else:
        # last batches are partial, eval is done on single (master) node
        num_batches = math.ceil(num_samples / args.batch_size)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )

    # FIXME not clear which approach is better, with_epoch before vs after dataloader?
    # hoping to resolve via https://github.com/webdataset/webdataset/issues/169
    # if is_train:
    #     # roll over and repeat a few samples to get same number of full batches on each node
    #     global_batch_size = args.batch_size * args.world_size
    #     num_batches = math.ceil(num_samples / global_batch_size)
    #     num_workers = max(1, args.num_workers)
    #     num_batches = math.ceil(num_batches / num_workers) * num_workers
    #     num_samples = num_batches * global_batch_size
    #     dataloader = dataloader.with_epoch(num_batches)
    # else:
    #     # last batches are partial, eval is done on single (master) node
    #     num_batches = math.ceil(num_samples / args.batch_size)

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)


def get_csv_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    dataset = CsvDataset(
        input_filename,
        preprocess_fn,
        img_key=args.csv_img_key,
        caption_key=args.csv_caption_key,
        sep=args.csv_separator,
        tokenizer=tokenizer,
    )
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


class SyntheticDataset(Dataset):
    def __init__(
        self,
        transform=None,
        image_size=(224, 224),
        caption="Dummy caption",
        dataset_size=100,
        tokenizer=None,
    ):
        self.transform = transform
        self.image_size = image_size
        self.caption = caption
        self.image = Image.new("RGB", image_size)
        self.dataset_size = dataset_size

        self.preprocess_txt = lambda text: tokenizer(text)[0]

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        if self.transform is not None:
            image = self.transform(self.image)
        return image, self.preprocess_txt(self.caption)


def get_synthetic_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    image_size = preprocess_fn.transforms[0].size
    dataset = SyntheticDataset(
        transform=preprocess_fn,
        image_size=image_size,
        dataset_size=args.train_num_samples,
        tokenizer=tokenizer,
    )
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def get_dataset_fn(data_path, dataset_type):
    if dataset_type == "webdataset":
        return get_wds_dataset
    elif dataset_type == "csv":
        return get_csv_dataset
    elif dataset_type == "synthetic":
        return get_synthetic_dataset
    elif dataset_type == "auto":
        ext = data_path.split(".")[-1]
        if ext in ["csv", "tsv"]:
            return get_csv_dataset
        elif ext in ["tar"]:
            return get_wds_dataset
        else:
            raise ValueError(
                f"Tried to figure out dataset type, but failed for extension {ext}."
            )
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")


def get_data(args, preprocess_fns, epoch=0, tokenizer=None):
    preprocess_train, preprocess_val = preprocess_fns
    data = {}

    if args.train_data or args.dataset_type == "synthetic":
        data["train"] = get_dataset_fn(args.train_data, args.dataset_type)(
            args, preprocess_train, is_train=True, epoch=epoch, tokenizer=tokenizer
        )

    if args.val_data:
        data["val"] = get_dataset_fn(args.val_data, args.dataset_type)(
            args, preprocess_val, is_train=False, tokenizer=tokenizer
        )

    if args.imagenet_val is not None:
        data["imagenet-val"] = get_imagenet(args, preprocess_fns, "val")

    if args.imagenet_v2 is not None:
        data["imagenet-v2"] = get_imagenet(args, preprocess_fns, "v2")

    return data


def get_joint_data(args, epoch=0, tokenizer=None, image_transform=None, logger=None):
    data = edict()
    data["val"] = edict()

    transform_train, transform_val = None, None

    world_size = get_world_size()
    rank = get_rank()

    if image_transform is not None:
        if isinstance(image_transform, (tuple, list)):
            transform_train, transform_val = image_transform
        else:
            transform_train, transform_val = image_transform, image_transform

    if "image" in args.train_modal_list:
        image_data_path = os.path.join(args.img_data_path, "train")
        image_train_dataset = ImageFolder_Sampler(
            image_data_path, transform=transform_train
        )

    if "image" in args.eval_modal_list:
        image_data_path = os.path.join(args.img_data_path, "val")
        image_val_dataset = ImageFolder_Sampler(
            image_data_path, transform=transform_val
        )
        
        image_val_sampler = DistributedSampler(image_val_dataset) if args.distributed else None

        image_val_dataloader = DataLoader(
            image_val_dataset,
            batch_size=256,
            sampler=image_val_sampler,
            num_workers=args.num_workers,
            shuffle=False,
            pin_memory=args.pin_mem,
            drop_last=False,
            collate_fn=image_val_dataset.collater
                    if hasattr(image_val_dataset, "collator")
                    else BatchCollator(dataset_type="val"),
        )
        image_val_dataloader.num_samples = len(image_val_dataset)
        image_val_dataloader.num_batches = len(image_val_dataloader)
        image_val_dataloader = PrefetchLoader(image_val_dataloader)
        data["val"]["image"] = DataInfo(image_val_dataloader, image_val_sampler).dataloader

    if "audio" in args.train_modal_list:
        audio_train_dataset = create_audio_datasets(
            args=args,
            is_train=True,
            tokenizer=tokenizer,
            image_transform=transform_train,
        )
        
        if 'vggsound_video' in args.audio_train_data and len(audio_train_dataset)>1:
            vggsound_video_train_dataset = audio_train_dataset.pop('vggsound_video')
            datasets_list = [v for _, v in audio_train_dataset.items()]
            audio_train_dataset=datasets_list[0]

        if args.multi_modal_distill:
            print_log("Using audio teacher model for knowledge distillation!!!",logger=logger)
            audio_train_dataset = DatasetSampleWrapper(
                audio_train_dataset,
                args.audio_logits_path,
                args.audio_topk,
                args.save_logits,
                args.audio_logits_name,
            )

    if "audio" in args.eval_modal_list:
        audio_val_dataset = create_audio_datasets(
            args=args,
            is_train=False,
            tokenizer=tokenizer,
            image_transform=transform_val,
        )

        if isinstance(audio_val_dataset, dict):
            multiple_datainfo = dict()
            for dname, dset in audio_val_dataset.items():
                num_samples = len(dset)
                
                sampler = DistributedSampler(dataset=dset,shuffle=False) if args.distributed else None

                dloader = DataLoader(
                    dset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                    pin_memory=args.pin_mem,
                    sampler=sampler,
                    drop_last=False,
                    collate_fn=dset.collater
                    if hasattr(dset, "collator")
                    else BatchCollator(dataset_type="val"),
                )
                dloader.num_samples = num_samples
                dloader.num_batches = len(dloader)
                dloader = PrefetchLoader(dloader)
                multiple_datainfo[dname] = DataInfo(dloader, sampler).dataloader
            data["val"]["audio"] = multiple_datainfo

        else:
            num_samples = len(audio_val_dataset)
            
            sampler = DistributedSampler(dataset=audio_val_dataset,shuffle=False) if args.distributed else None

            dataloader = DataLoader(
                audio_val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                sampler=sampler,
                drop_last=False,
                collate_fn=audio_val_dataset.collater
                if hasattr(audio_val_dataset, "collator")
                else BatchCollator(dataset_type="val"),
            )
            dataloader.num_samples = num_samples
            dataloader.num_batches = len(dataloader)
            dataloader = PrefetchLoader(dataloader)
            data["val"]["audio"] = DataInfo(dataloader, sampler).dataloader

    if "point" in args.train_modal_list:
        point_train_dataset = Dataset_3D(
            args=args,
            tokenizer=tokenizer,
            dataset_name=args.point_train_data,
            train_transform=transform_train,
        ).dataset

        if args.multi_modal_distill:
            print_log("Using point teacher model for knowledge distillation!!!",logger=logger)
            
            point_train_dataset = DatasetSampleWrapper(
                point_train_dataset,
                args.pc_logits_path,
                args.pc_topk,
                args.save_logits,
                args.pc_logits_name,
            )

    if "point" in args.eval_modal_list:
        dataset_names = args.point_val_data.split("::")
        multiple_datainfo = dict()
        for dname in dataset_names:
        
            point_val_dataset = Dataset_3D(
                args=args,
                tokenizer=tokenizer,
                dataset_name=dname,
                train_transform=transform_val,
            ).dataset
            num_samples = len(point_val_dataset)
            sampler = DistributedSampler(point_val_dataset,shuffle=False) if args.distributed else None
            
            dataloader = DataLoader(
                point_val_dataset,
                batch_size=64,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                sampler=sampler,
                drop_last=False,
                collate_fn=point_val_dataset.collater
                if hasattr(point_val_dataset, "collator")
                else BatchCollator(dataset_type="val"),
            )
            dataloader.num_samples = num_samples
            dataloader.num_batches = len(dataloader)
            multiple_datainfo[dname] = DataInfo(dataloader, sampler).dataloader
        data["val"]["point"] = multiple_datainfo

    if "rgbd" in args.train_modal_list:
        rgbd_train_dataset = create_rgbd_dataset(
            args=args,
            is_train=True,
            tokenizer=tokenizer,
            image_transform=transform_train,
        )
        if args.multi_modal_distill:
            print_log("Using point teacher model for knowledge distillation!!!",logger=logger)
            
            rgbd_train_dataset = TriDistillationDatasetWrapper(
                rgbd_train_dataset,
                args.rgbd_logits_path,
                args.rgbd_topk,
                args.save_logits,
                args.rgbd_logits_name,
                args.rgbd_text_logits_name,
                args.rgbd_image_logits_name,
            )

    if "rgbd" in args.eval_modal_list:
        rgbd_val_dataset = create_rgbd_dataset(
            args=args,
            is_train=False,
            tokenizer=tokenizer,
            image_transform=transform_val,
        )

        if isinstance(rgbd_val_dataset, dict):
            multiple_datainfo = dict()
            for dname, dset in rgbd_val_dataset.items():
                num_samples = len(dset)
                
                sampler = DistributedSampler(dset,shuffle=False) if args.distributed else None
                dloader = DataLoader(
                    dset,
                    batch_size=128,
                    shuffle=False,
                    num_workers=args.num_workers,
                    pin_memory=args.pin_mem,
                    sampler=sampler,
                    drop_last=False,
                    collate_fn=dset.collater
                    if hasattr(dset, "collator")
                    else BatchCollator(dataset_type="val"),
                )
                dloader.num_samples = num_samples
                dloader.num_batches = len(dloader)
                dloader = PrefetchLoader(dloader)
                multiple_datainfo[dname] = DataInfo(dloader, sampler).dataloader
            data["val"]["rgbd"] = multiple_datainfo

        else:
            num_samples = len(rgbd_val_dataset)
            
            sampler = DistributedSampler(rgbd_val_dataset) if args.distributed else None
            dataloader = DataLoader(
                rgbd_val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                sampler=sampler,
                drop_last=False,
                collate_fn=rgbd_val_dataset.collater
                if hasattr(rgbd_val_dataset, "collator")
                else BatchCollator(dataset_type="val"),
            )
            dataloader.num_samples = num_samples
            dataloader.num_batches = len(dataloader)
            dataloader = PrefetchLoader(dataloader)
            data["val"]["rgbd"] = DataInfo(dataloader, sampler).dataloader
        
    if "video" in args.train_modal_list:
        
        if 'vggsound_video' in args.audio_train_data:
            video_train_dataset = vggsound_video_train_dataset
        elif 'msrvtt' in args.video_train_data:
            video_train_dataset = dataloader_msrvtt_train(args,tokenizer)
        else:
            video_train_dataset, video_nb_classes = build_video_dataset(
                is_train=True, test_mode=False, args=args
            )

        if args.multi_modal_distill:
            print_log("use video teacher model for knowledge distillation r", logger=logger)
            video_train_dataset = DatasetSampleWrapper(
                video_train_dataset,
                args.video_logits_path,
                args.video_topk,
                args.save_logits,
            )
            
    if 'video' in args.eval_modal_list:
        if 'msrvtt' in args.video_val_data:
            video_val_dataset = dataloader_msrvtt_test(args,tokenizer)
        else:
            video_val_dataset, _ = build_video_dataset(
                is_train=False, test_mode=False, args=args
            )
        
        num_samples = len(video_val_dataset)
        sampler = DistributedSampler(video_val_dataset) if args.distributed else None
        
        dataloader = torch.utils.data.DataLoader(
            video_val_dataset,
            sampler=sampler,
            shuffle=False,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
            collate_fn=rgbd_val_dataset.collater
                if hasattr(video_val_dataset, "collator")
                else BatchCollator(dataset_type="val"),
        )
        dataloader.num_samples = num_samples
        dataloader.num_batches = len(dataloader)
        dataloader = PrefetchLoader(dataloader)
        data["val"]["video"] = DataInfo(dataloader, sampler).dataloader
        

        
    dataset_list = []
    repeat_factors = []
    for modal in args.train_modal_list:
        if modal == "image":
            dataset_list.append(image_train_dataset)
            repeat_factors.append(args.img_rep_w)
        elif modal == "audio":
            dataset_list.append(audio_train_dataset)
            repeat_factors.append(args.audio_rep_w)
        elif modal == "point":
            dataset_list.append(point_train_dataset)
            repeat_factors.append(args.pc_rep_w)
        elif modal == "rgbd":
            dataset_list.append(rgbd_train_dataset)
            repeat_factors.append(args.rgbd_rep_w)
        elif modal == "video":
            dataset_list.append(video_train_dataset)
            repeat_factors.append(args.video_rep_w)

    if args.weight_sampler:
        audio_weight_csv = torch.from_numpy(
            np.loadtxt(args.audio_weight_csv, delimiter=",")
        )
        samples_weight = []
        for m in args.train_modal_list:
            if m == "audio":
                samples_weight.append(audio_weight_csv)
                print_log("Using class-balanced audio sampler for AudioSet.",logger=logger)
            else:
                samples_weight.append(None)
        
    else:
        samples_weight = None

    subdataset_lens = []
    real_len = 0
    for i in range(len(dataset_list)):
        if repeat_factors[i] > 0:
            real_len += int(len(dataset_list[i]) * repeat_factors[i])
            subdataset_lens.append(int(len(dataset_list[i]) * repeat_factors[i]))
        else:
            real_len += len(dataset_list[i])
            subdataset_lens.append(len(dataset_list[i]))
    print_log(
        f"Total Dataset len:{real_len} \n Sub Dataset len: {subdataset_lens}",logger=logger
    )

    real_batch_size = []
    for i in range(len(dataset_list)):
        real_ratio_batch = int(
            args.batch_size * subdataset_lens[i] / real_len * world_size
        )
        if real_ratio_batch % 2 != 0:
            real_ratio_batch = real_ratio_batch - 1
        real_batch_size.append(real_ratio_batch)

    for i in range(len(real_batch_size)):
        print_log(
            f"real_batch_size of {args.train_modal_list[i]} is {real_batch_size[i]}",logger=logger
        )

    eff_batch_size = real_batch_size[0] * args.accum_iter

    train_loaders = []

    for i in range(len(dataset_list)):
        subsampler = SubDatasetSampler(
            dataset_list[i],
            subdataset_lens[i],
            repeat_factors[i],
            samples_weight[i] if samples_weight is not None else None,
            world_size,
            rank,
            drop_last=True,
        )

        loader = torch.utils.data.DataLoader(
            dataset_list[i],
            sampler=subsampler,
            batch_size=real_batch_size[i],
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
            
            collate_fn=dataset_list[i].collater
                if hasattr(dataset_list[i], "collator")
                else BatchCollator(dataset_type="train"),
        )

        loader.num_samples = subsampler.total_size
        loader.num_batches = len(loader)

        loader = PrefetchLoader(loader)

        train_loaders.append(DataInfo(loader, subsampler))

    data["train"] = MultiLoader(args.train_modal_list, *train_loaders)

    return data, eff_batch_size, real_batch_size


class MultiLoader:
    def __init__(self, modality_list, *datainfo):
        self.datainfo = datainfo
        self.loaders = [d.dataloader for d in datainfo]
        self.iterators = [iter(loader) for loader in self.loaders]
        self.current_loader_idx = 0
        self.modality_list = modality_list
        self.datasets = {}
        for i, modal in enumerate(modality_list):
            self.datasets[modal] = self.loaders[i].dataset

        self.num_batches = len(self)
        self.num_samples = sum([loader.num_samples for loader in self.loaders])

    def __iter__(self):
        return self

    def __next__(self):
        if not self.iterators:
            raise StopIteration

        current_iterator = self.iterators[self.current_loader_idx]
        
        try:
            batch = next(current_iterator)
        except StopIteration:
            
            self.current_loader_idx = (self.current_loader_idx + 1) % len(self.loaders)
            
            current_iterator = self.iterators[self.current_loader_idx]
            
            batch = next(current_iterator)

        current_modal = self.modality_list[self.current_loader_idx]
        
        self.current_loader_idx = (self.current_loader_idx + 1) % len(self.loaders)

        return batch, current_modal

    def set_epoch(self, epoch):
        self.current_loader_idx = 0
        self.iterators = []
        for data_info in self.datainfo:
            data_info.set_epoch(epoch)
            self.iterators.append(iter(data_info.dataloader))

    def __len__(self):
        return sum([len(loader) for loader in self.loaders])

