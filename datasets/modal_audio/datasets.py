import os
import csv
import random
import json
import logging
from turtle import rt
import pandas as pd
import numpy as np
import einops

from typing import Iterable
from PIL import Image
import torchaudio
import torch
import torch.nn as nn
from torch.utils.data import Dataset, ConcatDataset
from omegaconf import OmegaConf
from easydict import EasyDict as edict

from datasets.Sample import Sample, SampleCollator
from datasets.modal_audio.processors.at_processor import (
    PVProcessorTrain,
    PVProcessorEval,
    ASTProcessorTrain,
    ASTProcessorEval,
    CLAPProcessorTrain,
    CLAPProcessprEval,
    BlipCaptionProcessor,
)
from datasets.modal_audio.data.sound_cls_template import SOUND_AS_IMAGE_TEMPLATE
from datasets.constants import AUDIO_DATA_DIR, AUDIO_META_DATA_DIR
from util.logger import print_log

norm_stats = {
    "audioset": [-4.2677393, 4.5689974],
    "k400": [-4.2677393, 4.5689974],
    "esc50": [-6.6268077, 5.358466],
    "speechcommands": [-6.845978, 5.5654526],
}
target_length = {
    "audioset": 1024,
    "k400": 1024,
    "esc50": 512,
    "speechcommands": 128,
}
multilabel_dataset = {
    "audioset": True,
    "esc50": False,
    "k400": False,
    "speechcommands": True,
}


def extract_sound_description(input_string):
    prefixes = [
        "the sound of",
        "the sounds of",
        "sound of",
        "sounds of",
        "a sound of",
        "a sounds of",
        "the noise of",
        "the hum of",
        "the roar of",
        "the chirping of",
        "the rustle of",
        "the howl of",
        "the buzz of",
        "the patter of",
        "the crash of",
        "the whir of",
    ]

    input_string = input_string.lower()

    for prefix in prefixes:
        if input_string.startswith(prefix):
            return input_string[len(prefix) :].strip()

    return input_string


def wrap_list(x):
    if isinstance(x, list):
        return x
    return [
        x,
    ]


def load_annotation(filename, sep="\t", header=0):
    if filename.endswith(".json"):
        anno = json.load(open(filename, "r"))
    elif filename.endswith(".tsv"):
        anno = pd.read_csv(filename, sep=sep, header=header)
    else:
        raise NotImplementedError
    return anno


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
        audio_processor=None,
        text_processor=None,
        data_root=None,
        anno_path={},
        args=None,
        split=None,
        tokenizer=None,
        **kwargs,
    ):
        self.data_root = data_root
        self.anno_path = anno_path
        self.annotation = None

        self.vis_processor = vis_processor
        self.audio_processor = audio_processor
        self.text_processor = text_processor

        self.args = args
        self.split = split

        self.tokenizer = tokenizer
        self.data_type = None

    def __len__(self):
        return len(self.annotation)

    def collater(self, samples):
        return SampleCollator(self, samples)

    def set_processors(self, vis_processor, audio_processor, text_processor):
        self.vis_processor = vis_processor
        self.audio_processor = audio_processor
        self.text_processor = text_processor


class AudioBaseDataset(Dataset):
    def __init__(
        self,
        vis_processor=None,
        audio_processor=None,
        text_processor=None,
        data_root=None,
        anno_path={},
        args=None,
        split=None,
        tokenizer=None,
        image_transform=None,
        **kwargs,
    ):
        self.data_root = data_root
        self.anno_path = anno_path
        self.annotation = None

        self.vis_processor = vis_processor
        self.audio_processor = audio_processor
        self.text_processor = text_processor

        self.args = args
        self.split = split
        self.tokenizer = tokenizer
        self.image_transform = image_transform

        # whether load vision for training
        self.load_vision = args.audio_load_vision

    def __len__(self):
        return len(self.annotation)

    def collater(self, samples):
        return SampleCollator(self, samples)

    def set_processors(self, vis_processor, text_processor):
        self.vis_processor = vis_processor
        self.text_processor = text_processor

    def __getitem__(self, index):
        rtn = dict()
        return Sample(rtn)


def make_index_dict(label_csv):
    index_lookup = {}
    with open(label_csv, "r") as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            index_lookup[row["mid"]] = row["index"]
            line_count += 1
    return index_lookup


def make_name_dict(label_csv):
    name_lookup = {}
    with open(label_csv, "r") as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            name_lookup[row["index"]] = row["display_name"]
            line_count += 1
    return name_lookup


class AudioSetDataset(AudioBaseDataset):
    def __init__(
        self,
        vis_processor=None,
        audio_processor=None,
        text_processor=None,
        data_root=AUDIO_DATA_DIR,
        anno_path={
            "balanced_train": f"{AUDIO_META_DATA_DIR}/audioset_balanced_train.json",
            "unbalanced_train": f"{AUDIO_META_DATA_DIR}/unbalanced_train.json",
            "audioset_train_all": [
                f"{AUDIO_META_DATA_DIR}/audioset_balanced_train.json",
                f"{AUDIO_META_DATA_DIR}/audioset_unbalanced_train.json",
            ],
            "audioset_vgg": [
                # f"{AUDIO_META_DATA_DIR}/audioset_balanced_train.json",
                f"{AUDIO_META_DATA_DIR}/audioset_balanced_train.json",
                "data/vggsound/train.json",
            ],
            "vgg": "data/vggsound/train.json",
            "val": f"{AUDIO_META_DATA_DIR}/audioset_val.json",
            "test": f"{AUDIO_META_DATA_DIR}/audioset_val.json",
        },
        args=None,
        split=None,
        tokenizer=None,
        image_transform=None,
        **kwargs,
    ):
        super().__init__(
            vis_processor,
            audio_processor,
            text_processor,
            data_root,
            anno_path,
            args,
            split,
            tokenizer,
            image_transform,
            **kwargs,
        )

        # init annotation
        if isinstance(anno_path[split], list):
            self.annotation = []
            for path in anno_path[split]:
                self.annotation = self.annotation + load_annotation(path)
        else:
            self.annotation = load_annotation(anno_path[split])

        if (
            split
            in [
                "balanced_train",
                "unbalanced_train",
                "audioset_vgg",
                "audioset_train_all",
                
            ]
            and self.load_vision
        ):
            print_log("[AudiosetDataset] : Load good videos.",'Audioset')
            self.annotation = [
                item
                for item in self.annotation
                if item["is_good_video"] == True  # some videos are corrupted
            ]

        self.init_class_labels()

        # Pretrain specific
        self.is_train = self.split in [
            "train",
            "balanced_train",
            "unbalanced_train",
            "audioset_vgg",
            "audioset_train_all",
            "vgg"
        ]
        self.mix_up = self.is_train and self.args.audio_mix_up

        # Evaluation specific
        self.eval_metric = "mAP"

    def init_class_labels(self):
        
        self.num_classes = 527
        self.idx2label = []
        self.label2idx = {}

        class_f = pd.read_csv(
            f"{AUDIO_META_DATA_DIR}/audioset_class_labels_indices.csv", header=0
        )
        for i in range(len(class_f)):
            item = class_f.iloc[i]
            assert item["index"] == i
            cls_name = item["display_name"].lower()  # use lower case
            self.idx2label.append(cls_name)
            self.label2idx[cls_name] = i
        assert len(self.idx2label) == self.num_classes

    def __getitem__(self, index):
        rtn = dict()
        n_retry = 0
        while len(rtn) == 0:
            try:
                ann = self.annotation[index]
                ann["video_path"] = os.path.join(self.data_root, ann["video_path"])
                ann["audio_path"] = os.path.join(self.data_root, ann["audio_path"])
                # ann["audio_path"] = os.path.join(self.data_root, ann["wav"])
                second_ann = None  # for mix up
                mix_lambda = None
                if self.mix_up and random.random() < self.args.audio_mix_up_p:
                    second_ann = self.annotation[
                        random.randint(0, len(self.annotation) - 1)
                    ]
                    second_ann["video_path"] = os.path.join(
                        self.data_root, second_ann["video_path"]
                    )
                    second_ann["audio_path"] = os.path.join(
                        self.data_root, second_ann["audio_path"]
                    )
                    mix_lambda = np.random.beta(10, 10)

                if self.is_train and self.load_vision:  # load video data for training
                    if second_ann is not None:  # mixup
                        vis_data, (start, end) = self.vis_processor(ann["video_path"])
                        sec_vis_data, (sec_start, sec_end) = self.vis_processor(
                            second_ann["video_path"]
                        )
                        vis_data = (
                            mix_lambda * vis_data + (1 - mix_lambda) * sec_vis_data
                        )

                        wf = self.audio_processor.load_audio_clip(
                            ann["audio_path"], start=start, end=end
                        )  # already sub mean
                        sec_wf = self.audio_processor.load_audio_clip(
                            second_ann["audio_path"], start=sec_start, end=sec_end
                        )  # already sub mean
                        mix_wf = mix_lambda * wf + (1 - mix_lambda) * sec_wf
                        mix_wf = mix_wf - mix_wf.mean()
                        audio_data = self.audio_processor(mix_wf)

                    else:  # no mixup
                        vis_data, (start, end) = self.vis_processor(ann["video_path"])
                        audio_data = self.audio_processor(
                            ann["audio_path"], se=(start, end)
                        )

                else:
                    vis_data = None
                    if second_ann is not None:  # mixup
                        wf = self.audio_processor.load_audio_clip(
                            ann["audio_path"]
                        )  # already sub mean
                        sec_wf = self.audio_processor.load_audio_clip(
                            second_ann["audio_path"]
                        )  # already sub mean
                        mix_wf = mix_lambda * wf + (1 - mix_lambda) * sec_wf
                        mix_wf = mix_wf - mix_wf.mean()
                        audio_data = self.audio_processor(mix_wf)
                    else:  # no mixup
                        audio_data = self.audio_processor(ann["audio_path"])

                if vis_data is not None:
                    if (
                        vis_data.ndim == 4 and vis_data.size(1) == self.args.audio_n_frames
                    ):  # [ C x T x H x W ] 
                        # vis_data = einops.rearrange(vis_data, "c t h w -> t c h w")
                        # assert vis_data.size(1) == 3  # rgb channel
                        assert vis_data.size(0) == 3  # rgb channel
                        vis_data.squeeze(1)
                    elif vis_data.ndim == 3:
                        assert vis_data.size(0) == 3  # rgb channel
                        # vis_data.unsqueeze(0)

                rtn["image"] = vis_data
                rtn["audio"] = audio_data

                # text; TODO: make caption and template configurable later
                caption = None
                if len(ann["captions"]) > 1 and random.random() < 0.5:
                    caption = random.choice(
                        ann["captions"][1:]
                    )  # choose from additional captions

                else:
                    caption = ann["captions"][0]  # ann["captions"][0] is class names
                    caption = random.choice(SOUND_AS_IMAGE_TEMPLATE)(
                        caption
                    )  # add templates
                caption = self.text_processor(caption)

                if second_ann is not None:
                    # mixup case
                    sec_caption = random.choice(second_ann["captions"])
                    if caption.endswith("."):
                        caption = caption[:-1]
                    caption += f" and {sec_caption.lower()}"

                tokenized_caption = self.tokenizer([caption])[0]
                rtn["caption"] = tokenized_caption

                rtn["class_name"] = wrap_list(ann["class_name"])
                rtn["class_labels"] = wrap_list(ann["class_labels"])
                if second_ann:
                    # not use in training, in case for check
                    rtn["class_name"] = (
                        rtn["class_name"]
                        + [
                            "###",
                        ]
                        + wrap_list(second_ann["class_name"])
                    )
                    rtn["class_labels"] = (
                        rtn["class_labels"]
                        + [
                            "###",
                        ]
                        + wrap_list(second_ann["class_labels"])
                    )

                rtn["mixup_lambda"] = mix_lambda

                if self.split=='vgg':
                    rtn["target"] = ann["class_labels"]
                    
                else:
                    # for map
                    label_item = torch.zeros(self.num_classes, dtype=torch.float)
                    for lbl_id in rtn["class_labels"]:
                        if isinstance(lbl_id, str):
                            continue
                        label_item[lbl_id] = 1
                    label_item = label_item.unsqueeze(0)
                    rtn["target"] = label_item
                rtn["id"] = index

            except Exception as e:
                print(f"[AudiosetDataset] {e} -- {ann['audio_path']}")
                rtn = dict()
                index = random.randint(0, len(self.annotation) - 1)
                n_retry += 1
                if n_retry > 10:
                    raise ValueError("Exceed max retry.")

        return Sample(rtn)


class AudiosetDataset_ast(Dataset):
    def __init__(
        self,
        dataset_json_file,
        audio_conf,
        label_csv=None,
        use_fbank=False,
        fbank_dir=None,
        roll_mag_aug=False,
        load_video=False,
        mode="train",
        text_processor=None,
        tokenizer=None,
        use_text=False,
        args=None,
    ):
        """
        Dataset that manages audio recordings
        :param audio_conf: Dictionary containing the audio loading and preprocessing settings
        :param dataset_json_file
        """
        self.datapath = dataset_json_file
        with open(dataset_json_file, "r") as fp:
            data_json = json.load(fp)
        self.use_fbank = use_fbank
        self.fbank_dir = fbank_dir

        self.data = data_json["data"]
        self.audio_conf = audio_conf
        print(
            "---------------the {:s} dataloader---------------".format(
                self.audio_conf.get("mode")
            )
        )
        if "multilabel" in self.audio_conf.keys():
            self.multilabel = self.audio_conf["multilabel"]
        else:
            self.multilabel = False
        print(f"multilabel: {self.multilabel}")
        self.melbins = self.audio_conf.get("num_mel_bins")
        self.freqm = self.audio_conf.get("freqm")
        self.timem = self.audio_conf.get("timem")
        print(
            "using following mask: {:d} freq, {:d} time".format(
                self.audio_conf.get("freqm"), self.audio_conf.get("timem")
            )
        )
        self.mixup = self.audio_conf.get("mixup")
        print("using mix-up with rate {:f}".format(self.mixup))
        self.dataset = self.audio_conf.get("dataset")
        self.norm_mean = self.audio_conf.get("mean")
        self.norm_std = self.audio_conf.get("std")
        print(
            "Dataset: {}, mean {:.3f} and std {:.3f}".format(
                self.dataset, self.norm_mean, self.norm_std
            )
        )
        self.noise = self.audio_conf.get("noise")
        if self.noise == True:
            print("now use noise augmentation")
        self.index_dict = make_index_dict(label_csv)

        self.use_text = use_text
        self.mode = mode
        self.args=args
        if use_text and mode == "train" and not args.use_text_branch:
            self.label_map = make_name_dict(label_csv)
            self.audio_text_features = torch.load(args.audio_text_template_path)

        self.label_num = len(self.index_dict)
        self.roll_mag_aug = roll_mag_aug
        print(f"number of classes: {self.label_num}")
        print(f"size of dataset {self.__len__()}")

        self.eval_metric = "mAP"
        self.text_processor = text_processor
        self.tokenizer = tokenizer
        self.init_class_labels()

    def init_class_labels(self):
        self.num_classes = 527
        self.idx2label = []
        self.label2idx = {}

        class_f = pd.read_csv(
            f"{AUDIO_META_DATA_DIR}/audioset_class_labels_indices.csv", header=0
        )
        for i in range(len(class_f)):
            item = class_f.iloc[i]
            assert item["index"] == i
            cls_name = item["display_name"].lower()  # use lower case
            self.idx2label.append(cls_name)
            self.label2idx[cls_name] = i
        assert len(self.idx2label) == self.num_classes

    def _roll_mag_aug(self, waveform):
        waveform = waveform.numpy()
        idx = np.random.randint(len(waveform))
        rolled_waveform = np.roll(waveform, idx)
        mag = np.random.beta(10, 10) + 0.5
        return torch.Tensor(rolled_waveform * mag)

    def _wav2fbank(self, filename, filename2=None):
        if filename2 == None:
            waveform, sr = torchaudio.load(filename)
            waveform = waveform - waveform.mean()
            if self.roll_mag_aug:
                waveform = self._roll_mag_aug(waveform)
        # mixup
        else:
            waveform1, sr = torchaudio.load(filename)
            waveform2, _ = torchaudio.load(filename2)

            waveform1 = waveform1 - waveform1.mean()
            waveform2 = waveform2 - waveform2.mean()

            if self.roll_mag_aug:
                waveform1 = self._roll_mag_aug(waveform1)
                waveform2 = self._roll_mag_aug(waveform2)

            if waveform1.shape[0] > 10000:
                waveform1 = waveform1.view(1, -1)
            if waveform2.shape[0] > 10000:
                waveform2 = waveform2.view(1, -1)

            if waveform1.shape[0] == 1:
                waveform1 = waveform1.repeat(2, 1)
            if waveform2.shape[0] == 1:
                waveform2 = waveform2.repeat(2, 1)

            if waveform1.shape[0] > 2:
                waveform1 = waveform1[:2, :]
            if waveform2.shape[0] > 2:
                waveform2 = waveform2[:2, :]

            if waveform1.shape[1] != waveform2.shape[1]:
                if waveform1.shape[1] > waveform2.shape[1]:
                    # padding
                    temp_wav = torch.zeros(waveform2.shape[0], waveform1.shape[1])
                    temp_wav[:, 0 : waveform2.shape[1]] = waveform2
                    waveform2 = temp_wav
                else:
                    # cutting
                    waveform2 = waveform2[:, 0 : waveform1.shape[1]]

            # sample lambda from beta distribtion
            mix_lambda = np.random.beta(10, 10)
            mix_waveform = mix_lambda * waveform1 + (1 - mix_lambda) * waveform2
            waveform = mix_waveform - mix_waveform.mean()
        # 498 128, 998, 128
        fbank = torchaudio.compliance.kaldi.fbank(
            waveform,
            htk_compat=True,
            sample_frequency=sr,
            use_energy=False,
            window_type="hanning",
            num_mel_bins=self.melbins,
            dither=0.0,
            frame_shift=10,
        )
        # 512
        target_length = self.audio_conf.get("target_length")
        n_frames = fbank.shape[0]

        p = target_length - n_frames

        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:target_length, :]

        if filename2 == None:
            return fbank, 0
        else:
            return fbank, mix_lambda

    def _fbank(self, filename, filename2=None):
        if filename2 == None:
            fn1 = os.path.join(
                self.fbank_dir, os.path.basename(filename).replace(".wav", ".npy")
            )
            fbank = np.load(fn1)
            return torch.from_numpy(fbank), 0
        else:
            fn1 = os.path.join(
                self.fbank_dir, os.path.basename(filename).replace(".wav", ".npy")
            )
            fn2 = os.path.join(
                self.fbank_dir, os.path.basename(filename2).replace(".wav", ".npy")
            )
            # sample lambda from beta distribtion
            mix_lambda = np.random.beta(10, 10)
            fbank = mix_lambda * np.load(fn1) + (1 - mix_lambda) * np.load(fn2)
            return torch.from_numpy(fbank), mix_lambda

    def collater(self, samples):
        return SampleCollator(self, samples)

    def __getitem__(self, index):
        """
        returns: image, audio, nframes
        where image is a FloatTensor of size (3, H, W)
        audio is a FloatTensor of size (N_freq, N_frames) for spectrogram, or (N_frames) for waveform
        nframes is an integer
        """

        rtn = dict()

        # do mix-up for this sample (controlled by the given mixup rate)
        if (
            random.random() < self.mixup
        ):  # for audio_exp, when using mixup, assume multilabel
            datum = self.data[index]
            # find another sample to mix, also do balance sampling
            # sample the other sample from the multinomial distribution, will make the performance worse
            # mix_sample_idx = np.random.choice(len(self.data), p=self.sample_weight_file)
            # sample the other sample from the uniform distribution
            mix_sample_idx = random.randint(0, len(self.data) - 1)
            mix_datum = self.data[mix_sample_idx]

            # get the mixed fbank
            if not self.use_fbank:
                fbank, mix_lambda = self._wav2fbank(datum["wav"], mix_datum["wav"])
            else:
                fbank, mix_lambda = self._fbank(datum["wav"], mix_datum["wav"])
            # initialize the label
            label_indices = np.zeros(self.label_num)
            # add sample 1 labels
            for label_str in datum["labels"].split(","):
                label_indices[int(self.index_dict[label_str])] += mix_lambda
            # add sample 2 labels
            for label_str in mix_datum["labels"].split(","):
                label_indices[int(self.index_dict[label_str])] += 1.0 - mix_lambda
            label_indices = torch.FloatTensor(label_indices)
        # if not do mixup
        else:
            try:
                datum = self.data[index]
            except IndexError:
                print("index out of range")
                print(f"index: {index}")
                return None

            label_indices = np.zeros(self.label_num)
            label_name_idx = []
            if not self.use_fbank:
                fbank, mix_lambda = self._wav2fbank(datum["wav"])
            else:
                fbank, mix_lambda = self._fbank(datum["wav"])
            for label_str in datum["labels"].split(","):
                label_indices[int(self.index_dict[label_str])] = 1.0
                label_name_idx.append(int(self.index_dict[label_str]))

            if self.multilabel:
                label_indices = torch.FloatTensor(label_indices)
            else:
                # remark : for ft cross-ent
                label_indices = int(self.index_dict[label_str])

        rtn["target"] = label_indices
        rtn["id"] = index

        # SpecAug for training (not for eval)
        freqm = torchaudio.transforms.FrequencyMasking(self.freqm)
        timem = torchaudio.transforms.TimeMasking(self.timem)
        fbank = fbank.transpose(0, 1).unsqueeze(0)  # 1, 128, 1024 (...,freq,time)
        if self.freqm != 0:
            fbank = freqm(fbank)
        if self.timem != 0:
            fbank = timem(fbank)  # (..., freq, time)
        fbank = torch.transpose(fbank.squeeze(), 0, 1)  # time, freq
        fbank = (fbank - self.norm_mean) / (self.norm_std * 2)
        if self.noise:  # default is false, true for spc
            fbank = (
                fbank
                + torch.rand(fbank.shape[0], fbank.shape[1]) * np.random.rand() / 10
            )
            fbank = torch.roll(fbank, np.random.randint(-10, 10), 0)
        # the output fbank shape is [time_frame_num, frequency_bins], e.g., [1024, 128]

        rtn["audio"] = fbank

        if self.use_text and self.mode == "train" and not self.args.use_text_branch:
            if self.multilabel:
                text_features = []
                for name in label_name_idx:
                    if self.dataset == "audioset":
                        template_idx = random.randint(0, 15)
                    else:
                        template_idx = random.randint(0, 4)
                    text_features.append(
                        self.audio_text_features[name][template_idx].to(
                            dtype=torch.float16
                        )
                    )
                text_feature = torch.stack(text_features).mean(dim=0)
                # text_feature /= text_feature.norm()

            else:
                template_idx = random.randint(0, 4)
                text_feature = self.audio_text_features[str(label_indices)][
                    template_idx
                ].to(dtype=torch.float16)

            rtn["text_feature"] = text_feature

        caption=[self.idx2label[idx] for idx in label_name_idx]
        
        caption = random.choice(SOUND_AS_IMAGE_TEMPLATE)(caption)  # add templates
        caption = self.text_processor(caption)
        tokenized_caption = self.tokenizer([caption])[0]
        rtn["caption"] = tokenized_caption

        return Sample(rtn)

    def __len__(self):
        return len(self.data)


class AudioCapsDataset(AudioBaseDataset):
    def __init__(
        self,
        vis_processor=None,
        audio_processor=None,
        text_processor=None,
        data_root=AUDIO_DATA_DIR,
        anno_path={
            "val": dict(
                audio=f"{AUDIO_META_DATA_DIR}/audiocaps_val_new.tsv",
                text=f"{AUDIO_META_DATA_DIR}/audiocaps_val_texts.json",
            ),
            "test": dict(
                audio=f"{AUDIO_META_DATA_DIR}/audiocaps_test_new.tsv",
                text=f"{AUDIO_META_DATA_DIR}/audiocaps_test_texts.json",
            ),
            "test_ib": dict(
                audio=f"{AUDIO_META_DATA_DIR}/audiocaps_test_ib.tsv",
                text=f"{AUDIO_META_DATA_DIR}/audiocaps_test_ib_texts.json",
            ),
        },
        args=None,
        split=None,
        tokenizer=None,
        image_transform=None,
        **kwargs,
    ):
        super().__init__(
            vis_processor,
            audio_processor,
            text_processor,
            data_root,
            anno_path,
            args,
            split,
            tokenizer,
            image_transform,
            **kwargs,
        )

        self.annotation = load_annotation(anno_path[split]["audio"], header=0, sep="\t")

        self.text_ids = None
        self.texts = None
        self.init_ret_texts()

        # Evaluation specific
        self.eval_metric = "recall"

    def init_ret_texts(self):
        self.text_ids = []
        self.texts = []
        fn = self.anno_path[self.split]["text"]
        text_infos = load_annotation(fn)
        for text_id, text_list in text_infos.items():
            for text in text_list:
                self.text_ids.append(int(text_id))
                self.texts.append(text)
        # TODO: move `text_ids` to cuda, do when evaluating this task

    def __getitem__(self, index):
        ann = self.annotation.iloc[index]
        uniq_id = int(ann["uniq_id"])

        apath = os.path.join(self.data_root, ann["audio"])
        audio_data = self.audio_processor(apath)

        caption = ann["text"]
        tokenized_caption = self.tokenizer([caption])[0]

        # text_feature = self.audio_text_features[uniq_id].to(dtype=torch.float16)

        return Sample(
            {
                "audio": audio_data,
                "caption": tokenized_caption,
                "uniq_id": uniq_id,
            }
        )


class ClothoDataset(AudioBaseDataset):
    def __init__(
        self,
        vis_processor=None,
        audio_processor=None,
        text_processor=None,
        data_root=AUDIO_DATA_DIR,
        anno_path={
            "val": dict(
                audio=f"{AUDIO_META_DATA_DIR}/clotho_validation_new.tsv",
                text=f"{AUDIO_META_DATA_DIR}/clotho_validation_texts.json",
            ),
            "test": dict(
                audio=f"{AUDIO_META_DATA_DIR}/clotho_evaluation_new.tsv",
                text=f"{AUDIO_META_DATA_DIR}/clotho_evaluation_texts.json",
            ),
        },
        args=None,
        split=None,
        tokenizer=None,
        image_transform=None,
        **kwargs,
    ):
        super().__init__(
            vis_processor,
            audio_processor,
            text_processor,
            data_root,
            anno_path,
            args,
            split,
            tokenizer,
            image_transform,
            **kwargs,
        )

        self.annotation = load_annotation(anno_path[split]["audio"], header=0, sep="\t")

        self.text_ids = None
        self.texts = None
        self.init_ret_texts()

        # Evaluation specific
        self.eval_metric = "recall"

    def init_ret_texts(self):
        self.text_ids = []
        self.texts = []
        fn = self.anno_path[self.split]["text"]
        text_infos = load_annotation(fn)
        for text_id, text_list in text_infos.items():
            for text in text_list:
                self.text_ids.append(int(text_id))
                self.texts.append(text)
        # TODO: move `text_ids` to cuda, do when evaluating this task

    def __getitem__(self, index):
        ann = self.annotation.iloc[index]
        uniq_id = int(ann["uniq_id"])

        apath = os.path.join(self.data_root, ann["audio"])
        audio_data = self.audio_processor(apath)

        caption = ann["text"]
        tokenized_caption = self.tokenizer([caption])[0]

        return Sample(
            {
                "audio": audio_data,
                "caption": tokenized_caption,
                "uniq_id": uniq_id,
            }
        )


class ESC50Dataset(AudioBaseDataset):
    def __init__(
        self,
        vis_processor=None,
        audio_processor=None,
        text_processor=None,
        data_root=AUDIO_DATA_DIR,
        anno_path={
            "val-all": f"{AUDIO_META_DATA_DIR}/esc50_fold-all.json",
            "val-fold-1": f"{AUDIO_META_DATA_DIR}/esc50_fold-1.json",
            "val-fold-2": f"{AUDIO_META_DATA_DIR}/esc50_fold-2.json",
            "val-fold-3": f"{AUDIO_META_DATA_DIR}/esc50_fold-3.json",
            "val-fold-4": f"{AUDIO_META_DATA_DIR}/esc50_fold-4.json",
            "val-fold-5": f"{AUDIO_META_DATA_DIR}/esc50_fold-5.json",
        },
        args=None,
        split=None,
        tokenizer=None,
        image_transform=None,
        **kwargs,
    ):
        super().__init__(
            vis_processor,
            audio_processor,
            text_processor,
            data_root,
            anno_path,
            args,
            split,
            tokenizer,
            image_transform,
            **kwargs,
        )

        self.annotation = load_annotation(anno_path[split])
        self.init_class_labels()

        # Evaluation specific
        self.eval_metric = "acc"

    def init_class_labels(self):
        self.num_classes = 50

        self.idx2label = []
        self.label2idx = {}
        tmp_idx2label = {}

        class_f = load_annotation(f"{AUDIO_META_DATA_DIR}/esc50_label.json")
        for stri, names in class_f.items():
            assert len(names) == 1
            cls_name = names[0].lower()
            self.label2idx[cls_name] = int(stri)
            tmp_idx2label[stri] = cls_name

        assert len(self.label2idx) == self.num_classes

        for i in range(self.num_classes):
            self.idx2label.append(tmp_idx2label[str(i)])
        assert len(self.idx2label) == self.num_classes

    def __getitem__(self, index):
        ann = self.annotation[index]
        uniq_id = int(ann["uniq_id"])

        apath = os.path.join(self.data_root, ann["audio_path"])
        audio_data = self.audio_processor(apath)

        caption = ann["text"]
        tokenized_caption = self.tokenizer([caption])[0]

        label = ann["class_label"]

        return Sample(
            {
                "id": index,
                "audio": audio_data,
                "caption": tokenized_caption,
                "uniq_id": uniq_id,
                "label": label,
            }
        )


class VGGSoundCLSDataset(AudioBaseDataset):
    def __init__(
        self,
        vis_processor=None,
        audio_processor=None,
        text_processor=None,
        data_root=AUDIO_DATA_DIR,
        anno_path={"val": "/home/zhoubo/farm/M2PT/Fuse/data/vggsound/test.json"},
        args=None,
        split=None,
        tokenizer=None,
        image_transform=None,
        **kwargs,
    ):
        super().__init__(
            vis_processor,
            audio_processor,
            text_processor,
            data_root,
            anno_path,
            args,
            split,
            tokenizer,
            image_transform,
            **kwargs,
        )

        self.annotation = load_annotation(anno_path[split])
        self.init_class_labels()

        # Evaluation specific
        self.eval_metric = "acc"

    def init_class_labels(self):
        self.num_classes = 309

        df = pd.read_csv(f"{AUDIO_META_DATA_DIR}/vggsound_stat.csv", header=None)
        self.idx2label = []
        self.label2idx = {}
        for i in range(len(df)):
            item = df.iloc[i]
            cls_name = item[0].strip()
            self.idx2label.append(cls_name)
            self.label2idx[cls_name] = i

        assert len(self.idx2label) == self.num_classes

    def __getitem__(self, index):
        rtn = None
        while rtn is None:
            try:
                ann = self.annotation[index]
                vid = ann["vid"]

                apath = os.path.join(self.data_root, ann["audio_path"])
                audio_data = self.audio_processor(apath)

                caption = random.choice(ann["captions"])
                tokenized_caption = self.tokenizer([caption])[0]

                label = ann["class_labels"]

                rtn = {
                    "id": index,
                    "audio": audio_data,
                    "caption": tokenized_caption,
                    "uniq_id": vid,
                    "label": label,
                }
            except Exception as e:
                print(f"[VGGSound CLS]: {vid} -- {e}")
                rtn = None
                index = random.randint(0, len(self.annotation) - 1)

        return Sample(rtn)

class VGGSoundVideoDataset(AudioBaseDataset):
    def __init__(
        self,
        vis_processor=None,
        audio_processor=None,
        text_processor=None,
        data_root=AUDIO_DATA_DIR,
        anno_path={ "train": "data/vggsound/train.json",
            "val": "/home/zhoubo/farm/M2PT/Fuse/data/vggsound/test.json"},
        args=None,
        split=None,
        tokenizer=None,
        image_transform=None,
        **kwargs,
    ):
        super().__init__(
            vis_processor,
            audio_processor,
            text_processor,
            data_root,
            anno_path,
            args,
            split,
            tokenizer,
            image_transform,
            **kwargs,
        )

        self.annotation = load_annotation(anno_path[split])
        self.init_class_labels()

        # Evaluation specific
        self.eval_metric = "acc"

    def init_class_labels(self):
        self.num_classes = 309

        df = pd.read_csv(f"{AUDIO_META_DATA_DIR}/vggsound_stat.csv", header=None)
        self.idx2label = []
        self.label2idx = {}
        for i in range(len(df)):
            item = df.iloc[i]
            cls_name = item[0].strip()
            self.idx2label.append(cls_name)
            self.label2idx[cls_name] = i

        assert len(self.idx2label) == self.num_classes

    def __getitem__(self, index):
        rtn = None
        while rtn is None:
            try:
                ann = self.annotation[index]
                ann["video_path"] = os.path.join(self.data_root, ann["video_path"])
                ann["audio_path"] = os.path.join(self.data_root, ann["audio_path"])
                if self.is_train and self.load_vision:
                    vis_data, (start, end) = self.vis_processor(ann["video_path"])
                else:
                    vis_data=None
                
                audio_data = self.audio_processor(
                        ann["audio_path"], se=(start, end)
                    )
                
                if vis_data is not None:
                    if (
                        vis_data.ndim == 4 and vis_data.size(1) == self.args.video_num_frames
                    ):  # [ C x T x H x W ] 
                        # vis_data = einops.rearrange(vis_data, "c t h w -> t c h w")
                        # assert vis_data.size(1) == 3  # rgb channel
                        assert vis_data.size(0) == 3  # rgb channel
                        vis_data.squeeze(1)
                    elif vis_data.ndim == 3:
                        assert vis_data.size(0) == 3  # rgb channel
                        # vis_data.unsqueeze(0)
                
                vid = ann["vid"]

                caption = random.choice(ann["captions"])
                tokenized_caption = self.tokenizer([caption])[0]

                label = ann["class_labels"]

                rtn = {
                    "id": index,
                    "audio": audio_data,
                    "video": vis_data,
                    "caption": tokenized_caption,
                    "uniq_id": vid,
                    "label": label,
                }
            except Exception as e:
                print(f"[VGGSound CLS]: {vid} -- {e}")
                rtn = None
                index = random.randint(0, len(self.annotation) - 1)

        return Sample(rtn)

name2dataset = {
    
    "audioset": AudioSetDataset,
    "esc50": ESC50Dataset,
    "clotho": ClothoDataset,
    "vggsound": VGGSoundCLSDataset,
    "vggsound_video": VGGSoundVideoDataset,
    "audiocaps": AudioCapsDataset,
}

nclip_cfg_2 = {
    "audioset": 6,
    "esc50": 3,
    "clotho": 10,
    "vggsound": 6,
    "vggsound_video":6,
    "audiocaps": 6,
}
nclip_cfg_5 = {
    "audioset": 3,
    "esc50": 2,
    "clotho": 6,
    "vggsound": 3,
    "vggsound_video":3,
    "audiocaps": 3,
}
nclip_cfg_8 = {
    "audioset": 2,
    "esc50": 2,
    "clotho": 6,
    "vggsound": 2,
    "vggsound_video":2,
    "audiocaps": 2,
}
nclip_cfg_10 = {
    "audioset": 2,
    "esc50": 2,
    "clotho": 5,
    "vggsound": 2,
    "vggsound_video":2,
    "audiocaps": 2,
}

DURATION2CLIP = {
    "2": nclip_cfg_2,
    "5": nclip_cfg_5,
    "8": nclip_cfg_8,
    "10": nclip_cfg_10,
}


def create_audio_datasets(
    args, is_train, tokenizer, mean=None, std=None, image_transform=None
):
    dataset_names = args.audio_train_data if is_train else args.audio_val_data
    dataset_names = dataset_names.split("::")
   
    datasets = dict()
    for dset_ns in dataset_names:
        
        ns = dset_ns.split("@")
        assert len(ns) == 2
        name, specific_split = ns[0], ns[1]
        dataset_cls = name2dataset[name]
        
        conf = OmegaConf.create(
            {
                "params": {
                    "sampling_rate": args.audio_sampling_rate,
                    "clip_duration": args.audio_clip_duration,
                    "n_clip": DURATION2CLIP[str(int(args.audio_clip_duration))][name],
                    "target_length": args.audio_target_length,
                    "mel_bins": args.audio_mel_bins,
                    "nframes": args.audio_n_frames if name != "vggsound_video" else args.video_num_frames,
                    "freqm": args.freqm,
                    "timem": args.timem,
                    "noise_aug": args.audio_noise_aug,
                    "agg_eval": True,
                },
            }
        )

        vis_proc_cls = None
        if args.audio_load_vision:
            vis_proc_cls = PVProcessorTrain if is_train else None
        audio_proc_cls = ASTProcessorTrain if is_train else ASTProcessorEval

        vis_processor = (
            vis_proc_cls.from_config(cfg=conf) if vis_proc_cls is not None else None
        )
        text_processor = BlipCaptionProcessor.from_config()
        audio_processor = audio_proc_cls.from_config(cfg=conf)
        
        dataset = dataset_cls(
            vis_processor=vis_processor,
            audio_processor=audio_processor,
            text_processor=text_processor,
            args=args,
            split=specific_split,
            tokenizer=tokenizer,
            image_transform=image_transform,
        )

        datasets[name] = dataset

    datasets_list = [v for _, v in datasets.items()]
    if len(datasets) > 1 and is_train:
        
        return datasets
    elif len(datasets) == 1 and is_train:
        return datasets_list[0]
    elif len(datasets) == 1 and not is_train:
        return datasets_list[0]
    else:
        return datasets
