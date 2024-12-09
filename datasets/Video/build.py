# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import os

from .datasets import RawFrameClsDataset, VideoClsDataset
from .pretrain_datasets import (  # noqa: F401
    DataAugmentationForVideoMAEv2, HybridVideoMAE, VideoMAE,
)


def build_pretraining_dataset(args):
    transform = DataAugmentationForVideoMAEv2(args)
    dataset = VideoMAE(
        root=args.video_data_root,
        setting=args.video_data_path,
        train=True,
        test_mode=False,
        name_pattern=args.video_fname_tmpl,
        video_ext='mp4',
        is_color=True,
        modality='rgb',
        num_segments=1,
        num_crop=1,
        new_length=args.video_num_frames,
        new_step=args.video_sampling_rate,
        transform=transform,
        temporal_jitter=False,
        lazy_init=False,
        num_sample=args.video_num_sample)
    print("Data Aug = %s" % str(transform))
    return dataset


def build_dataset(is_train, test_mode, args):
    if is_train:
        mode = 'train'
        anno_path = os.path.join(args.video_data_path, 'train.csv')
    elif test_mode:
        mode = 'test'
        anno_path = os.path.join(args.video_data_path, 'val.csv')
    else:
        mode = 'validation'
        anno_path = os.path.join(args.video_data_path, 'val.csv')

    if args.video_dataset == 'Kinetics-400':
        if not args.video_sparse_sample:
            dataset = VideoClsDataset(
                anno_path=anno_path,
                data_root=args.video_data_root,
                mode=mode,
                clip_len=args.video_num_frames,
                frame_sample_rate=args.video_sampling_rate,
                num_segment=1,
                test_num_segment=args.video_test_num_segment,
                test_num_crop=args.video_test_num_crop,
                num_crop=1 if not test_mode else 3,
                keep_aspect_ratio=True,
                crop_size=args.video_input_size,
                short_side_size=args.video_short_side_size,
                new_height=256,
                new_width=320,
                sparse_sample=False,
                args=args)
        else:
            dataset = VideoClsDataset(
                anno_path=anno_path,
                data_root=args.video_data_root,
                mode=mode,
                clip_len=1,
                frame_sample_rate=1,
                num_segment=args.video_num_frames,
                test_num_segment=args.video_test_num_segment,
                test_num_crop=args.video_test_num_crop,
                num_crop=1 if not test_mode else 3,
                keep_aspect_ratio=True,
                crop_size=args.video_input_size,
                short_side_size=args.video_short_side_size,
                new_height=256,
                new_width=320,
                sparse_sample=True,
                args=args)
        nb_classes = 400

    elif args.video_dataset == 'Kinetics-600':
        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_root=args.video_data_root,
            mode=mode,
            clip_len=args.video_num_frames,
            frame_sample_rate=args.video_sampling_rate,
            num_segment=1,
            test_num_segment=args.video_test_num_segment,
            test_num_crop=args.video_test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.video_input_size,
            short_side_size=args.video_short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 600

    elif args.video_dataset == 'Kinetics-700':
        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_root=args.video_data_root,
            mode=mode,
            clip_len=args.video_num_frames,
            frame_sample_rate=args.video_sampling_rate,
            num_segment=1,
            test_num_segment=args.video_test_num_segment,
            test_num_crop=args.video_test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.video_input_size,
            short_side_size=args.video_short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 700

    elif args.video_dataset == 'Kinetics-710':
        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_root=args.video_data_root,
            mode=mode,
            clip_len=args.video_num_frames,
            frame_sample_rate=args.video_sampling_rate,
            num_segment=1,
            test_num_segment=args.video_test_num_segment,
            test_num_crop=args.video_test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.video_input_size,
            short_side_size=args.video_short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 710

    elif args.video_dataset == 'SSV2':
        dataset = RawFrameClsDataset(
            anno_path=anno_path,
            data_root=args.video_data_root,
            mode=mode,
            clip_len=1,
            num_segment=args.video_num_frames,
            test_num_segment=args.video_test_num_segment,
            test_num_crop=args.video_test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.video_input_size,
            short_side_size=args.video_short_side_size,
            new_height=256,
            new_width=320,
            filename_tmpl=args.video_fname_tmpl,
            start_idx=args.video_start_idx,
            args=args)

        nb_classes = 174

    elif args.video_dataset == 'UCF101':
        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_root=args.video_data_root,
            mode=mode,
            clip_len=args.video_num_frames,
            frame_sample_rate=args.video_sampling_rate,
            num_segment=1,
            test_num_segment=args.video_test_num_segment,
            test_num_crop=args.video_test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.video_input_size,
            short_side_size=args.video_short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 101

    elif args.video_dataset == 'HMDB51':
        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_root=args.video_data_root,
            mode=mode,
            clip_len=args.video_num_frames,
            frame_sample_rate=args.video_sampling_rate,
            num_segment=1,
            test_num_segment=args.video_test_num_segment,
            test_num_crop=args.video_test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.video_input_size,
            short_side_size=args.video_short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 51

    elif args.video_dataset == 'Diving48':
        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_root=args.video_data_root,
            mode=mode,
            clip_len=args.video_num_frames,
            frame_sample_rate=args.video_sampling_rate,
            num_segment=1,
            test_num_segment=args.video_test_num_segment,
            test_num_crop=args.video_test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.video_input_size,
            short_side_size=args.video_short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 48
    elif args.video_dataset == 'MIT':
        if not args.video_sparse_sample:
            dataset = VideoClsDataset(
                anno_path=anno_path,
                data_root=args.video_data_root,
                mode=mode,
                clip_len=args.video_num_frames,
                frame_sample_rate=args.video_sampling_rate,
                num_segment=1,
                test_num_segment=args.video_test_num_segment,
                test_num_crop=args.video_test_num_crop,
                num_crop=1 if not test_mode else 3,
                keep_aspect_ratio=True,
                crop_size=args.video_input_size,
                short_side_size=args.video_short_side_size,
                new_height=256,
                new_width=320,
                sparse_sample=False,
                args=args)
        else:
            dataset = VideoClsDataset(
                anno_path=anno_path,
                data_root=args.video_data_root,
                mode=mode,
                clip_len=1,
                frame_sample_rate=1,
                num_segment=args.video_num_frames,
                test_num_segment=args.video_test_num_segment,
                test_num_crop=args.video_test_num_crop,
                num_crop=1 if not test_mode else 3,
                keep_aspect_ratio=True,
                crop_size=args.video_input_size,
                short_side_size=args.video_short_side_size,
                new_height=256,
                new_width=320,
                sparse_sample=True,
                args=args)
        nb_classes = 339
    else:
        raise NotImplementedError('Unsupported Dataset')

    assert nb_classes == args.video_nb_classes
    print("Number of the class = %d" % args.video_nb_classes)

    return dataset, nb_classes
