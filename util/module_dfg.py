from calendar import c
from easydict import EasyDict as edict


def get_peft_cfg(args):
    cfg = edict()
    cfg.moe_type = args.moe_type
    cfg.target_modules = args.lora_trainable.split(",")
    modules_to_save = (
        args.modules_to_save
    )  # not lora paras, but is trainable, i.e., not freeze
    if modules_to_save is not None:
        modules_to_save = modules_to_save.split(",")
    cfg.modules_to_save = modules_to_save
    cfg.r = args.lora_rank
    cfg.lora_dropout = args.lora_dropout
    cfg.lora_alpha = args.lora_alpha

    cfg.lora_nums = args.lora_nums

    lora_modal_list = args.model_modal_list.copy()
    lora_modal_list.pop(0)

    cfg.modal_list = lora_modal_list

    cfg.continue_training = (
        args.continue_train.split(",") if args.continue_train else None
    )
    
    cfg.use_orthogonal_loss=args.use_orthogonal_loss
    
    cfg.topk = args.moe_topk
    
    cfg.expert_nums= args.expert_nums

    return cfg


def get_point_tokenizer_cfg(args):
    cfg = edict()
    cfg.trans_dim = args.pc_trans_dim
    cfg.group_size = args.pc_group_size
    cfg.num_group = args.pc_num_group
    cfg.encoder_dims = args.pc_encoder_dims
    cfg.in_dim = args.pc_in_channel
    return cfg


def get_audio_tokenizer_cfg(args):
    cfg = edict()
    cfg.audio_fstride = args.audio_fstride
    cfg.audio_tstride = args.audio_tstride
    cfg.audio_mel_bins = args.audio_mel_bins
    cfg.audio_target_length = args.audio_target_length
    return cfg


def get_distill_cfg(args):
    cfg = edict()
    cfg.distill = args.multi_modal_distill
    cfg.distill_modal = args.multi_modal_distill_modal_list
    cfg.distill_dim = {
        "image": args.image_distill_dim,
        "audio": args.audio_distill_dim,
        "point": args.point_distill_dim,
        "rgbd": args.rgbd_distill_dim,
        "video": args.video_distill_dim,
    }
    return cfg


def get_head_cfg(args):
    cfg = edict()

    cfg.has_cls_head = args.has_cls_head

    cfg.num_classes = {
        "image": args.image_nb_classes,
        "audio": args.audio_nb_classes,
        "point": args.point_nb_classes,
        "video": args.video_nb_classes,
    }

    return cfg
