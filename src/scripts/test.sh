export CUDA_VISIBLE_DEVICES=0,1,2,3

audio_pretrained_path=/path/to/audio/checkpoint
point_pretrained_path=/path/to/point/checkpoint

ckpt=/path/to/ckpt

video_pretrained_path=/path/to/video/checkpoint

rgbd_pretrained_path=/path/to/rgbd/checkpoint

audio_dataset=audioset
audio_train_data=audioset@unbalanced_train
audio_val_data=audioset@val
audioset=unbal
audio_topk=1536
audio_distill_dim=1536
audio_logits_path=/path/to/audio/logits
audio_logits_name=/path/to/audio/logits

audio_n_frames=1
audio_stride=10

if [[ $audio_dataset == 'audioset' ]]; then
    if [[ $audioset == 'unbal' ]]; then
        audio_data_train=/path/to/audio/data
    else
        audio_data_train=/path/to/audio/data
    fi
    audioset_train_weight=/path/to/audio/weight
    audio_data_eval=/path/to/audio/data
    audio_label_csv=/path/to/audio/label
    audio_nb_classes=527
    
    freqm=12
    timem=48
    audio_target_length=512
elif [[ $audio_dataset == 'speechcommands' ]]; then
    audioset_train_weight=/path/to/audio/weight

    audio_data_train=/path/to/audio/data
    audio_data_eval=/path/to/audio/data
    audio_label_csv=/path/to/audio/label
    audio_nb_classes=35
    freqm=48
    timem=48
    audio_target_length=128
else
    echo "Invalid audio dataset: $audio_dataset"
    exit 1
fi

audiocaps_text_path=/path/to/audio/caps/text

point_train_data=shapenet
point_val_data=scanobjectnn::modelnet40
point_train_data_prompt=shapenet_64
point_val_data_prompt=modelnet40_64

pc_train_dataset=shapenet55
pc_nb_classes=55
pc_root_path=/path/to/pc/root
pc_data_path=/path/to/pc/data
pc_image_data_path=/path/to/pc/image/data
pc_text_data_path=/path/to/pc/text/data
pc_dataset_n_points=8192
pc_n_points=8192
pc_num_group=512

pc_topk=512
point_distill_dim=512
pc_logits_path=/path/to/pc/logits
pc_logits_name=/path/to/pc/logits
pc_text_logits_name=/path/to/pc/logits
pc_image_logits_name=/path/to/pc/logits

video_train_data=msrvtt
video_val_data=msrvtt
video_train_csv=/path/to/video/train/csv
video_val_csv=/path/to/video/val/csv
video_features_path=/path/to/video/features
video_max_words=77
video_feature_framerate=1
video_slice_framepos=2


video_dataset=msrvtt 
video_nb_classes=101

video_data_path=/path/to/video/data

video_data_root=/path/to/video/data
video_short_side_size=224
video_num_frames=12
video_sampling_rate=4
video_test_num_segment=5
video_test_num_crop=3


video_logits_path=/path/to/video/logits
video_logits_name=/path/to/video/logits
video_topk=768
video_distill_dim=768


rgbd_train_data=sun-rgbd
rgbd_val_data=sun-rgbd::nyu-depth-v2-val2
rgbd_logits_path=/path/to/rgbd/logits
rgbd_distill_dim=768
rgbd_topk=768
rgbd_logits_name=/path/to/rgbd/logits
rgbd_text_logits_name=/path/to/rgbd/logits
rgbd_image_logits_name=/path/to/rgbd/logits
depth_channel=1

moe_type=
lora_rank=4
lora_alpha=32
lora_trainable="fc1,fc2"
lora_dropout=0.05
lora_nums=2

expert_nums=1

moe_topk=1

text_embed_dim=512

if [ "$text_embed_dim" = "1536" ]; then
    img_text_feature_path=/path/to/img/text/feature
    audio_text_feature_path=/path/to/audio/text/feature
    point_text_feature_path=/path/to/point/text/feature
    video_text_feature_path=/path/to/video/text/feature
    rgbd_train_text_feature_path=/path/to/rgbd/text/feature
    rgbd_sunrgbd_val_text_feature_path=/path/to/rgbd/text/feature
    rgbd_nyu_val1_text_feature_path=/path/to/rgbd/text/feature
    rgbd_nyu_val2_text_feature_path=/path/to/rgbd/text/feature

    audio_text_template_path=/path/to/audio/text/template
    point_text_template_path=/path/to/point/text/template
    video_text_template_path=/path/to/video/text/template
    rgbd_text_template_path=/path/to/rgbd/text/template


else
    img_text_feature_path=/path/to/img/text/feature
    audio_text_feature_path=/path/to/audio/text/feature
    point_text_feature_path=/path/to/point/text/feature
    video_text_feature_path=/path/to/video/text/feature
    rgbd_train_text_feature_path=/path/to/rgbd/text/feature
    rgbd_sunrgbd_val_text_feature_path=/path/to/rgbd/text/feature
    rgbd_nyu_val1_text_feature_path=/path/to/rgbd/text/feature
    rgbd_nyu_val2_text_feature_path=/path/to/rgbd/text/feature

    audio_text_template_path=/path/to/audio/text/template
    point_text_template_path=/path/to/point/text/template
    video_text_template_path=/path/to/video/text/template
    rgbd_text_template_path=/path/to/rgbd/text/template

fi

exp_name=
log_dir=./work_dir/$exp_name

python -m torch.distributed.launch --nproc_per_node=4 src/train/pretrain_one_moe_anchor.py \
    --accum_iter 4 \
    --num_workers 16 \
    --batch_size 256  \
    --model vit_base_patch16 \
    --finetune $ckpt \
    --output_dir $log_dir \
    --log_dir $log_dir \
    --log_name $exp_name \
    --epochs 100 \
    --blr 2e-4 --layer_decay 0.65 \
    --base_batchsize 512 \
    --clip_grad 1.0 \
    --weight_decay 0.05 --drop_path 0.1 --mixup 0 --cutmix 0 --reprob 0.25 \
    --audio_pretrained_path $audio_pretrained_path \
    --point_pretrained_path $point_pretrained_path \
    --rgbd_pretrained_path $rgbd_pretrained_path \
    --img_data_path  \
    --pc_root_path $pc_root_path \
    --pc_data_path $pc_data_path \
    --pc_nb_classes $pc_nb_classes \
    --pc_train_dataset $pc_train_dataset \
    --pc_image_data_path $pc_image_data_path \
    --pc_text_data_path $pc_text_data_path \
    --pc_dataset_n_points $pc_dataset_n_points \
    --pc_n_points $pc_n_points \
    --pc_logits_path $pc_logits_path \
    --pc_topk $pc_topk \
    --pc_logits_name $pc_logits_name \
    --pc_text_logits_name $pc_text_logits_name \
    --pc_image_logits_name $pc_image_logits_name \
    --point_distill_dim $point_distill_dim \
    --seed 0 \
    --distributed \
    --audio_dataset $audio_dataset \
    --audio_data_train $audio_data_train \
    --audio_data_eval $audio_data_eval \
    --audio_label_csv $audio_label_csv \
    --audio_weight_csv $audioset_train_weight \
    --audio_nb_classes $audio_nb_classes \
    --audio_topk $audio_topk \
    --audio_distill_dim $audio_distill_dim \
    --freqm $freqm \
    --timem $timem \
    --audio_target_length $audio_target_length \
    --audio_logits_path $audio_logits_path \
    --audio_logits_name $audio_logits_name \
    --audio_noise_aug \
    --audio_load_vision \
    --audio_stride $audio_stride \
    --audio_n_frames $audio_n_frames \
    --use_custom_patch \
    --roll_mag_aug True \
    --mask_2d True \
    --mask_t_prob 0.2 \
    --mask_f_prob 0.2 \
    --video_dataset $video_dataset \
    --video_data_root $video_data_root \
    --video_data_path $video_data_path \
    --video_nb_classes $video_nb_classes \
    --video_short_side_size $video_short_side_size \
    --video_num_frames $video_num_frames \
    --video_sampling_rate $video_sampling_rate \
    --video_test_num_segment $video_test_num_segment \
    --video_test_num_crop $video_test_num_crop \
    --video_pretrained_path $video_pretrained_path \
    --video_train_csv $video_train_csv \
    --video_val_csv $video_val_csv \
    --video_features_path $video_features_path \
    --video_max_words $video_max_words \
    --video_feature_framerate $video_feature_framerate \
    --video_slice_framepos $video_slice_framepos \
    --video_train_data $video_train_data \
    --video_val_data $video_val_data \
    --loose_type \
    --img_weight 1.0 \
    --audio_weight 1.0 \
    --pc_weight 1.0 \
    --video_weight 1.0 \
    --rgbd_weight 1.0 \
    --batch_mode use_one \
    --img_rep_w 0.2 \
    --audio_rep_w 0.25 \
    --pc_rep_w 5 \
    --video_rep_w 1 \
    --rgbd_rep_w 50 \
    --concat  \
    --moe_type ${moe_type} \
    --lora_rank ${lora_rank} \
    --lora_alpha ${lora_alpha} \
    --lora_nums ${lora_nums} \
    --lora_trainable ${lora_trainable} \
    --lora_dropout ${lora_dropout} \
    --expert_nums ${expert_nums} \
    --moe_topk ${moe_topk} \
    --frozen_backbone \
    --train_modal_list  \
    --eval_modal_list   \
    --model_modal_list \
    --use_text \
    --use_text_template \
    --text_embed_dim $text_embed_dim \
    --img_text_feature_path $img_text_feature_path \
    --audio_text_feature_path $audio_text_feature_path \
    --point_text_feature_path $point_text_feature_path \
    --video_text_feature_path $video_text_feature_path \
    --rgbd_train_text_feature_path $rgbd_train_text_feature_path \
    --rgbd_sunrgbd_val_text_feature_path $rgbd_sunrgbd_val_text_feature_path \
    --rgbd_nyu_val1_text_feature_path $rgbd_nyu_val1_text_feature_path \
    --rgbd_nyu_val2_text_feature_path $rgbd_nyu_val2_text_feature_path \
    --audio_text_template_path $audio_text_template_path \
    --point_text_template_path $point_text_template_path \
    --video_text_template_path $video_text_template_path \
    --rgbd_text_template_path $rgbd_text_template_path \
    --audio_train_data $audio_train_data \
    --audio_val_data $audio_val_data \
    --point_train_data $point_train_data \
    --point_val_data $point_val_data \
    --point_train_data_prompt $point_train_data_prompt \
    --point_val_data_prompt $point_val_data_prompt \
    --rgbd_train_data $rgbd_train_data \
    --rgbd_val_data $rgbd_val_data \
    --rgbd_logits_path $rgbd_logits_path \
    --rgbd_distill_dim $rgbd_distill_dim \
    --rgbd_topk $rgbd_topk \
    --rgbd_logits_name $rgbd_logits_name \
    --rgbd_text_logits_name $rgbd_text_logits_name \
    --rgbd_image_logits_name $rgbd_image_logits_name \
    --depth_channel $depth_channel \
    --use_depth_only \
    --uni_align \
    --cross_align \
    --dist_eval \
    --use_flash_attn \
    --use_pc_image \
    --save_best \
    --use_aux_cls_loss \
    --use_clip_loss \
    --gather_with_grad \
    --local_loss \
    --use_text_branch \
    --video_2d_patch \
    --use_peft \
    --multi_modal_distill \
    --multi_modal_distill_modal_list \
    --eval \