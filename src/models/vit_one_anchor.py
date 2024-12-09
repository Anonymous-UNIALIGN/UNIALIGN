# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial
import time
from regex import B
import torch
import torch.nn as nn
import torch.nn.functional as F

from knn_cuda import KNN
from clip import model
from util import misc
import timm

from timm.layers import trunc_normal_
from flash_attn.modules.mha import MHA as FlashMHA
from flash_attn.modules.mlp import Mlp as FlashMlp
from timm.models._manipulate import checkpoint_seq

import numpy as np
from torch.utils.checkpoint import checkpoint
from util.pos_embed import build_2d_sincos_posemb
from einops import rearrange

# sin-cos position encoding
# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py#L31
def get_sinusoid_encoding_table(n_position, d_hid):
    """Sinusoid position encoding table"""

    # TODO: make it with torch instead of numpy
    def get_position_angle_vec(position):
        return [
            position / np.power(10000, 2 * (hid_j // 2) / d_hid)
            for hid_j in range(d_hid)
        ]

    sinusoid_table = np.array(
        [get_position_angle_vec(pos_i) for pos_i in range(n_position)]
    )
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.tensor(
        sinusoid_table, dtype=torch.float, requires_grad=False
    ).unsqueeze(0)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class Flash_Block(timm.models.vision_transformer.Block):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        use_flash_attn: bool = True,
        **kwargs,
    ) -> None:
        super(Flash_Block, self).__init__(dim, num_heads, **kwargs)

        self.attn = FlashMHA(
            embed_dim=dim,
            num_heads=num_heads,
            cross_attn=False,
            dropout=0.0,
            use_flash_attn=use_flash_attn,
        )
        mlp_width = int(dim * 4)
        self.mlp = FlashMlp(dim, hidden_features=mlp_width, activation=QuickGELU())


class MoEMlp(timm.layers.Mlp):
    def __init__(self, in_features, hidden_features):
        super(MoEMlp, self).__init__(
            in_features=in_features, hidden_features=hidden_features
        )

    def forward(self, x, modal):
        x = self.fc1(x, modal)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x, modal)
        x = self.drop2(x)
        return x


class Flash_MoE_Block(timm.models.vision_transformer.Block):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        use_flash_attn: bool = True,
        **kwargs,
    ) -> None:
        super(Flash_MoE_Block, self).__init__(dim, num_heads, **kwargs)

        self.attn = FlashMHA(
            embed_dim=dim,
            num_heads=num_heads,
            cross_attn=False,
            dropout=0.0,
            use_flash_attn=use_flash_attn,
        )
        mlp_width = int(dim * 4)
        self.mlp = MoEMlp(dim, mlp_width)

    def forward(self, x: torch.Tensor, modal) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x), modal)))
        return x


class Group(nn.Module):  # FPS + KNN
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz):
        """
        input: B N 3
        ---------------------------
        output: B G M 3
        center : B G 3
        """
        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        center = misc.fps(xyz, self.num_group)  # B G 3
        # knn to get the neighborhood
        _, idx = self.knn(xyz, center)  # B G M
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = (
            torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        )
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(
            batch_size, self.num_group, self.group_size, 3
        ).contiguous()
        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center


class Adapter(nn.Module):
    def __init__(
        self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=False
    ):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.proj_in = nn.Linear(D_features, D_hidden_features)
        self.proj_out = nn.Linear(D_hidden_features, D_features)

    def forward(self, x):
        # x is (BT, HW+1, D)
        xs = self.proj_in(x)
        xs = self.act(xs)
        xs = self.proj_out(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x


class GPT_text_projector(nn.Module):
    def __init__(self, embed_dim, clip_embed_dim, openai_embed_dim):
        super().__init__()
        self.clip_proj = nn.Linear(in_features=embed_dim, out_features=clip_embed_dim)
        self.openai_proj = nn.Linear(
            in_features=clip_embed_dim, out_features=openai_embed_dim
        )
        self.act = nn.GELU()

        trunc_normal_(self.clip_proj.weight, std=0.02)
        trunc_normal_(self.openai_proj.weight, std=0.02)

    def forward(self, x):
        x = self.clip_proj(x)
        x = self.act(x)
        x = self.openai_proj(x)
        return x


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """Vision Transformer with support for global average pooling"""

    def __init__(
        self,
        has_cls_head,
        global_pool_dict,
        distill_feature_dict,
        audio_length,
        mask_2d=True,
        use_custom_patch=False,
        model_type="mae",
        modals=["image", "audio", "point"],
        args=None,
        **kwargs,
    ):
        super(VisionTransformer, self).__init__(**kwargs)

        del (
            self.patch_embed,
            self.norm,
            self.fc_norm,
            self.head,
            self.pos_embed,
            self.cls_token,
        )
        if model_type == "clip":
            self.norm_pre = nn.LayerNorm(self.embed_dim)

        else:
            self.norm_pre = nn.Identity()

        self.has_cls_head = has_cls_head
        self.global_pool_dict = global_pool_dict
        self.distill_modal_list = args.multi_modal_distill_modal_list
        self.distill_feature_dict = distill_feature_dict
        self.patch_embed = nn.ModuleDict()
        self.head = nn.ModuleDict()

        self.pos_embed = nn.ParameterDict()
        self.cls_token = nn.ParameterDict()
        self.norm = nn.ModuleDict()
        self.fc_norm = nn.ModuleDict()

        if args.multi_modal_distill:
            self.connector = nn.ModuleDict()

        
        self.use_modality_adapter = args.use_modality_adapter
        if self.use_modality_adapter:
            self.modal_adapter = nn.ModuleDict()

        self.text_projector = nn.ModuleDict()

        self.logit_scale = nn.ParameterDict()

        self.patch_drop = nn.ModuleDict()

        self.modals = modals
        for modal in self.modals:
            if modal == "image":
                from src.models.tokenizer import Image

                self.patch_embed[modal] = Image.PatchEmbed(embed_dim=self.embed_dim)
                self.cls_token[modal] = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
                self.pos_embed[modal] = nn.Parameter(
                    torch.randn(
                        1, self.patch_embed[modal].num_patches + 1, self.embed_dim
                    )
                    * 0.02
                )
                if has_cls_head[modal]:
                    self.head[modal] = nn.Linear(
                        in_features=self.embed_dim, out_features=self.num_classes
                    )
                if self.global_pool_dict[modal]:
                    self.norm[modal] = nn.Identity()
                    self.fc_norm[modal] = nn.LayerNorm(self.embed_dim)
                else:
                    self.norm[modal] = nn.LayerNorm(self.embed_dim)
                    self.fc_norm[modal] = nn.Identity()

                
                num_prefix_tokens = 1
                if args.patch_drop_rate > 0:
                    self.patch_drop[modal] = timm.layers.PatchDropout(
                        args.patch_drop_rate,
                        num_prefix_tokens=num_prefix_tokens,
                    )
                else:
                    self.patch_drop[modal] = nn.Identity()

            elif modal == "video":
                from src.models.tokenizer import Video
                all_frames = args.video_num_frames * args.video_num_segments
                
                if args.video_2d_patch:
                    self.patch_embed[modal] = Video.PatchEmbed_2D(
                        embed_dim=self.embed_dim,
                    )
                    self.pos_embed[modal] = nn.Parameter(
                        torch.randn(
                            1, self.patch_embed[modal].num_patches + 1, self.embed_dim
                        )
                        * 0.02
                    )
                    self.cls_token[modal] = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
                    num_prefix_tokens = 1
                    
                    self.video_temporal_embedding = nn.Parameter(torch.zeros(1, all_frames, self.embed_dim))
                    
                    self.video_num_frames = all_frames
                    
                    self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
                    
                else:
                    tubelet_size = args.video_tubelet_size
                    self.patch_embed[modal] = Video.PatchEmbed_3D(
                        embed_dim=self.embed_dim,
                        num_frames=all_frames,
                        tubelet_size=tubelet_size,
                    )
                
                    self.pos_embed[modal] = get_sinusoid_encoding_table(
                        self.patch_embed[modal].num_patches, self.embed_dim
                    )
                    num_prefix_tokens = 0
                
                if args.patch_drop_rate > 0:
                    self.patch_drop[modal] = timm.layers.PatchDropout(
                        args.patch_drop_rate,
                        num_prefix_tokens=num_prefix_tokens,
                    )
                else:
                    self.patch_drop[modal] = nn.Identity()

                if has_cls_head[modal]:
                    self.head[modal] = nn.Linear(
                        in_features=self.embed_dim, out_features=args.video_nb_classes
                    )
                if self.global_pool_dict[modal]:
                    self.norm[modal] = nn.Identity()
                    self.fc_norm[modal] = nn.LayerNorm(self.embed_dim)
                else:
                    self.norm[modal] = nn.LayerNorm(self.embed_dim)
                    self.fc_norm[modal] = nn.Identity()

            elif modal == "rgbd":
                from src.models.tokenizer import Depth
                self.depth_in_channels=args.depth_channel
                self.patch_embed[modal] = Depth.DepthTokenizer(in_channels=self.depth_in_channels)
                if self.depth_in_channels==3:
                    self.cls_token[modal] = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
                    
                    self.pos_embed[modal] = nn.Parameter(
                        torch.randn(1, self.patch_embed[modal].num_patches+1, self.embed_dim)
                        * 0.02
                    )
                    
                    num_prefix_tokens = 1
                else:
                    self.pos_embed[modal] = nn.Parameter(
                        torch.randn(1, self.patch_embed[modal].num_patches, self.embed_dim)
                        * 0.02
                    )
                
                if self.global_pool_dict[modal]:
                    self.norm[modal] = nn.Identity()
                    self.fc_norm[modal] = nn.LayerNorm(self.embed_dim)
                else:
                    self.norm[modal] = nn.LayerNorm(self.embed_dim)
                    self.fc_norm[modal] = nn.Identity()

                num_prefix_tokens = 0
                if args.patch_drop_rate > 0:
                    self.patch_drop[modal] = timm.layers.PatchDropout(
                        args.patch_drop_rate,
                        num_prefix_tokens=num_prefix_tokens,
                    )
                else:
                    self.patch_drop[modal] = nn.Identity()

            elif modal == "audio":
                from src.models.tokenizer import Audio

                self.patch_embed[modal] = Audio.PatchEmbed(
                    img_size=(audio_length, 128),
                    patch_size=(16, 16),
                    in_chans=1,
                    embed_dim=self.embed_dim,
                    stride=args.audio_stride,
                )
                self.cls_token[modal] = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
                trunc_normal_(self.cls_token[modal], std=0.02)
                self.mask_2d = mask_2d
                self.use_custom_patch = use_custom_patch
                audio_embed_len = self.patch_embed[modal].num_patches
                self.pos_embed[modal] = nn.Parameter(
                    torch.randn(1, audio_embed_len + 1, self.embed_dim) * 0.02
                )  # fixed sin-cos embedding

                if has_cls_head[modal]:
                    self.audio_num_classes = args.audio_nb_classes
                    self.head[modal] = nn.Linear(
                        self.embed_dim, out_features=self.audio_num_classes
                    )

                self.patch_drop[modal] = nn.Identity()

                if self.global_pool_dict[modal]:
                    self.norm[modal] = nn.Identity()
                    self.fc_norm[modal] = nn.LayerNorm(self.embed_dim)
                else:
                    self.norm[modal] = nn.LayerNorm(self.embed_dim)
                    self.fc_norm[modal] = nn.Identity()

                self.audio_dataset = args.audio_dataset
                

            elif modal == "point":
                from src.models.tokenizer import Point_cloud

                self.patch_embed[modal] = Point_cloud.Encoder(self.embed_dim)
                self.group_divider = Group(
                    num_group=args.pc_num_group, group_size=args.pc_group_size
                )
                self.cls_token[modal] = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
                self.point_cls_pos = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
                trunc_normal_(self.cls_token[modal], std=0.02)
                trunc_normal_(self.point_cls_pos, std=0.02)

                self.pos_embed[modal] = nn.Sequential(
                    nn.Linear(3, 128), nn.GELU(), nn.Linear(128, self.embed_dim)
                )

                num_prefix_tokens = 1
                if args.patch_drop_rate > 0:
                    self.patch_drop[modal] = timm.layers.PatchDropout(
                        args.patch_drop_rate,
                        num_prefix_tokens=num_prefix_tokens,
                    )
                else:
                    self.patch_drop[modal] = nn.Identity()

                if self.global_pool_dict[modal]:
                    self.norm[modal] = nn.Identity()
                    self.fc_norm[modal] = nn.LayerNorm(self.embed_dim)
                else:
                    self.norm[modal] = nn.LayerNorm(self.embed_dim)
                    self.fc_norm[modal] = nn.Identity()

                if has_cls_head[modal]:
                    self.head[modal] = nn.Sequential(
                        nn.Linear(self.embed_dim * 2, 256),
                        nn.BatchNorm1d(256),
                        nn.ReLU(inplace=True),
                        nn.Dropout(0.5),
                        nn.Linear(256, 256),
                        nn.BatchNorm1d(256),
                        nn.ReLU(inplace=True),
                        nn.Dropout(0.5),
                        nn.Linear(256, args.pc_nb_classes),
                    )


            elif modal == "fmri":
                self.patch_embed[modal] = nn.Linear(15724, 8192)
                self.pos_embed[modal] = nn.Parameter(
                    torch.empty([8 + 1, self.embed_dim])
                )
                nn.init.normal_(self.pos_embed[modal], std=0.02)
            elif modal == "imu":
                self.patch_embed[modal] = nn.Conv1d(
                    in_channels=6,
                    out_channels=self.embed_dim,
                    kernel_size=10,
                    bias=False,
                )
                self.pos_embed[modal] = nn.Parameter(
                    torch.empty([391 + 1, self.embed_dim])
                )
                nn.init.normal_(self.pos_embed[modal], std=0.02)

            if modal != "image" and self.use_modality_adapter:
                self.modal_adapter[modal] = Adapter(self.embed_dim)

            if args.text_embed_dim == 1536:
                self.text_projector[modal] = nn.Linear(
                    in_features=self.embed_dim, out_features=1536
                )
            else:
                self.text_projector[modal] = nn.Linear(
                    in_features=self.embed_dim, out_features=512
                )

            if args.multi_modal_distill and modal in self.distill_modal_list:
                self.connector[modal] = timm.layers.Mlp(
                    in_features=self.embed_dim,
                    hidden_features=self.embed_dim*4,
                    out_features=distill_feature_dict[modal],
                )

            self.logit_scale[modal] = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.grad_checkpointing = False

        self.use_flash_attn = args.use_flash_attn
        
        self.moe_type = args.moe_type

        if self.use_flash_attn:
            num_heads = kwargs["num_heads"]
            # embed_dim = kwargs["embed_dim"]
            depth = kwargs["depth"]
            if args.moe_type=='lora_moe_mg':
                self.blocks = nn.Sequential(
                    *[
                        Flash_MoE_Block(
                            dim=self.embed_dim,
                            num_heads=num_heads,
                            use_flash_attn=self.use_flash_attn,
                        )
                        for i in range(depth)
                    ]
                )
            else:
                self.blocks = nn.Sequential(
                    *[
                        Flash_Block(
                            dim=self.embed_dim,
                            num_heads=num_heads,
                            use_flash_attn=self.use_flash_attn,
                        )
                        for i in range(depth)
                    ]
                )

        self.use_orthogonal_loss = args.use_orthogonal_loss
        
        self.use_moe_loss = args.use_moe_loss

        self.apply(self._init_weights)

        if self.use_modality_adapter:
            for n, m in self.named_modules():
                if "adapter" in n:
                    for n2, m2 in m.named_modules():
                        if "proj_out" in n2:
                            print("init adapter")
                            if isinstance(m2, nn.Linear):
                                nn.init.constant_(m2.weight, 0)
                                nn.init.constant_(m2.bias, 0)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv3d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def random_masking_2d(self, x, mask_t_prob, mask_f_prob, audio_dataset="audioset"):
        """
        2D: Spectrogram (masking t and f under mask_t_prob and mask_f_prob)
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """

        N, L, D = x.shape  # batch, length, dim
        if self.use_custom_patch:
            if audio_dataset == "audioset":
                # for AS
                # T = 101  # 64,101
                # F = 12  # 8,12
                T = 50
                F = 12
            elif audio_dataset == "esc50":
                # # for ESC
                T = 50
                F = 12
            elif audio_dataset == "speechcommands":
                # for SPC
                T = 12
                F = 12
            elif audio_dataset == "vgg":
                T = 50
                F = 12
        else:
            if audio_dataset == "audioset":
                # for AS
                T = 64
                F = 8
                
            elif audio_dataset == "esc50":
                # for ESC
                T = 32
                F = 8
            elif audio_dataset == "speechcommands":
                # for SPC
                T = 8
                F = 8
            elif audio_dataset == "vgg":
                T = 50
                F = 12

        # mask T
        x = x.reshape(N, T, F, D)
        len_keep_T = int(T * (1 - mask_t_prob))
        noise = torch.rand(N, T, device=x.device)  # noise in [0, 1]
        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_keep = ids_shuffle[:, :len_keep_T]
        index = ids_keep.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, F, D)
        # x_masked = torch.gather(x, dim=1, index=index)
        # x_masked = x_masked.reshape(N,len_keep_T*F,D)
        x = torch.gather(x, dim=1, index=index)  # N, len_keep_T(T'), F, D

        # mask F
        # x = x.reshape(N, T, F, D)
        x = x.permute(0, 2, 1, 3)  # N T' F D => N F T' D
        len_keep_F = int(F * (1 - mask_f_prob))
        noise = torch.rand(N, F, device=x.device)  # noise in [0, 1]
        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_keep = ids_shuffle[:, :len_keep_F]
        # index = ids_keep.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, T, D)
        index = ids_keep.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, len_keep_T, D)
        x_masked = torch.gather(x, dim=1, index=index)
        x_masked = x_masked.permute(0, 2, 1, 3)  # N F' T' D => N T' F' D
        # x_masked = x_masked.reshape(N,len_keep*T,D)
        x_masked = x_masked.reshape(N, len_keep_F * len_keep_T, D)

        return x_masked, None, None

    def forward_features(
        self, x: torch.Tensor, modal: str, anchor:str, mask_t_prob=0.0, mask_f_prob=0.0
    ) -> torch.Tensor:
        bsz = x.size(0)
        if anchor == "image":
            x = self.patch_embed[anchor](x)
            cls_token = self.cls_token[anchor].expand(bsz, -1, -1)
            x = torch.cat((cls_token, x), dim=1)
            x = x + self.pos_embed[anchor]
        elif anchor == "audio":
            x = self.patch_embed[modal](x)
            x = x + self.pos_embed[modal][:, 1:, :]
            if self.random_masking_2d:
                x, mask, ids_restore = self.random_masking_2d(
                    x, mask_t_prob, mask_f_prob, self.audio_dataset
                )
            else:
                x, mask, ids_restore = self.random_masking(x, mask_t_prob)
            cls_token = self.cls_token[anchor] #+ self.pos_embed[modal][:, :1, :]
            cls_token = cls_token.expand(bsz, -1, -1)
            x = torch.cat((cls_token, x), dim=1)
        elif anchor == "point":
            neighborhood, center = self.group_divider(x)
            group_pc_input_tokens = self.patch_embed[anchor](neighborhood)
            cls_token = self.cls_token[anchor].expand(bsz, -1, -1)
            pc_cls_pos = self.point_cls_pos.expand(bsz, -1, -1)
            pc_pos = self.pos_embed[anchor](center)
            # x = torch.cat((cls_token, img_token, text_token, group_pc_input_tokens), dim=1)
            # pos = torch.cat((pc_cls_pos, img_pos, text_pos, pc_pos), dim=1)
            x_pc = torch.cat((cls_token, group_pc_input_tokens), dim=1)
            pc_pos = torch.cat((pc_cls_pos, pc_pos), dim=1)
            x = x_pc + pc_pos
        elif anchor == "video":
            
            x = self.patch_embed[anchor](x)
            
            x = torch.cat([self.cls_token[anchor].to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
            x = x + self.pos_embed[anchor].to(x.dtype)
            
            n = x.shape[1]
            x = rearrange(x, '(b t) n d -> (b n) t d', t=self.num_frames)
            x = x + self.video_temporal_embedding
            x = rearrange(x, '(b n) t d -> (b t) n d', n=n)
            
        elif anchor == "rgbd":
            x = self.patch_embed[anchor](x)
            if self.depth_in_channels==3:
                cls_token = self.cls_token[anchor].expand(bsz, -1, -1)
                x = torch.cat((cls_token, x), dim=1)
            
            x = x + self.pos_embed[anchor]

        x = self.patch_drop[anchor](x)
        
        if anchor != "image" and self.use_modality_adapter:
            x = x + self.modal_adapter[anchor](x)
        
        x = self.norm_pre(x)
        if self.moe_type=='lora_moe_mg':
            for blk in self.blocks:
                if self.grad_checkpointing and not torch.jit.is_scripting():
                    x = checkpoint(blk, x, modal)
                else:
                    x = blk(x, modal)
        else:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint_seq(self.blocks, x)
            else:
                x = self.blocks(x)

        x = self.norm[anchor](x)
        
        if anchor == "video":
            T = self.num_frames
            B = x.size(0) // T
            x = x[:, 0]
            x = rearrange(x, '(b t) d -> b d t',b=B,t=T)
            x = x.unsqueeze(-1).unsqueeze(-1)  # BDTHW for I3D head
            x = self.avg_pool(x)
            x = x.view(x.shape[0], -1)

        return x

    def forward_head(
        self, x: torch.Tensor, modal: str, pre_logits: bool = False
    ) -> torch.Tensor:
        # if self.attn_pool is not None:
        #     x = self.attn_pool(x)
        if modal == "point":
            x = torch.cat([x[:, 0], x[:, 1:].max(1)[0]], dim=-1)
            # x = torch.cat([x[:, 0], x[:, 1], x[:, 2], x[:, 3:].max(1)[0]], dim=-1)
        elif modal != "audio":
            if self.global_pool_dict[modal]:
                if modal in self.cls_token:
                    x = x[:, 1:].mean(dim=1)
                else:
                    x = x.mean(dim=1)
            else:
                x = x[:, 0]  # class token
        x = self.fc_norm[modal](x)
        # x = self.head_drop(x)
        return x if pre_logits else self.head[modal](x)

    def extract_text_embedding(self, x, modal):
        x = self.text_projector[modal](x, modal)

        return x

    def forward(
        self,
        x_list,
        modal_list,
        anchor_list,
        mask_t_prob=0.0,
        mask_f_prob=0.0,
        extract_feature=False,
    ) -> torch.Tensor:
        logits = {}
        feature_dict = {}
        text_feature_dict = {}
        out_teacher_feature_dict = {}
        logit_scale_dict = {}

        orth_loss = {}
        
        moe_loss = {}
        
        # modal_embed_dict={}
        for i, _ in enumerate(x_list):
            modal = modal_list[i]

            anchor_feature = {}
            
            anchor_teacher_feature = {}
            
            modal_moe_loss=0.0
            
            for anchor in anchor_list:
                x_anchor = x_list[i][anchor]
                x_feature = self.forward_features(
                    x_anchor, modal,anchor, mask_t_prob, mask_f_prob
                )
                if self.has_cls_head[modal] and anchor == modal:
                    x = self.forward_head(x_feature, modal)
                    logits[modal] = x
                
                if anchor != "video":
                    if self.global_pool_dict[anchor]:
                        pooled_faeature = x_feature.mean(dim=1)
                    else:
                        pooled_faeature = x_feature[:, 0, :]
                else:
                    pooled_faeature = x_feature

                if extract_feature :
                    if modal in self.distill_modal_list and anchor == modal:
                        t_visual_features = self.connector[modal](pooled_faeature)
                        t_visual_features = t_visual_features / t_visual_features.norm(
                            dim=-1, keepdim=True
                        )
                        anchor_teacher_feature[anchor] = t_visual_features

                if self.use_orthogonal_loss and anchor != "image":
                    orthogonal_loss = self.orthogonal_loss()
                    orth_loss.update({f"orthogonal_loss_{modal}": orthogonal_loss})
                
                if self.use_moe_loss and anchor == modal:
                    modal_moe_loss += self.load_balance_loss()

                visual_feature = self.text_projector[anchor](pooled_faeature)

                visual_feature = visual_feature / visual_feature.norm(
                    dim=-1, keepdim=True
                )
                visual_feature = visual_feature.to(dtype=torch.float16)
                
                anchor_feature[anchor] = visual_feature
            feature_dict[modal] = anchor_feature
            out_teacher_feature_dict[modal] = anchor_teacher_feature
            
            if self.use_moe_loss:
                moe_loss.update({f'{modal}_moe_loss':modal_moe_loss})

        for m in self.modals:
            logit_scale_dict[m] = self.logit_scale[m].exp()

        return dict(
            logits=logits,
            features=feature_dict,
            text_features=text_feature_dict,
            teacher_features=out_teacher_feature_dict,
            logit_scale=logit_scale_dict,
            moe_loss=moe_loss,
            orth_loss=orth_loss,
        )
    
    def load_balance_loss(self):
        aux_balance_loss_coef=1.0
        # param_dict = {name: param for name, param in self.named_parameters()}
        moe_loss=0.0
        # for name, param in param_dict.items():
        for blk in self.blocks:
            l_aux_1=blk.mlp.fc1.l_aux
            l_aux_2=blk.mlp.fc2.l_aux
            moe_loss = l_aux_1+l_aux_2
        
        return moe_loss * aux_balance_loss_coef

    def orthogonal_loss(self):
        ########################### Regularization ##########################
        orthogonal_loss = 0.0
        lambad_1 = 1.0

        param_dict = {name: param for name, param in self.named_parameters()}

        for name, param in param_dict.items():
            if "lora_A" in name:
                param_A = param

                param_B = param_dict[name.replace("lora_A", "lora_B")]

                lora_refer_A = param_dict[name.replace("lora_A.weight", "lora_refer_A")]

                lora_refer_B = param_dict[name.replace("lora_A.weight", "lora_refer_B")]

                weight = param_dict[name.replace("lora_A.", "")]

                base_ref = lora_refer_A @ lora_refer_B
                loss = torch.pow(
                    weight - base_ref, 2
                ).mean()  # 保证分解矩阵和模型原始权重一样。
                loss += torch.abs(torch.mm(param_A, lora_refer_B.T)).sum()  # A正交
                loss += torch.abs(torch.mm(param_B, lora_refer_A.T)).sum()  # B正交

                orthogonal_loss += loss

        return lambad_1 * orthogonal_loss


def vit_small_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=32,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model
