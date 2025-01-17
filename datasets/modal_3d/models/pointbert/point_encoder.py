import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from datasets.modal_3d.models.pointbert.dvae import Group
from datasets.modal_3d.models.pointbert.dvae import Encoder
# from open_clip.modal_3d.models.pointbert.logger import print_log
import logging
from datasets.modal_3d.models.pointbert.checkpoint import (
    get_missing_parameters_message,
    get_unexpected_parameters_message,
)
from timm.models.layers import trunc_normal_
from datasets.Sample import Sample


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    """Transformer Encoder without hierarchical structure"""

    def __init__(
        self,
        embed_dim=768,
        depth=4,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
    ):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=drop_path_rate[i]
                    if isinstance(drop_path_rate, list)
                    else drop_path_rate,
                )
                for i in range(depth)
            ]
        )

    def forward(self, x, pos):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)
        return x


class PointTransformer(nn.Module):
    def __init__(self, config, output_dim=None):
        super().__init__()
        self.config = config

        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.drop_path_rate = config.drop_path_rate
        self.num_heads = config.num_heads

        self.group_size = config.group_size
        self.num_group = config.num_group
        # grouper
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        # define the encoder
        self.encoder_dims = config.encoder_dims
        self.encoder = Encoder(encoder_channel=self.encoder_dims)
        # bridge encoder and transformer
        self.reduce_dim = nn.Linear(self.encoder_dims, self.trans_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128), nn.GELU(), nn.Linear(128, self.trans_dim)
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
        )

        self.norm = nn.LayerNorm(self.trans_dim)

        self.do_cat = config.do_cat
        cat_factor = 2 if self.do_cat else 1

        self.proj = None
        if output_dim is not None:
            self.proj = nn.Parameter(
                torch.empty(cat_factor * self.trans_dim, output_dim)
            )
            nn.init.normal_(self.proj, std=output_dim**-0.5)

    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()

    def get_loss_acc(self, pred, gt, smoothing=True):
        gt = gt.contiguous().view(-1).long()

        if smoothing:
            eps = 0.2
            n_class = pred.size(1)

            one_hot = torch.zeros_like(pred).scatter(1, gt.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_prb = F.log_softmax(pred, dim=1)

            loss = -(one_hot * log_prb).sum(dim=1).mean()
        else:
            loss = self.loss_ce(pred, gt.long())

        pred = pred.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))

        return loss, acc * 100

    def load_model_from_ckpt(self, bert_ckpt_path):
        ckpt = torch.load(bert_ckpt_path)
        base_ckpt = {k.replace("module.", ""): v for k, v in ckpt["base_model"].items()}
        for k in list(base_ckpt.keys()):
            if k.startswith("transformer_q") and not k.startswith(
                "transformer_q.cls_head"
            ):
                base_ckpt[k[len("transformer_q.") :]] = base_ckpt[k]
            elif k.startswith("base_model"):
                base_ckpt[k[len("base_model.") :]] = base_ckpt[k]
            del base_ckpt[k]

        incompatible = self.load_state_dict(base_ckpt, strict=False)

        if incompatible.missing_keys:
            logging.info("missing_keys", )
            logging.info(
                get_missing_parameters_message(incompatible.missing_keys),
                
            )
        if incompatible.unexpected_keys:
            logging.info("unexpected_keys", )
            logging.info(
                get_unexpected_parameters_message(incompatible.unexpected_keys),
               
            )

        logging.info(
            f"[Transformer] Successful Loading the ckpt from {bert_ckpt_path}",
            
        )

    def forward(self, pts):
        # divide the point cloud in the same form. This is important
        neighborhood, center = self.group_divider(pts)
        # encoder the input cloud blocks
        group_input_tokens = self.encoder(neighborhood)  # B G N
        group_input_tokens = self.reduce_dim(group_input_tokens)
        # prepare cls
        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)
        # add pos embedding
        pos = self.pos_embed(center)
        # final input
        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        # transformer
        x = self.blocks(x, pos)
        x = self.norm(x)
        concat_f = (
            torch.cat([x[:, 0], x[:, 1:].max(1)[0]], dim=-1) if self.do_cat else x[:, 0]
        )
        # ret = self.cls_head_finetune(concat_f)
        if self.proj is not None:
            concat_f = concat_f @ self.proj
        return concat_f


# For tokenize everything for CLIP
class PointTokenizer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.trans_dim = config.trans_dim
        self.group_size = config.group_size
        self.num_group = config.num_group
        # grouper
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        # define the encoder
        self.encoder_dims = config.encoder_dims
        self.encoder = Encoder(encoder_channel=self.encoder_dims, input_dim=config.in_dim)
        # bridge encoder and transformer
        # self.reduce_dim = nn.Linear(self.encoder_dims, self.trans_dim)

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128), nn.GELU(), nn.Linear(128, self.trans_dim)
        )

        self.point_cls_pos = nn.Parameter(torch.zeros(1, 1, self.encoder_dims))
        trunc_normal_(self.point_cls_pos, std=0.02)
        
        # self.load_model_from_ckpt(config.point_pretrain_path)
        

    def load_model_from_ckpt(self, point_pretrain_path):
        ckpt = torch.load(point_pretrain_path, map_location="cpu")
        base_ckpt = {k.replace("module.model.", ""): v for k, v in ckpt["base_model"].items()}
        for k in list(base_ckpt.keys()):
            if k.startswith("embed"):
                new_k=k.replace("embed","encoder")
                base_ckpt[new_k]=base_ckpt[k]
                del base_ckpt[k]
            elif k.startswith('pos_embed'):
                base_ckpt[new_k]=base_ckpt[k]
            else:
                del base_ckpt[k]

        incompatible = self.load_state_dict(base_ckpt, strict=False)

        if incompatible.missing_keys:
            logging.info("missing_keys", )
            logging.info(
                get_missing_parameters_message(incompatible.missing_keys),
                
            )
        if incompatible.unexpected_keys:
            logging.info("unexpected_keys", )
            logging.info(
                get_unexpected_parameters_message(incompatible.unexpected_keys),
                
            )
        logging.info(
            f"[PointTokenizer] Successful Loading the ckpt from {point_pretrain_path}",
            
        )

    def forward(self, pts):
        # divide the point cloud in the same form. This is important
        neighborhood, center = self.group_divider(pts)

        # encoder the input cloud blocks
        group_input_tokens = self.encoder(neighborhood)  # B G N
        # group_input_tokens = self.reduce_dim(group_input_tokens)

        # add pos embedding
        pos = self.pos_embed(center)
        point_cls_pos = self.point_cls_pos.expand(group_input_tokens.size(0), -1, -1)
        pos = torch.cat((point_cls_pos, pos), dim=1)

        # final input
        return Sample({"x": group_input_tokens, "pos": pos})
        # return group_input_tokens
