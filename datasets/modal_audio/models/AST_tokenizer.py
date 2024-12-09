import logging
from cv2 import log
import torch
import torch.nn as nn

from datasets.Sample import Sample


class AST_tokenizer(nn.Module):
    def __init__(
        self, fstride, tstride, input_fdim, input_tdim, patch_size=(16, 16), width=768
    ):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)

        self.fstride = fstride
        self.tstride = tstride
        self.input_fdim = input_fdim
        self.input_tdim = input_tdim
        self.width = width  # transformer dim

        # tokenize component
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=width,
            kernel_size=patch_size,
            stride=(fstride, tstride),
            bias=True,
        )

        self.f_dim, self.t_dim = self.get_tokenized_dim()
        self.num_patches = self.f_dim * self.t_dim

        scale = width**-0.5
        self.pos_emb = nn.Parameter(scale * torch.randn(self.num_patches + 1, width))
        
        if self.fstride==10:
            self.load_model_from_ckpt("/home/zhoubo/farm/M2PT/Fuse/ckpt/audioset_10_10_0.4593.pth")
        else:
            self.load_model_from_ckpt("/home/zhoubo/farm/M2PT/Fuse/ckpt/audio_pretrained.pth")
    
    def load_model_from_ckpt(self, ckpt_path):
        if self.fstride==10:
            ckpt = torch.load(ckpt_path, map_location="cpu")
        else:
            ckpt = torch.load(ckpt_path, map_location="cpu")['model']
        
        for k, v in ckpt.items():
            
            if 'pos_embed' in k:
                if self.fstride==10:
                    new_pos_embed = v[:, 2:, :].detach().reshape(1, 1212, 768).transpose(1, 2).reshape(1, 768, 12, 101)
                    # if the input sequence length is larger than the original audioset (10s), then cut the positional embedding
                    if self.t_dim < 101:
                        new_pos_embed = new_pos_embed[:, :, :, 50 - int(self.t_dim/2): 50 - int(self.t_dim/2) + self.t_dim]
                    # otherwise interpolate
                    else:
                        new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(12, self.t_dim), mode='bilinear')
                    if self.f_dim < 12:
                        new_pos_embed = new_pos_embed[:, :, 6 - int(self.f_dim/2): 6 - int(self.f_dim/2) + self.f_dim, :]
                    # otherwise interpolate
                    elif self.f_dim > 12:
                        new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(self.f_dim, self.t_dim), mode='bilinear')
                    new_pos_embed = new_pos_embed.reshape(1, 768, self.num_patches).transpose(1, 2)
                    self.pos_emb = nn.Parameter(torch.cat([ckpt['module.v.pos_embed'][:, :1, :].detach(), new_pos_embed], dim=1))
                    
                    logging.info(f"Init positional embedding loaded from {ckpt_path}")
                else:
                    if 'decoder' not in k:
                        new_pos_embed = v[:, 1:, :].detach().transpose(1, 2).reshape(1, 768, 8, 64)
                        if self.t_dim<32:
                            new_pos_embed = new_pos_embed[:, :, :, 32 - int(self.t_dim/2): 32 - int(self.t_dim/2) + self.t_dim]
                        new_pos_embed = new_pos_embed.reshape(1, 768, self.num_patches).transpose(1, 2)
                        self.pos_emb = nn.Parameter(torch.cat([ckpt['pos_embed'][:, :1, :].detach(), new_pos_embed], dim=1))
                    
            elif 'patch_embed' in k:
                
                if 'weight' in k:
                    self.conv1.weight = nn.Parameter(v)
                elif 'bias' in k:
                    self.conv1.bias = nn.Parameter(v)
                
                logging.info(f"Init patch embedding loaded from {ckpt_path}: {k}")

    
    def get_tokenized_dim(self):
        with torch.no_grad():
            test_inp = torch.randn(1, 1, self.input_fdim, self.input_tdim)
            test_outp = self.conv1(test_inp)
            fd = test_outp.shape[2]
            td = test_outp.shape[3]
        return fd, td

    def forward(self, x):
        # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)
        x = self.conv1(x)  # shape = [*, width, fdim, tdim]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, fdim * tdim]
        x = x.permute(0, 2, 1)  # shape = [*, fdim * tdim, width]

        return Sample(
            {
                "x": x,
                "pos": self.pos_emb,
            }
        )
        
