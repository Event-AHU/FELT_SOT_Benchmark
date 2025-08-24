import numpy as np
import torch.nn as nn

from timm.models.layers import to_2tuple
import torch
import torch.nn.functional as F

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """

    def __init__(self, img_size=256, patch_size=16, in_chans=3, embed_dim=768, norm_layer=False, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        # allow different input size
        # B, C, H, W = x.shape
        # _assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        # _assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


def xcorr_depthwise(x, kernel):  # x, event_x
    """depthwise cross correlation
    """
    batch = kernel.size(0)
    channel = kernel.size(2)
    H = W = int(torch.sqrt(torch.tensor(x.size(1))).item())
    x = x.reshape(1, batch * channel, H, W)
    kernel = kernel.reshape(batch * channel, 1, H, W)
    corr_weight = torch.nn.functional.conv2d(x, kernel, groups=batch * channel)
    # out = out.reshape(batch, H*W, channel)
    out = x * corr_weight
    return out.reshape(batch, H * W, channel)


def xcorr_depthwise(x, kernel):  # x, event_x
    """depthwise cross correlation
    """
    batch = kernel.size(0)
    channel = kernel.size(2)
    H = W = int(np.sqrt(x.size(1)))
    x = x.reshape(1, batch*channel, H, W)
    kernel = kernel.reshape(batch*channel, 1, H, W)
    corr_weight = F.conv2d(x, kernel, groups=batch*channel)
    # out = out.reshape(batch, H*W, channel)
    out = x * corr_weight
    return out.reshape(batch, H*W, channel)