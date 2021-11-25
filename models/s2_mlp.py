import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import PatchEmbed
from einops.layers.torch import Rearrange, Reduce


class LayerNorm2d(nn.LayerNorm):
    """ LayerNorm for channels of '2D' spatial BCHW tensors """
    def __init__(self, num_channels):
        super().__init__(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps).permute(0, 3, 1, 2)

class Spatial_Shift(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        b,w,h,c = x.size()
        x[:,1:,:,:c//4] = x[:,:w-1,:,:c//4]
        x[:,:w-1,:,c//4:c//2] = x[:,1:,:,c//4:c//2]
        x[:,:,1:,c//2:c*3//4] = x[:,:,:h-1,c//2:c*3//4]
        x[:,:,:h-1,3*c//4:] = x[:,:,1:,3*c//4:]
        return x

class S2Block(nn.Module):
    def __init__(self, dim, expand_ratio=4, mlp_bias=True):
        super().__init__()

        self.channel_mlp1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=mlp_bias),
            nn.GELU(),
            Spatial_Shift(),
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=mlp_bias),
            LayerNorm2d(dim),
        )
        self.channel_mlp2 = nn.Sequential(
            nn.Conv2d(dim, dim * expand_ratio, kernel_size=1, stride=1, bias=mlp_bias),
            nn.GELU(),
            nn.Conv2d(dim * expand_ratio, dim, kernel_size=1, stride=1, bias=mlp_bias),
            LayerNorm2d(dim)
        )
    
    def forward(self, x):
        x = self.channel_mlp1(x) + x
        x = self.channel_mlp2(x) + x
        return x


class S2MLP(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=384, depth=36, num_classes=1000, expand_ratio=4, mlp_bias=False):
        super().__init__()
        self.patch_emb = PatchEmbed(img_size, patch_size, in_chans, embed_dim, flatten=False)
        self.stages = nn.Sequential(
            *[S2Block(embed_dim, expand_ratio, mlp_bias) for i in range(depth)]
        )
        self.mlp_head = nn.Sequential(
            Reduce('b c h w -> b c', 'mean'),
            nn.Linear(embed_dim, num_classes)
        )
    
    def forward(self, x):
        x = self.patch_emb(x)
        x = self.stages(x)
        out = self.mlp_head(x)
        return out
    