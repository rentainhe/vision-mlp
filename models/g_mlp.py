from random import randrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Reduce


# functions
def exists(val):
    return val is not None


def dropout_layers(layers, prob_survival):
    if prob_survival == 1:
        return layers

    num_layers = len(layers)
    to_drop = torch.zeros(num_layers).uniform_(0., 1.) > prob_survival

    # make sure at least one layer makes it
    if all(to_drop):
        rand_index = randrange(num_layers)
        to_drop[rand_index] = False

    layers = [layer for (layer, drop) in zip(layers, to_drop) if not drop]
    return layers


class Attention(nn.Module):
    def __init__(self, dim_in, dim_out, dim_inner):
        super().__init__()
        self.scale = dim_inner ** -0.5

        self.to_qkv = nn.Linear(dim_in, dim_inner * 3, bias=False)
        self.to_out = nn.Linear(dim_inner, dim_out)

    def forward(self, x):
        device = x.device
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale

        attn = sim.softmax(dim=-1)
        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        return self.to_out(out)


class SpatialGatingUnit(nn.Module):
    def __init__(self, dim, dim_seq, attn_dim=None):
        super().__init__()

        self.norm = nn.LayerNorm(dim // 2)
        self.proj = nn.Conv1d(dim_seq, dim_seq, 1)
        self.attn = Attention(dim * 2, dim, attn_dim) if exists(attn_dim) else None
        # initialization mentioned in paper
        nn.init.zeros_(self.proj.weight)
        nn.init.constant_(self.proj.bias, 1.)

    def forward(self, x):
        device = x.device

        res, gate = x.chunk(2, dim=-1)

        gate = self.norm(gate)
        gate = self.proj(gate)

        if exists(self.attn):
            gate += self.attn(x)

        return gate * res

class gMLPBlock(nn.Module):
    def __init__(self, hidden_dim, ffn_dim, num_patches, attn_dim):
        super(gMLPBlock, self).__init__()
        self.gmlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            SpatialGatingUnit(ffn_dim, num_patches, attn_dim),
            nn.Linear(ffn_dim // 2, hidden_dim)
        )

    def forward(self, x):
        return x + self.gmlp(x)


class gMLPVision(nn.Module):
    def __init__(self, num_classes, num_blocks, patch_size, hidden_dim, ffn_dim, attn_dim=None, prob_survival=1.,
                 image_size=224):
        super().__init__()
        assert (image_size % patch_size) == 0, 'image size must be divisible by the patch size'

        num_patches = (image_size // patch_size) ** 2

        self.patch_embedding = nn.Conv2d(3, hidden_dim, kernel_size=patch_size, stride=patch_size)

        self.prob_survival = prob_survival

        self.gmlp_layers = nn.ModuleList([gMLPBlock(hidden_dim, ffn_dim, num_patches, attn_dim) for _ in range(num_blocks)])

        self.to_logits = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            Reduce('b n c -> b c', 'mean'),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        x = self.patch_embedding(x)
        x = x.flatten(2).transpose(1, 2)
        layers = self.gmlp_layers if not self.training else dropout_layers(self.gmlp_layers, self.prob_survival)
        x = nn.Sequential(*layers)(x)
        return self.to_logits(x)

def print_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')

def count_parameters(net):
    params = sum([param.nelement() for param in net.parameters() if param.requires_grad])
    print("Params: %f M" % (params/1000000))

if __name__ == "__main__":
    x = torch.randn(1,3,224,224)
    gmlp_small = gMLPVision(1000, 30, patch_size=16, hidden_dim=128, ffn_dim=768, prob_survival=0.99)
    print(gmlp_small(x).size())
    count_parameters(gmlp_small)
