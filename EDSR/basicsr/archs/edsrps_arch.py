import torch,math
from torch import nn as nn
import numpy as np
from basicsr.archs.arch_util import make_layer, default_init_weights
from basicsr.utils.registry import ARCH_REGISTRY
try:
    from .hhps_layers.conv2d_layer import HashConv2d
except:
    from hhps_layers.conv2d_layer import HashConv2d
FIXED_SEED = 725
torch.manual_seed(FIXED_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(FIXED_SEED)
    torch.cuda.manual_seed_all(FIXED_SEED)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
np.random.seed(FIXED_SEED)

param_total =  43089923
param_convlast = 6915
param_other = 19016

class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.

    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, shared_weight=None, num_feat=64, res_scale=1, pytorch_init=False, split_ratio=0, grad_type ='add',):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale

        self.conv1 = HashConv2d(shared_weight, num_feat, num_feat, 3, 1, 1, bias=True, split_ratio=split_ratio, grad_type=grad_type)
        self.conv2 = HashConv2d(shared_weight, num_feat, num_feat, 3, 1, 1, bias=True, split_ratio=split_ratio, grad_type=grad_type)
        self.relu = nn.ReLU(inplace=True)

        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale

class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat, shared_weight=None, split_ratio=0, grad_type ='add',):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(HashConv2d(shared_weight, num_feat, 4 * num_feat, 3, 1, 1, split_ratio=split_ratio, grad_type=grad_type))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(HashConv2d(shared_weight, num_feat, 9 * num_feat, 3, 1, 1, split_ratio=split_ratio, grad_type=grad_type))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


@ARCH_REGISTRY.register()
class EDSRPS(nn.Module):
    """EDSRPS network structure.
    """

    def __init__(self,
                 num_in_ch,
                 num_out_ch,
                 num_feat=64,
                 num_block=16,
                 upscale=4,
                 res_scale=1,
                 img_range=255.,
                 rgb_mean=(0.4488, 0.4371, 0.4040),
                 split_ratio=0,
                 grad_type ='add',
                 cr = 0.001,
                   ):
        super(EDSRPS, self).__init__()
        print("*"*20, f"sparsity is {cr} equals to {1/cr}x compression",  "*"*20 )
        self.img_range = img_range
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        compressable_parameter = param_total - param_other
        shared_weight_size = int(compressable_parameter * cr - param_convlast)
        shared_weight =np.random.uniform(-1/np.sqrt(10000), 1/np.sqrt(10000), size=(shared_weight_size,)).astype(np.float32)   # one hs
        shared_weight = torch.tensor(shared_weight).cuda()
        self.shared_weight = nn.Parameter(shared_weight.clone(), requires_grad=True)

        self.conv_first = HashConv2d(self.shared_weight, num_in_ch, num_feat, 3, 1, 1, split_ratio=split_ratio, grad_type=grad_type)
        self.body = make_layer(ResidualBlockNoBN, num_block, shared_weight=self.shared_weight, num_feat=num_feat,
                                res_scale=res_scale, pytorch_init=True, split_ratio=split_ratio, grad_type=grad_type)

        self.conv_after_body = HashConv2d(self.shared_weight, num_feat, num_feat, 3, 1, 1, split_ratio=split_ratio, grad_type=grad_type)

        self.upsample = Upsample(upscale, num_feat, shared_weight=self.shared_weight, split_ratio=split_ratio, grad_type=grad_type)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1) 
        # parameter statistics
        print("*"*30)
        print('[before compression] compressable_parameter: ', compressable_parameter, '\nshared_weight_size: ', shared_weight_size)
        self.total_params = sum(p.numel() for p in self.parameters())
        print(f"[after compression] params total : {self.total_params:,}")
        print("The first-time inference will be slower due to Triton's autotuning! Subsequent inferences will be better.")
        print("*"*30)


    def forward(self, x):
        self.mean = self.mean.type_as(x)

        x = (x - self.mean) * self.img_range
        x = self.conv_first(x)
        res = self.conv_after_body(self.body(x))
        res += x

        x = self.conv_last(self.upsample(res))
        x = x / self.img_range + self.mean

        return x

if __name__ == '__main__':
    height=128
    width=128
    split_ratio =0.5
    grad_type= 'avg'
    cr=0.01
    model = EDSRPS(num_in_ch=3,
                 num_out_ch=3,
                 num_feat=256,
                 num_block=32,
                 upscale=4,
                 res_scale= 0.1,
                 img_range=255.,
                rgb_mean=(0.4488, 0.4371, 0.4040),
                cr=cr,
                grad_type=grad_type,
                split_ratio=split_ratio)
    print(f"model Total params: {sum(map(lambda x: x.numel(), model.parameters())):,d}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # clean GPU memory
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    model = model.to(device).eval()
    #  Stats GPU memory usage for loading the model
    if device.type == 'cuda':
        model_memory_allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
        print(f'Model loading GPU memory allocated: {model_memory_allocated:.2f} MiB')