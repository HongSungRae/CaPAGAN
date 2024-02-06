import torch
from torch.nn import Module, Conv2d
from torch.nn.functional import relu
from torch.nn.utils import spectral_norm
from .spade import SPADE

class SPADEResBlk(Module):
    def __init__(self, args, k_in, k_out, skip=False):
        super().__init__()
        kernel_size = args.spade_resblk_kernel
        self.skip = skip

        
        if self.skip:
            self.spade1 = SPADE(args, k_in)
            self.conv1 = Conv2d(k_in, k_in, kernel_size=(kernel_size, kernel_size), padding=1, bias=False)
            self.spade_skip = SPADE(args, k_in)
            self.conv_skip = Conv2d(k_in, k_out, kernel_size=(kernel_size, kernel_size), padding=1, bias=False)
        else:
            self.spade1 = SPADE(args, k_in)
            self.conv1 = Conv2d(k_in, k_in, kernel_size=(kernel_size, kernel_size), padding=1, bias=False)

        self.spade2 = SPADE(args, k_in)
        self.conv2 = Conv2d(k_in, k_out, kernel_size=(kernel_size, kernel_size), padding=1, bias=False)
    
    def forward(self, x, seg):
        x_skip = x
    
        x = relu(self.spade1(x, seg))
        x = self.conv1(x)
        x = relu(self.spade2(x, seg))
        x = self.conv2(x)

        if self.skip:
            x_skip = relu(self.spade_skip(x_skip, seg))
            x_skip = self.conv_skip(x_skip)
        
        return x_skip + x