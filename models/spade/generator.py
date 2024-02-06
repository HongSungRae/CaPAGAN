import torch
from torch.nn import Module, Linear, Conv2d
from torch.nn.utils import spectral_norm
from torch.nn.functional import tanh, interpolate 
from .spade_resblk import SPADEResBlk

class SPADEGenerator(Module):
    def __init__(self, args):
        super().__init__()
        self.linear = Linear(args.gen_input_size, args.gen_hidden_size)
        self.spade_resblk1 = SPADEResBlk(args, 1024, 1024, True)
        self.spade_resblk2 = SPADEResBlk(args, 1024, 1024, True)
        self.spade_resblk3 = SPADEResBlk(args, 1024, 1024, True)
        self.spade_resblk4 = SPADEResBlk(args, 1024, 512, True)
        self.spade_resblk5 = SPADEResBlk(args, 512, 256, True)
        self.spade_resblk6 = SPADEResBlk(args, 256, 128, True)
        self.spade_resblk7 = SPADEResBlk(args, 128, 64, True)
        self.conv = spectral_norm(Conv2d(64, 3, kernel_size=(3,3), padding=1))

    def forward(self, x, seg):
        b, _, _, _ = seg.size()
        h, w = 4, 4
        x = self.linear(x)
        x = x.view(b, -1, 4, 4) # (b,1024,4,4)

        x = interpolate(self.spade_resblk1(x, seg), size=(2*h, 2*w), mode='nearest') #; print(f'\n block1 : {torch.mean(x).item()}') # (b,1024,8,8)
        x = interpolate(self.spade_resblk2(x, seg), size=(4*h, 4*w), mode='nearest') #; print(f'\n block2 : {torch.mean(x).item()}')# (b,1024,16,16)
        x = interpolate(self.spade_resblk3(x, seg), size=(8*h, 8*w), mode='nearest') #; print(f'\n block3 : {torch.mean(x).item()}')# (b,1024,32,32)
        x = interpolate(self.spade_resblk4(x, seg), size=(16*h, 16*w), mode='nearest') #; print(f'\n block4 : {torch.mean(x).item()}')# (b,512,64,64)
        x = interpolate(self.spade_resblk5(x, seg), size=(32*h, 32*w), mode='nearest') #; print(f'\n block5 : {torch.mean(x).item()}')# (b,256,128,128)
        x = interpolate(self.spade_resblk6(x, seg), size=(64*h, 64*w), mode='nearest') #; print(f'\n block6 : {torch.mean(x).item()}')# (b,128,256,256)
        x = interpolate(self.spade_resblk7(x, seg), size=(128*h, 128*w), mode='nearest') #; print(f'\n block7 : {torch.mean(x).item()}')# (b,64,512,512)
        
        x = tanh(self.conv(x)) # (b,3,512,512)

        return x