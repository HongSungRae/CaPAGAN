import torch
from torch.nn import Module, Conv2d
from torch.nn.utils import spectral_norm
from torch.nn.functional import interpolate, relu
import math


import warnings
warnings.filterwarnings("ignore")

class SPADE(Module):
    def __init__(self, args, k):
        super().__init__()
        num_filters = args.spade_filter
        kernel_size = args.spade_kernel
        self.eps = 1e-9
        self.conv = spectral_norm(Conv2d(1, num_filters, kernel_size=(kernel_size, kernel_size), padding=1))
        self.conv_gamma = spectral_norm(Conv2d(num_filters, k, kernel_size=(kernel_size, kernel_size), padding=1))
        self.conv_beta = spectral_norm(Conv2d(num_filters, k, kernel_size=(kernel_size, kernel_size), padding=1))
        self.normalization = torch.nn.SyncBatchNorm(k, affine=False)

    def forward(self, x, seg):
        N, C, H, W = x.size()
        #print(f'x : {torch.min(x).item()}, {torch.max(x).item()}')

        # 원본
        # sum_channel = torch.sum(x.reshape(N, C, H*W), dim=-1)
        # mean = sum_channel / (N*H*W+self.eps) ; print(f'mean : {torch.min(mean).item()}, {torch.max(mean).item()}')
        # std = torch.sqrt((sum_channel**2 - mean**2) / (N*H*W+self.eps)) ; print(f'std : {torch.min(std).item()}, {torch.max(std).item()}')

        # mean = torch.unsqueeze(torch.unsqueeze(mean, -1), -1)
        # std = torch.unsqueeze(torch.unsqueeze(std, -1), -1)
        # x = (x - mean) / (std+self.eps) ; print(f'x_normed : {torch.min(x).item()}, {torch.max(x).item()}')

        # 내가
        # sum_channel = torch.sum(x,dim=(0,2,3)) ; print(f'sum_channel : {torch.min(sum_channel).item()}, {torch.max(sum_channel).item()}')
        # mean = sum_channel / (N*H*W)
        # std = torch.sqrt(torch.nn.functional.relu((sum_channel**2-mean**2) / (N*H*W)))

        # mean = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(mean, -1), -1), 0) ; print(f'mean : {torch.min(mean).item()}, {torch.max(mean).item()}')
        # std = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(std, -1), -1), 0) ; print(f'std : {torch.min(std).item()}, {torch.max(std).item()}')
        # x = (x - mean) / (std + self.eps) ; print(f'x_normed : {torch.min(x).item()}, {torch.max(x).item()}')
       
        x = self.normalization(x)# ; print(f'x_normed : {torch.min(x).item()}, {torch.max(x).item()}')

        seg = interpolate(seg, size=(H,W), mode='nearest')
        seg = relu(self.conv(seg))
        seg_gamma = self.conv_gamma(seg)# ; print(f'gamma : {torch.min(seg_gamma).item()}, {torch.max(seg_gamma).item()}\n')
        seg_beta = self.conv_beta(seg)# ; print(f'beta : {torch.min(seg_beta).item()}, {torch.max(seg_beta).item()}\n')
        
        # matmul = torch.matmul(seg_gamma, x) ; print(f'matmul : {torch.min(matmul).item()}, {torch.max(matmul).item()}\n')
        matmul = x*seg_gamma# ; print(f'matmul : {torch.min(matmul).item()}, {torch.max(matmul).item()}\n')
        x = matmul + seg_beta

        # if math.isnan(torch.mean(matmul).item()):
        #     print(torch.mean(x).item())
        #     print(torch.mean(matmul).item())
       
        return x