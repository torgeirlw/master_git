import torch
from torch import nn
import PIL
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt




class MaskedConv2d(nn.Conv2d):
    #sourc: https://github.com/pilipolio/learn-pytorch/blob/master/201708_IconPixelCNN.ipynb
    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        assert mask_type in {'A', 'B'}
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kW // 2 + (mask_type == 'B'):] = 0
        self.mask[:, :, kH // 2 + 1:] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)




class PixelCNN(nn.Module):
    n_channels = 20
    kernel_size = 3
    padding = 1
    #n_pixels_out = 2 # binary 0/1 pixels
    #nr_mix
    
    def __init__(self, nr_mix):
        super(PixelCNN, self).__init__()
        self.nr_mix = nr_mix
        """self.layers = nn.Sequential(
            MaskedConv2d('A', in_channels=1, out_channels=self.n_channels, kernel_size=self.kernel_size, padding=self.padding, bias=True), nn.BatchNorm2d(self.n_channels), nn.ReLU(True),
            MaskedConv2d('B', self.n_channels, self.n_channels, kernel_size=self.kernel_size, padding=self.padding, bias=True), nn.BatchNorm2d(self.n_channels), nn.ReLU(True),
            MaskedConv2d('B', self.n_channels, self.n_channels//2, kernel_size=self.kernel_size, padding=self.padding, bias=True), nn.BatchNorm2d(self.n_channels), nn.ReLU(True),
            MaskedConv2d('B', self.n_channels//2, self.n_channels, kernel_size=self.kernel_size, padding=self.padding, bias=True), nn.BatchNorm2d(self.n_channels), nn.ReLU(True),
            MaskedConv2d('B', self.n_channels, self.n_channels, kernel_size=self.kernel_size, padding=self.padding, bias=True), nn.BatchNorm2d(self.n_channels), nn.ReLU(True),
            nn.Conv2d(in_channels=self.n_channels, out_channels=3*self.nr_mix, kernel_size=1)
        )"""
        
        self.layer1 = nn.Sequential(MaskedConv2d('A', in_channels=1, out_channels=self.n_channels, kernel_size=self.kernel_size+2, padding=self.padding+1, bias=True), nn.BatchNorm2d(self.n_channels), nn.ReLU(True))
        self.layer2 = nn.Sequential(MaskedConv2d('B', self.n_channels, self.n_channels, kernel_size=self.kernel_size, padding=self.padding, bias=True), nn.BatchNorm2d(self.n_channels), nn.ReLU(True))
        self.layer3 = nn.Sequential(MaskedConv2d('B', self.n_channels, self.n_channels//2, kernel_size=self.kernel_size, padding=self.padding, bias=True), nn.BatchNorm2d(self.n_channels//2), nn.ReLU(True))
        self.layer4 = nn.Sequential(MaskedConv2d('B', self.n_channels//2, self.n_channels, kernel_size=self.kernel_size, padding=self.padding, bias=True), nn.BatchNorm2d(self.n_channels), nn.ReLU(True))
        #self.layer5 = nn.Sequential(MaskedConv2d('B', self.n_channels, self.n_channels, kernel_size=self.kernel_size, padding=self.padding, bias=True), nn.BatchNorm2d(self.n_channels), nn.ReLU(True))
        self.layer6 = nn.Sequential(nn.Conv2d(in_channels=self.n_channels, out_channels=3*self.nr_mix, kernel_size=1))
        
        
        
    def forward(self, x):
        #x = self.layers(x) # shape  = [batch, 3*nr_mix, H, W]
        x = self.layer1(x)
        x = self.layer2(x)  
        #residual = x
        x = self.layer3(x) 
        x = self.layer4(x) 
        
        #x = self.layer5(x) 
        #x = x + residual   
        x = self.layer6(x)

        softmax = nn.Softmax(dim=1)
        softplus = nn.Softplus()
        sigmoid = nn.Sigmoid()

        mu =  x[:, :self.nr_mix, :, :]
        var = x[:, self.nr_mix:2*self.nr_mix, :, :]
        pi_mix = x[:, 2*self.nr_mix:3*self.nr_mix, :, :]
        
        mus = torch.empty(mu.shape)
        mus[:, 0, :, :] = mu[:, 0, :, :]
        for i in range(1, self.nr_mix):
            mus[:, i, :, :] = mus[:, i-1, :, :] + softplus(mu[:, i, :, :])

        mus = sigmoid(mus)
        var = softplus(var)
        var = torch.clamp(var, min=0.00000001)
        pi_mix = softmax(pi_mix)
        
        x = torch.cat((mus, var, pi_mix), dim=1)

        return x





















