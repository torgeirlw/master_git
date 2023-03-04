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

class MLP_Mix_prior(nn.Module):#define neighborhood size
    '''
    Multilayer Perceptron.
    '''
    def __init__(self, first_layer_size, num_mixtures):
        super().__init__()
        self.first_layer_size = first_layer_size
        self.num_mixtures = num_mixtures
        self.layers = nn.Sequential(
        nn.Flatten(),
        nn.Linear(self.first_layer_size, 100),
        nn.ReLU(),
        #nn.LeakyReLU(),
        nn.Linear(100, 100),
        nn.ReLU(),
        #nn.LeakyReLU(),
        nn.Linear(100, 100),
        nn.ReLU(), ## leaky relu
        #nn.LeakyReLU(),
        nn.Linear(100, 100),
        nn.ReLU(), ## leaky relu
        nn.Linear(100, 3*self.num_mixtures),#,
        #nn.ReLU()
        #nn.Sigmoid()
        
        )
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        '''Forward pass'''
        out = self.layers(x)

        softplus = nn.Softplus()
        sigmoid = nn.Sigmoid()

        mu = torch.reshape(out[:,0:self.num_mixtures],(out.shape[0],self.num_mixtures))
        mu1 = torch.reshape(mu[:,0],  (out.shape[0], 1))
        mu2 = mu1 + torch.reshape(softplus(mu[:,1]), (out.shape[0], 1))
        mu = torch.cat((mu1,mu2), dim=1)
        mu = sigmoid(mu) #- 0.5

        sig = torch.reshape(softplus(out[:,1*self.num_mixtures:2*self.num_mixtures]),(out.shape[0],self.num_mixtures)) #sig squared
        sig = torch.clamp(sig, 0.00000001)

        pi_mix = torch.reshape(out[:,2*self.num_mixtures:3*self.num_mixtures],(out.shape[0],self.num_mixtures))
        pi_mix = self.softmax(pi_mix)

        out = torch.cat((mu,sig,pi_mix),dim=1)

        return out
















