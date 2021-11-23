# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime as dt

import torch
from torch import optim, nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision.utils import make_grid
from torchvision import transforms as T
from torchvision import datasets
#import opendatasets as od
import os
from random import randint
import urllib
import zipfile
import copy
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


class MBConv(nn.Module):
    '''
    An Inverted Bottleneck Module. Takes
    k x d x d  as inputs and applies first 1x1 convolutions with t channels,
    this turns the input to tk x d x d. Then it applies a depthwise convolution
    with a kernel of 3x3 dimensions. This keeps the output at tk x d x d.
    We apply Batch Normalization (BN) and Relu after the first 2 transformations and
    BN after the last transformation but before the residual connection.
    '''

    def __init__(self, input_c, expand_t, k):
        super(MBConv, self).__init__()
        self.conv_exp = nn.Conv2d(input_c, expand_t * input_c, 1, padding='same')
        self.conv_depthwise = nn.Conv2d(expand_t * input_c, expand_t * input_c, k, padding='same',
                                        groups=expand_t * input_c)
        self.conv_red = nn.Conv2d(expand_t * input_c, input_c, 1, padding='same')

        self.relu_exp = nn.ReLU()
        self.relu_conv = nn.ReLU()

        self.batch_norm_exp = nn.BatchNorm2d(expand_t * input_c)
        self.batch_norm_conv = nn.BatchNorm2d(expand_t * input_c)
        self.batch_norm_red = nn.BatchNorm2d(input_c)

    def forward(self, x1):
        x = self.conv_exp(x1)
        x = self.batch_norm_exp(x)
        x = self.relu_exp(x)

        x = self.conv_depthwise(x)
        x = self.batch_norm_conv(x)
        x = self.relu_conv(x)

        x = self.conv_red(x)
        x = self.batch_norm_red(x)

        return x + x1

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Pool_Expand(nn.Module):
    '''
    A layer that creates copies of smaller layers, and a layer
    that pools featuremaps, then expands the number of channels by zero padding.
    '''
    def __init__(self, exp_dims, pooling):
        super(Pool_Expand, self).__init__()
        self.pool = nn.MaxPool2d(2)
        self.exp_dims = exp_dims
        self.pooling = pooling

    def forward(self, x):
        if self.pooling == True:
            x = self.pool(x)
        return F.pad(x, (0, 0, 0, 0, 0, self.exp_dims - x.shape[1]), "constant", 0)
        # return x


class LayerSuperBlock(nn.Module):
    '''
    The above blocks into a superblock of concecutive smaller blocks.
    Core encoder is a stack of N layers
    '''
    def __init__(self, layer, N):
        super(LayerSuperBlock, self).__init__()
        self.layers = clones(layer, N)

    def forward(self, x):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x)
        return x


class Final_Layer(nn.Module):
    '''
    Implementation of the final layer of EfficientNet.
    An expansion layer then a global average pooling followed by
    a linear classification layer.
    '''
    def __init__(self, exp_dims, input_ch):
        super(Final_Layer, self).__init__()
        self.conv_exp = nn.Conv2d(input_ch, exp_dims, 1, padding='same')
        self.pool = nn.AvgPool2d(7, stride=None, padding=0)
        self.flatten = nn.Flatten()
        self.linear_layer = nn.Linear(exp_dims, 200)

    def forward(self, x):
        x = self.conv_exp(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.linear_layer(x)

        return x



def make_model():
    '''
    Function that returns the full model.
    '''

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding='same'),
        nn.BatchNorm2d(16),
        nn.MaxPool2d(2),
        LayerSuperBlock(copy.deepcopy(MBConv(16, 1, 3)), 1),
        Pool_Expand(24, pooling=False),
        LayerSuperBlock(copy.deepcopy(MBConv(24, 1, 3)), 2),
        Pool_Expand(40, pooling=True),
        LayerSuperBlock(copy.deepcopy(MBConv(40, 1, 5)), 2),
        Pool_Expand(80, pooling=True),
        LayerSuperBlock(copy.deepcopy(MBConv(80, 1, 3)), 3),
        Pool_Expand(112, pooling=True),
        LayerSuperBlock(copy.deepcopy(MBConv(112, 1, 5)), 3),
        Pool_Expand(192, pooling=False),
        LayerSuperBlock(copy.deepcopy(MBConv(192, 1, 5)), 4),
        Pool_Expand(320, pooling=True),
        LayerSuperBlock(copy.deepcopy(MBConv(320, 1, 3)), 1),
        Final_Layer(1280, 320)
    )

    model.to(device)

    return model