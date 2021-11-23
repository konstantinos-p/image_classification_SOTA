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


class ResNetBlock(nn.Module):
  '''
  A Resnet block from the original He, Kaining, et al. paper. Takes
  k x d x d  as inputs and applies two 3x3 convolutions with k channels.
  We adopt batch normalization (BN) right after each convolution and
  before activation. In the original paper there is no dropout.
  '''
  def __init__(self, channels):
    super(ResNetBlock, self).__init__()
    self.conv1 = nn.Conv2d(channels,channels,3,padding='same')
    self.conv2 = nn.Conv2d(channels,channels,3,padding='same')
    self.relu1 = nn.ReLU()
    self.relu2 = nn.ReLU()
    self.batch_norm1 = nn.BatchNorm2d(channels)
    self.batch_norm2 = nn.BatchNorm2d(channels)
    self.batch_norm3 = nn.BatchNorm2d(channels)

  def forward(self,x):
    x1 = self.relu1(self.batch_norm1(self.conv1(x)))
    x2 = self.batch_norm2(self.conv2(x1))
    return  self.relu2(self.batch_norm3(x+x2))


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class ResNetSuperBlock(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(ResNetSuperBlock, self).__init__()
        self.layers = clones(layer, N)

    def forward(self, x):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x)
        return x


class Pool_Expand(nn.Module):
    '''
    Pool the layer inputs and expand across channels.
    '''
    def __init__(self):
        super(Pool_Expand, self).__init__()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.pool(x)
        return F.pad(x, (0, 0, 0, 0, 0, x.shape[1]), "constant", 0)
        # return x


def make_model():
    '''
    Define the full model using this function.
    '''
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = nn.Sequential(
        nn.Conv2d(3, 64, 7, padding='same'),
        nn.MaxPool2d(2),
        nn.MaxPool2d(2),
        ResNetSuperBlock(copy.deepcopy(ResNetBlock(64)), 1),
        Pool_Expand(),
        ResNetSuperBlock(copy.deepcopy(ResNetBlock(128)), 1),
        Pool_Expand(),
        ResNetSuperBlock(copy.deepcopy(ResNetBlock(256)), 1),
        Pool_Expand(),
        ResNetSuperBlock(copy.deepcopy(ResNetBlock(512)), 1),
        nn.Flatten(),
        nn.Linear(25088, 200)
    )

    model.to(device)

    return model

