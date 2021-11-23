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

# noinspection PyUnresolvedReferences
from models import efficientnet
# noinspection PyUnresolvedReferences
from scripts import setup_TinyImagenet, generate_dataloader, train_with_tensorboard, test_with_tensorboard



cwd = os.getcwd()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Define main data directory
DATA_DIR = 'tiny-imagenet-200'  # Original images come in shapes of [3,64,64]
# Define training and validation data paths
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VALID_DIR = os.path.join(DATA_DIR, 'val')

val_img_dir = setup_TinyImagenet(DATA_DIR,TRAIN_DIR,VALID_DIR)


# Define transformation sequence for image pre-processing
# If not using pre-trained model, normalize with 0.5, 0.5, 0.5 (mean and SD)
# If using pre-trained ImageNet, normalize with mean=[0.485, 0.456, 0.406],
# std=[0.229, 0.224, 0.225])

preprocess_transform_pretrain = T.Compose([
                T.Resize(256), # Resize images to 256 x 256
                T.CenterCrop(224), # Center crop image
                T.RandomHorizontalFlip(),
                T.ToTensor(),  # Converting cropped images to tensors
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
])


# Define batch size for DataLoaders
batch_size = 64

# Create DataLoaders for pre-trained models (normalized based on specific requirements)
train_loader_pretrain = generate_dataloader(TRAIN_DIR, "train",batch_size,use_cuda,
                                  transform=preprocess_transform_pretrain)

val_loader_pretrain = generate_dataloader(val_img_dir, "val",batch_size,use_cuda,
                                 transform=preprocess_transform_pretrain)

model_EfficientNet_v3 = efficientnet.make_model_v3()
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs2')
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_EfficientNet_v3.parameters(),lr=1e-3)

epochs = 3
for t in range(epochs):
    print("Epoch "+ str(t+1) +"\n-----------------------")
    train_with_tensorboard(train_loader_pretrain,model_EfficientNet_v3,loss_fn,optimizer,writer,t)
    test_with_tensorboard(val_loader_pretrain, model_EfficientNet_v3,loss_fn,writer,t,len(train_loader_pretrain))
print("Done!")