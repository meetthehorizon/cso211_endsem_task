# generic support imports
import numpy as np
import pandas as pd
import dgl

# pytorch imports
import torch 
import torch.nn as nn 
import torch.optim as optim
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

# rdkit imports
from rdkit import Chem
import model

# custom imports
from utils.data import load_dataset, drop_invalid_smile, train_val_split
from utils.pytorch_data import SmilesDataset
from utils.helper import custom_collate

# hyper parameters
num_epochs = 50
batch_size = 32
hidden_dim = 64  
num_layers = 2
num_classes = 1
learning_rate = 0.001

# Step 0 : Pre Processing the Dataset

dataset = load_dataset(r'./data/training_data_bitter.csv', smiles=True)
dataset = drop_invalid_smile(dataset, smilesColumnName='smiles')
test_dataset = load_dataset(r'./data/testing_data_bitter.csv', smiles=True)

train_dataset, valid_dataset = train_val_split(dataset, split_ratio=0.8, random_state=59)

train_dataset, valid_dataset, test_dataset = SmilesDataset(train_dataset), SmilesDataset(valid_dataset), SmilesDataset(test_dataset)
