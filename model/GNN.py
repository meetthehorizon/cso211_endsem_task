# generic support imports
from cgi import test
import numpy as np
import pandas as pd

# dgl imports
import dgl
import dgl.function as fn
from dgl.data import DGLDataset, register_data_args

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
from utils import data

# custom imports
from utils.data import load_dataset, drop_invalid_smile, train_val_split
from utils.pytorch_data import SmilesDataset
from utils.helper import custom_collate

# hyper parameters
num_epochs = 10
hidden_dim = 32
num_layers = 4
out_dim = 2
learning_rate = 0.1

# Step 0 : Pre Processing the Dataset

dataset = load_dataset(r'./data/training_data_bitter.csv', smiles=True)
dataset = drop_invalid_smile(dataset, smilesColumnName='smiles')
test_dataset = load_dataset(r'./data/testing_data_bitter.csv', smiles=True)

train_dataset, valid_dataset = train_val_split(dataset, split_ratio=0.8, random_state=59)

train_dataset, valid_dataset, test_dataset = SmilesDataset(train_dataset), SmilesDataset(valid_dataset), SmilesDataset(test_dataset)

for i in range(train_dataset.__len__()):
    graph, label = train_dataset.__getitem__(i)
    # print(type(graph))

class DiMeNet(nn.Module):
    def __init__(self, num_metapaths, in_dim, hidden_dim, out_dim):
        super(DiMeNet, self).__init__()
        self.metapath_weights = nn.ParameterList([
            nn.Parameter(torch.randn(in_dim, out_dim)) for _ in range(num_metapaths)
        ])
        self.lin = nn.Linear(in_dim, hidden_dim)

    def forward(self, g):
        # Metapath-based aggregation
        metapath_results = []
        for metapath_idx, weight in enumerate(self.metapath_weights):
            g.ndata['h'] = self.lin(g.nodes['node_type'], g.ndata['h'])
            g.update_all(message_func=dgl.function.u_mul_e('h', 'weight', 'm'),
                         reduce_func=dgl.function.sum('m', 'neigh'))
            g.ndata['h'] = torch.matmul(g.ndata['neigh'], weight)
            metapath_results.append(g.ndata['h'])
        
        # Node updates
        node_feats = torch.stack(metapath_results, dim=1).sum(dim=1)
        return node_feats
    