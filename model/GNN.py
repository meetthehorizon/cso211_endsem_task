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

class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, g, node_feats, edge_feats):
        g = g.local_var()
        g.ndata['h'] = node_feats
        g.edata['e'] = edge_feats

        def message_func(edges):
            return {'m': edges.src['h']}

        def reduce_func(nodes):
            return {'h': torch.sum(nodes.mailbox['m'], dim=1)}

        g.update_all(message_func=message_func, reduce_func=reduce_func)

        node_feats = g.ndata['h']
        return self.linear(node_feats)

class GraphClassificationModel(nn.Module):
    def __init__(self, in_node_feats, in_edge_feats, hidden_dim, out_dim):
        super().__init__()
        self.gcn1 = GCNLayer(in_node_feats, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, hidden_dim)
        self.gcn3 = GCNLayer(hidden_dim, out_dim)
        self.linear = nn.Linear(in_node_feats, 1) 
    def forward(self, g):
        # Extract node and edge features
        node_feats = g.ndata['features']
        edge_feats = g.edata['attributes']

        # Apply the first GCN layer
        h1 = self.gcn1(g, node_feats, edge_feats)

        # Apply the second GCN layer
        h2 = self.gcn2(g, h1, edge_feats)
        h2 = self.gcn3(g, h2, edge_feats)
        # Calculate the graph embedding by averaging node features
        graph_embedding = dgl.mean_nodes(g, 'features')  # Assuming 'h' is the node feature name

        prediction = self.linear(graph_embedding)
        binary_prediction = torch.sigmoid(prediction)

        return binary_prediction    
        
in_node_feats = train_dataset[0][0].ndata['features'].shape[1] #type: ignore    
in_edge_feats = train_dataset[0][0].edata['attributes'].shape[1] #type: ignore

model = GraphClassificationModel(in_node_feats, in_edge_feats, hidden_dim, out_dim)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

train_loss_coordinates = []
valid_loss_coordinates = []
train_acc_coordinates = []
valid_acc_coordinates = []

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    total_corr = 0
    num_samples = 0  # Track the number of samples processed

    for graph, label in train_dataset:  # Assuming you have a DataLoader
        optimizer.zero_grad()
        prediction = model(graph)
        loss = criterion(prediction, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()  # Accumulate the loss
        total_corr += (prediction.argmax(dim=1) == label).sum().item()  # Accumulate correct predictions
        num_samples += label.size(0)

    # Calculate the average loss and accuracy for the epoch
    avg_loss = total_loss / num_samples
    accuracy = total_corr / num_samples

    print("Epoch {} | Loss: {}".format(epoch + 1, avg_loss, accuracy))

model.eval() 
total_corr = 0

for i in range(len(test_dataset)):
    graph, label = test_dataset[i]
    prediction = model(graph)
    
    total_corr += (label.item() > 0.5) == (prediction.item() > 0.5)

print("Test Accuracy {}".format(total_corr / len(test_dataset)))