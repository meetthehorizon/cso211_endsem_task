from networkx import graph_atlas_g
import numpy as np
from rdkit import Chem
import torch, torchvision
from torch_geometric.data import Data
from utils.smiles_utils import get_atom_features, get_bond_features, one_hot_encoding
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
    
# defining transformation classes
class SmilesToGraph:
    def __call__(self, smile, label):
        mol = Chem.MolFromSmiles(smile) # type: ignore

        # get feature dimensions
        n_nodes = mol.GetNumAtoms()
        n_edges = 2*mol.GetNumBonds()
        unrelated_smiles = "O=O"
        unrelated_mol = Chem.MolFromSmiles(unrelated_smiles) # type: ignore
        n_node_features = len(get_atom_features(unrelated_mol.GetAtomWithIdx(0)))
        n_edge_features = len(get_bond_features(unrelated_mol.GetBondBetweenAtoms(0,1)))
        # construct node feature matrix X of shape (n_nodes, n_node_features)
        X = np.zeros((n_nodes, n_node_features))
        for atom in mol.GetAtoms():
            X[atom.GetIdx(), :] = get_atom_features(atom)
            
        X = torch.tensor(X, dtype = torch.float)
        
        # construct edge index array E of shape (2, n_edges)
        (rows, cols) = np.nonzero(GetAdjacencyMatrix(mol)) # type: ignore
        torch_rows = torch.from_numpy(rows.astype(np.int64)).to(torch.long)
        torch_cols = torch.from_numpy(cols.astype(np.int64)).to(torch.long)
        E = torch.stack([torch_rows, torch_cols], dim = 0)
        
        # construct edge feature array EF of shape (n_edges, n_edge_features)
        EF = np.zeros((n_edges, n_edge_features))
        
        for (k, (i,j)) in enumerate(zip(rows, cols)):
            EF[k] = get_bond_features(mol.GetBondBetweenAtoms(int(i),int(j)))
        
        EF = torch.tensor(EF, dtype = torch.float)
        
        # construct label tensor
        y_tensor = torch.tensor(np.array([label]))

        # construct Pytorch Geometric data object and append to data list
        return Data(x = X, edge_index = E, edge_attr = EF, y = y_tensor)


# defining dataset classes
class SmilesDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, smilesColumnName='smiles', labelColumnName='Label', Transform=SmilesToGraph()):
        self.dataset = dataset
        self.smilesColumnName = smilesColumnName
        self.labelColumnName = labelColumnName
        self.Transform = Transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        smile = self.dataset.iloc[index][self.smilesColumnName]
        label = self.dataset.iloc[index][self.labelColumnName]
        graph = None

        #applying transformations
        if self.Transform:
            graph = self.Transform(smile, label)
    
        return graph

if __name__ == '__main__':
    print('passed')
