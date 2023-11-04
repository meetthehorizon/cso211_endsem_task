from networkx import graph_atlas_g
import numpy as np
from rdkit import Chem
import torch, torchvision
from torch_geometric.data import Data
import dgl
from utils.smiles_utils import get_atom_features, get_bond_features, one_hot_encoding
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="dgl")
# defining transformation classes
class SmilesToGraph():
    def __call__(self, smiles):
        mol = Chem.MolFromSmiles(smiles) # type: ignore

        if mol is None:
            return None

        # Generate atom and bond features
        num_atoms = mol.GetNumAtoms()
        num_bonds = mol.GetNumBonds()


        if mol is None:
            return None

        # Generate atom and bond features
        num_atoms = mol.GetNumAtoms()
        num_bonds = mol.GetNumBonds()

        # Create a DGL graph
        g = dgl.DGLGraph()

        # Add nodes (atoms) to the graph
        g.add_nodes(num_atoms)

        # Extract atom features
        atom_features = []
        for atom in mol.GetAtoms():
            atom_features.append(atom.GetAtomicNum())  # You can extract other features as needed
        g.ndata['feat'] = torch.tensor(atom_features)

        # Add edges (bonds) to the graph
        src_indices = []
        dst_indices = []
        bond_features = []

        for bond in mol.GetBonds():
            src_idx = bond.GetBeginAtomIdx()
            dst_idx = bond.GetEndAtomIdx()
            src_indices.extend([src_idx, dst_idx])
            dst_indices.extend([dst_idx, src_idx])
            bond_features.extend([bond.GetBondTypeAsDouble()] * 2)  # You can extract other features as needed

        g.add_edges(src_indices, dst_indices)

        # Set edge features (bond features)
        g.edata['feat'] = torch.tensor(bond_features)

        return g
        
#defining dataset classes
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
        graph = self.Transform(smile)

        return graph, torch.tensor(label)

if __name__ == '__main__':
    molecule = "CCO"
    
