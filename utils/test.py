import dgl
import rdkit
from rdkit import Chem
import numpy as np

def smiles_to_heterogeneous_dgl(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            # Convert the molecule to a DGL graph
            g = dgl.DGLGraph()

            # Add nodes for atoms
            num_atoms = mol.GetNumAtoms()
            g.add_nodes(num_atoms)

            # Add edges for bonds
            for bond in mol.GetBonds():
                start_idx = bond.GetBeginAtomIdx()
                end_idx = bond.GetEndAtomIdx()
                g.add_edge(start_idx, end_idx)
                g.add_edge(end_idx, start_idx)  # Assuming undirected graph

            # Optional: You can add node or edge features if needed
            # For example, you can add node features for atom types, and edge features for bond types.

            # Return the DGL graph
            return g
        else:
            return None
    except Exception as e:
        print(f"Error converting SMILES to DGL graph: {e}")
        return None

# Example usage:
smiles = "CCO"  # Replace with your SMILES string
graph = smiles_to_heterogeneous_dgl(smiles)
if graph is not None:
    print("DGL Graph:", graph)
