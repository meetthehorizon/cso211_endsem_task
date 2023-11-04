import warnings
import dgl
from torch import int32
from numpy import number
from rdkit import Chem
from rdkit import RDLogger

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')

def is_valid_smiles(smilesString):
    """checks whether smilesString is valid or not"""
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            mol = Chem.MolFromSmiles(smilesString) # type: ignore
            return mol is not None
    except Exception as e:
        return False

if __name__ == '__main__':
    print('passed')

def custom_collate(batch):
    return batch

def convert_torch_geometric_data_Data_to_dgl_graph(d):
    # Extract node features, edge connections, and edge attributes
    node_features = d.x
    edge_index = d.edge_index
    edge_attributes = d.edge_attr

    # Create a DGL Graph
    graph = dgl.graph(
        (edge_index[0], edge_index[1]),
        num_nodes=d.num_nodes,
        idtype=int32,  # Change the data type if needed
    )

    # Set node features and edge attributes in the DGL graph
    graph.ndata['features'] = node_features
    graph.edata['attributes'] = edge_attributes

    return graph