import warnings
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
