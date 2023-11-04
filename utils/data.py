import re
from tarfile import data_filter
import pandas as pd
import numpy as np
from utils.errors.custom_errors import BothFalseError
from .helper import is_valid_smiles

def load_dataset(load_path, canonicalSmiles=False, smiles=False, smilesName='smiles', canonicalSmilesName='canoncialSmiles', labelName='Label'):
    """function to load dataset
    
    Parameters
    ----------
    load_path : str
        path to a csv file dataset having columns of SMILES, canonicalSmiles and labels
    canonicalSmiles : bool
        if True, return input features as canoncialSmiles
    smiles : bool
        if True, return input features as smiles
    smilesName : str
        column name of smiles
    canonicalSmilesName : str
        column name of canonicalSmiles
    labelName : str
        column name of labels

    Returns
    -------
    dataset : pandas.DataFrame
        datframe with columns smiles, canonicalSmiles and labels
    """

    # loading dataset
    dataset = pd.read_csv(load_path)
    loaded_dataset = pd.DataFrame()

    #both smiles and canoncialSmiles cannot be False
    try:
        BothFalseError.check_booleans(smiles, canonicalSmiles)
    except BothFalseError as e:
        print(e)
        return None

    # assigning input features and labels
    if(smiles):
        loaded_dataset[smilesName] = dataset[smilesName]
    elif(canonicalSmiles):
        loaded_dataset[canonicalSmilesName] = dataset[canonicalSmilesName]
    
    loaded_dataset[labelName] = dataset[labelName].astype(np.float32)

    return loaded_dataset
    
def drop_invalid_smile(dataset, smilesColumnName='SMILES'):
    """function to drop invalid smiles from dataset
    
    Parameters
    ----------
    dataset : pandas.DataFrame
        dataframe with columns smiles, canonicalSmiles and labels
    smilesColumnName : str
        column name of smiles
    
    Returns
    -------
    clean_dataset : pandas.DataFrame
        returns cleaned dataset with only valid SMILES rows
    """

    dataset = dataset[dataset[smilesColumnName].apply(is_valid_smiles)]
    return dataset

def train_val_split(dataset, split_ratio=0.8, random_state=104):
    """splits dataset into train and validation sets
    
    Parameters
    ----------
    dataset : pandas.DataFrame
        dataframe with columns smiles and/or canonicalSmiles and labels
    split_ratio : float
        size of train set (default 0.8)
    
    Returns
    -------
    train_dataset : pandas.DataFrame
        dataframe with columns smiles and/or canonicalSmiles and labels
    val_dataset : pandas.DataFrame
        dataframe with columns smiles and/or canonicalSmiles and labels
    """

    split_index = int(len(dataset) * split_ratio)

    dataset = dataset.sample(frac=1, random_state=random_state).reset_index(drop=True) #shuffling dataset

    train_dataset = dataset.iloc[:split_index].reset_index(drop=True)
    val_dataset = dataset.iloc[split_index:].reset_index(drop=True)

    return train_dataset, val_dataset
    
if __name__ == '__main__':
    
    df = load_dataset(r'./data/training_data_bitter.csv', smiles=True)
    print(df)