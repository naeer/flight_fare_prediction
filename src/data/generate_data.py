import pandas as pd
import os
import zipfile
import numpy as np

def unzip_all_files(path):
    """
    Function to unzip all zipped files

    Parameters
    ----------
    path : str
        Path to the root directory containing all the zipped files
    """
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.zip'):
                file_path = os.path.join(root, file)
                with zipfile.ZipFile(file_path, 'r') as zip:
                    zip.extractall(root)
                os.remove(file_path)
        print('Finished unzipping files for: ' + str(root))

def load_data_into_dataframe(path):
    """
    Function to load data from all csv files into a dataframe

    Parameters
    ----------
    path : str
        Path to the root directory containing all the csv files
    """
    dfs_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path)
                dfs_list.append(df)
        print('Finished loading files for: ' + str(root))
    final_df = pd.concat(dfs_list, ignore_index=True)
    return final_df