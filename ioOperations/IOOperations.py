'''
    Data read and write operations.
    Created on 15.03.2019

    @author: el-sharkawy
'''
import numpy as np           # Saving, reading and manipulation of arrays as binary files
import pandas as pd          # For reading CSV files
from numpy import float64


def readCSV(path, columnsToRemove=None):
    '''
    Returns the content of the CSV-file as 2D numpy array
    :param path: The path to the CSV-file, including file extension
    :param columnsToRemove: Optional selection of which columns should be removed from the data set (0-based),
                            may be used to remove unnecessary columns
    '''
    csv_data= pd.read_csv(path, sep=";", index_col=False, quotechar="\"")
    data = np.asarray(csv_data)#, dtype=np.float64)
     
    if (columnsToRemove is not None):
        data = np.delete(data, columnsToRemove, 1)

    data = data.astype(float64)
    return data
    
def saveNumpyData(data, path):
    '''
    Stores the given data (numpy array) at the specified location.
    :param data: A numpy array to store.
    :param path: The destination, including file extension.
    '''
    np.save(path, data);

def readNPY(path):
    data = np.load(path)
    
    return data