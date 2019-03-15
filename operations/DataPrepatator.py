'''
    Highlevel API to prepare date before passing it to ML algorithms
    Created on 15.03.2019
    @author: el-sharkawy
'''
import numpy as np           # Saving, reading and manipulation of arrays as binary files
import ioOperations.IOOperations as io
import ioOperations.MLDataOperations as converter
from ioOperations.NormalizationType import NormalizationType

def preprocessData(csvPath, destinationFolder, errorIndex=0, columnsToRemove=None, shuffle=True):
    '''
    Used once during the preprocessing phase to convert the data into ML-/Python-conform format.
    :param csvPath:
    :param destinationFolder:
    :param errorIndex: The index of nErrors, i.e., the labels, AFTER columnsToRemove have been removed 
    :param columnsToRemove:
    :param shuffle:
    '''
    print("Read data from: " + csvPath)
    # Already converted to Numpy array
    data = io.readCSV(csvPath, columnsToRemove)
    data = converter.normalizeBinaryLabels(data, errorIndex);
    
    print("Determine good and bad items on " + str(len(data)) + " elements")
    # Split data into two data sets (good vs. bad)
    allGoodOnes, allBadOnes = converter.binarySplit(data, errorIndex)
    
    # Remove nErrors column, see: https://stackoverflow.com/a/34008274
    allGoodOnes = np.delete(allGoodOnes, [errorIndex], 1)
    allBadOnes  = np.delete(allBadOnes, [errorIndex], 1)
    
    if (shuffle):
        print("Shuffle data")
        np.random.shuffle(allGoodOnes)
        np.random.shuffle(allBadOnes)
    
    # Store the two data sets (not balanced so far; this is part of analysis, not of preparation)
    if (not(destinationFolder.endswith("/"))):
        destinationFolder = destinationFolder + "/"
    print("Save data to: " + destinationFolder)
    io.saveNumpyData(allGoodOnes, destinationFolder + "allGoodOnes.npy")
    io.saveNumpyData(allBadOnes, destinationFolder + "allBadOnes.npy")
    
def loadData(goodFile, badFile, normalization=NormalizationType.SCALED_LOGARITHM, balance=True):
    '''
    Used always before starting ML algorithm to load and divided already converted data.
    :param goodFile:
    :param badFile:
    :param normalization:
    :param balance:
    '''
    allGoodOnes = io.readNPY(goodFile)
    allBadOnes = io.readNPY(badFile)
    goodSamples, badSamples = converter.createSamples(allGoodOnes, allBadOnes, normalization, balance)
    data, labels = converter.mergeSamplesAndGenerateLabels(goodSamples, badSamples)
    
    return data, labels