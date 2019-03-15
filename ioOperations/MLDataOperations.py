'''
    Data convert operations with focus on operations required for ML tasks.
    Created on 15.03.2019

    @author: el-sharkawy
'''
import numpy as np # Manipulation of arrays as binary files
from ioOperations.NormalizationType import NormalizationType
import ioOperations.Normalizator as Normalizator

def normalizeBinaryLabels(data, errorIndex = 0):
    data[:, errorIndex][data[:,errorIndex] > 0 ] = 1
    return data

def binarySplit(data, errorIndex):
    allBadOnes = data[data[:,errorIndex] > 0 ]
    allGoodOnes = data[data[:,errorIndex] == 0 ]
    
    return allGoodOnes, allBadOnes

def createSamples(allGoodOnes, allBadOnes, normalization=NormalizationType.SCALED_LOGARITHM, balance=True):
    # Considers all values from both data sets and scales them consistently to a [0;1] scale
    scaledGoodOnes, scaledBadOnes, nonExisting = Normalizator.normalization(normalization, allGoodOnes, allBadOnes)
    
    if (balance):
        nRows = min(scaledGoodOnes.shape[0], scaledBadOnes.shape[0])
        # See: https://stackoverflow.com/a/14262743
        scaledGoodOnes = scaledGoodOnes[np.random.randint(scaledGoodOnes.shape[0], size=nRows), :]
        scaledBadOnes = scaledBadOnes[np.random.randint(scaledBadOnes.shape[0], size=nRows), :]
        
    return scaledGoodOnes, scaledBadOnes

def mergeSamplesAndGenerateLabels(goodSamples, badSamples):
    goodLabels = np.ones(len(goodSamples))
    badLabels = np.zeros(len(badSamples))
    data = np.concatenate((goodSamples, badSamples))
    labels = np.concatenate((goodLabels, badLabels))
    
    return data, labels