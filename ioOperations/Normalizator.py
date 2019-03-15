from ioOperations.NormalizationType import NormalizationType
import numpy as np

def normalization(normalization, train_data, test_data, validation_data=None):
    if not isinstance(normalization, NormalizationType):
        print("Unknown normalization specified, use " + str(NormalizationType.PERCENTAGE) + " for normalizing data")
        normalization = NormalizationType.PERCENTAGE;
    
    if (normalization is NormalizationType.NO_NORMALIZATION):
        print("No normalization selected")
    elif (normalization is NormalizationType.PERCENTAGE):
        train_data, test_data, validation_data = __percentageNormalization(train_data, test_data, validation_data)
    elif (normalization is NormalizationType.LOGARITHM):
        train_data, test_data, validation_data= __logarithmNormalization(train_data, test_data, validation_data)
    elif (normalization is NormalizationType.SCALED_LOGARITHM):
        train_data, test_data, validation_data = __logarithmNormalization(train_data, test_data, validation_data)
        train_data, test_data, validation_data = __percentageNormalization(train_data, test_data, validation_data)
    else:
        raise TypeError("Unhandled normalization selected")
    
    return train_data, test_data, validation_data


def __percentageNormalization(train_data, test_data, validation_data=None):
    nColumns = train_data.shape[1] if len(train_data.shape) == 2 else 0;
    train_max = np.amax(train_data, axis=0)
    test_max = np.amax(test_data, axis=0)
    if (validation_data is not None):
        validation_max =  np.amax(validation_data, axis=0)
    else:
        validation_max = np.zeros(nColumns)
    
    max_vector = np.amax([train_max, test_max, validation_max], axis=0)
    
    train_data = train_data/max_vector
    test_data = test_data/max_vector
    
    if (validation_data is not None):
        validation_data = validation_data/max_vector
        
    return train_data, test_data, validation_data;
        
def __logarithmNormalization(train_data, test_data, validation_data=None):
    train_data = np.log1p(train_data)
    test_data = np.log1p(test_data)
    
    if (validation_data is not None):
        validation_data = np.log1p(validation_data)
    
    return train_data, test_data, validation_data;
