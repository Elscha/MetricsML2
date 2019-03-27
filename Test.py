from ioOperations.NormalizationType import NormalizationType
import ioOperations.Normalizator as Normalizator
import ioOperations.MLDataOperations as converter
import numpy as np
import operations.DataPrepatator as DataPrepatator
import operations.BinaryClassification as BinaryClassification


# columnSlice = (slice(None), slice(4,349,1));
# 
# x = np.array([
#              ["path","func","lines","nErrors", "metric1", "metric2"],
#              ["path2","func2","lines2","nErrors2", "metric1-2", "metric2-2"],
#              ["path3","func3","lines3","nErrors3", "metric1-3", "metric2-3"]
#              ])
#  
# print(x)
# x = x[columnSlice]
# print(x)
# columnSlice = [0,1,2]
DataPrepatator.preprocessData("data/RelevantMetricsData.csv", "data", 0, columnsToRemove=None, shuffle=True)
# data, labels = DataPrepatator.loadData("data/allGoodOnes.npy", "data/allBadOnes.npy")
# BinaryClassification.binaryClassification(data, labels, [256, 128, 32], lrate=0.1, nEpochs=1000, kSplitt=10, plotName="Visualization_(256-128-32)_(0.1)-(1000)-")



# (slice(None), slice(4,349,1))