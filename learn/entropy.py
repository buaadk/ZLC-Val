# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from math import log

from math import log2
def calcShannonEnt(dataSet):
    length,dataDict=float(len(dataSet)),{}
    for data in dataSet:
        try:dataDict[data]+=1
        except:dataDict[data]=1
    return sum([-d/length*log2(d/length) for d in list(dataDict.values())])

print(calcShannonEnt(['A','B','C','D','E']))

