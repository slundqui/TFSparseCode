import matplotlib
matplotlib.use('Agg')
from dataObj.image import cifarObj
from tf.ista import ISTA
#from plot.roc import makeRocCurve
import numpy as np
import pdb
import os
import matplotlib.pyplot as plt

#Input vgg file for preloaded weights
trainImageLists =  "/home/slundquist/mountData/datasets/cifar/images/train.txt"
testImageLists = "/home/slundquist/mountData/datasets/cifar/images/test.txt"

#Get object from which tensorflow will pull data from
trainDataObj = cifarObj(trainImageLists, resizeMethod="pad")
#testDataObj = cifarObj(testImageLists, resizeMethod="pad")

#ISTA params
params = {
    #Base output directory
    'outDir': "/home/slundquist/mountData/tfLCA/",
    #Inner run directory
    'runDir': "/cifar_batch16_stride2/",
    'tfDir': "/tfout",
    'ckptDir': '/checkpoints/',
    'saveFile': '/save-model',
    #Flag for loading weights from checkpoint
    'load': True,
    'loadFile': "/home/slundquist/mountData/tfLCA/saved/cifar_batch16_stride2.ckpt",
    'numIterations': 1000000,
    'displayPeriod': 200,
    'savePeriod': 100, #In terms of displayPeriod
    #output plots directory
    'plotDir': "plots/",
    'plotPeriod': 20, #With respect to displayPeriod
    #Progress step (also controls how often to save and write out to tensorboard)
    'progress': 300,

    #####ISTA PARAMS######
    'batchSize': 32,
    #Learning rate for optimizer
    'learningRateA': 1e-3,
    'learningRateW': 1e-3,
    #Lambda in energy function
    'thresh': .015,
    #Number of features in V1
    'numV': 128,
    #Stride of V1
    'VStride': 2,
    #Patch size
    'patchSizeY': 12,
    'patchSizeX': 12,
    #Progress step (also controls how often to save and write out to tensorboard)
    'progress': 200,
}

#Allocate tensorflow object
tfObj = ISTA(params, trainDataObj)
print "Done init"

tfObj.runModel()
print "Done run"

tfObj.closeSess()

