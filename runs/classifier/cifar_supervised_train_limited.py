import matplotlib
matplotlib.use('Agg')
from dataObj.image import cifarObj
from tf.supervised_control import Supervised
import numpy as np
import pdb

#Paths to list of filenames
trainList =  "/home/slundquist/mountData/datasets/cifar/images/train.txt"
testList =  "/home/slundquist/mountData/datasets/cifar/images/test.txt"

#Get object from which tensorflow will pull data from
trainDataObj = cifarObj(trainList, resizeMethod="crop", shuffle=True, skip=1, seed=None, getGT=True, rangeIdx=range(128))
testDataObj = cifarObj(testList, resizeMethod="crop", shuffle=True, skip=1, seed=None, getGT=True)

params = {
    #Base output directory
    'outDir':          "/home/slundquist/mountData/tfSparseCode/",
    #Inner run directory
    'runDir':          "/cifar_sup_256_limited/",
    'tfDir':           "/tfout",
    #Save parameters
    'ckptDir':         "/checkpoints/",
    'saveFile':        "/save-model",
    'savePeriod':      100, #In terms of displayPeriod
    #output plots directory
    'plotDir':         "plots/",
    'plotPeriod':      200, #With respect to displayPeriod
    #Progress step
    'progress':        10,
    #Controls how often to write out to tensorboard
    'writeStep':       100, #300,
    #Flag for loading weights from checkpoint
    'load':            False,
    'loadFile':        "/home/slundquist/mountData/DeepGAP/saved/cifar.ckpt",
    #Device to run on
    'device':          '/gpu:1',
    #Num iterations
    'numIterations':   500,
    'displayPeriod':   100,
    #Batch size
    'batchSize':       4,
    #Learning rate for optimizer
    'learningRate':    1e-4,
    'numClasses':      10,

    'epsilon': 1e-8,

    'regularizer': 'none',
    'regWeight': .3,

    #####ISTA PARAMS######
    'VStrideY':        2,
    'VStrideX':        2,
    'patchSizeX':      12,
    'patchSizeY':      12,
    'numV':            256,
    #####New encode parapms#####
    'maxPool':         True, #Controls max or avg pool

    'preTrain': False,
}

#Allocate tensorflow object
#This will build the graph
tfObj = Supervised(params, trainDataObj)

print "Done init"
tfObj.runModel(trainDataObj, testDataObj = testDataObj, numTest=256)
print "Done run"

tfObj.closeSess()

