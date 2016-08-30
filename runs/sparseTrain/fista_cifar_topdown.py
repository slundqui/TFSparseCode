import matplotlib
matplotlib.use('Agg')
from dataObj.image import cifarObj
from tf.fistaTopDown import FISTATopDown
#from plot.roc import makeRocCurve
import numpy as np
import pdb

trainImageLists =  "/home/slundquist/mountData/datasets/cifar/images/train.txt"
#testImageLists = "/home/slundquist/mountData/datasets/cifar/images/test.txt"
randImageSeed = None
#Get object from which tensorflow will pull data from
trainDataObj = cifarObj(trainImageLists, resizeMethod="pad", shuffle=True, seed=randImageSeed)
#testDataObj = cifarObj(testImageLists, resizeMethod="pad")

#FISTA params
params = {
    #Base output directory
    'outDir':          "/home/slundquist/mountData/tfSparseCode/",
    #Inner run directory
    'runDir':          "/fista_cifar_topdown/",
    'tfDir':           "/tfout",
    #Save parameters
    'ckptDir':         "/checkpoints/",
    'saveFile':        "/save-model",
    'savePeriod':      100, #In terms of displayPeriod
    #output plots directory
    'plotDir':         "plots/",
    'plotPeriod':      100, #With respect to displayPeriod
    #Progress step
    'progress':        100,
    #Controls how often to write out to tensorboard
    'writeStep':       500,
    #Flag for loading weights from checkpoint
    'load':            False,
    'loadFile':        "/home/slundquist/mountData/tfSparseCode/saved/fista_cifar_nf256.ckpt",
    #Device to run on
    'device':          '/gpu:1',
    #####FISTA PARAMS######
    'numIterations':   1000000,
    'displayPeriod':   500,
    #Batch size
    'batchSize':       4,
    #Heirarchy params
    'numLayers':       3,
    #Learning rate for optimizer
    'learningRateA':   .001,
    'learningRateW':   .1,
    #Lambda in energy function
    'thresh':          [.2, .005, .001],
    #Number of features in V1
    'numV':            [64, 256, 1024],
    #Stride of V1
    'VStrideY':        [2, 2, 2],
    'VStrideX':        [2, 2, 2],
    #Patch size
    'patchSizeY':      [8, 4, 4],
    'patchSizeX':      [8, 4, 4],
}

#Allocate tensorflow object
tfObj = FISTATopDown(params, trainDataObj)
print "Done init"

tfObj.runModel()
print "Done run"

tfObj.closeSess()

