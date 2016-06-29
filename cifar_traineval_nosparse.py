import matplotlib
matplotlib.use('Agg')
from dataObj.image import cifarObj
from tf.ista import ISTA
#from plot.roc import makeRocCurve
import numpy as np
import pdb

trainImageLists =  "/home/slundquist/mountData/datasets/cifar/images/train.txt"
#testImageLists = "/home/slundquist/mountData/datasets/cifar/images/test.txt"
randImageSeed = None
#Get object from which tensorflow will pull data from
trainDataObj = cifarObj(trainImageLists, resizeMethod="pad", shuffle=False, seed=randImageSeed)
#testDataObj = cifarObj(testImageLists, resizeMethod="pad")

#ISTA params
params = {
    #Base output directory
    'outDir':          "/home/slundquist/mountData/tfLCA/",
    #Inner run directory
    'runDir':          "/cifar_eval/",
    'tfDir':           "/tfout",
    #Save parameters
    'ckptDir':         "/checkpoints/",
    'saveFile':        "/save-model",
    'savePeriod':      100, #In terms of displayPeriod
    #output plots directory
    'plotDir':         "plots/",
    'plotPeriod':      20, #With respect to displayPeriod
    #Progress step
    'progress':        10,
    #Controls how often to write out to tensorboard
    'writeStep':       10, #300,
    #Threshold
    'zeroThresh':      0,
    #Flag for loading weights from checkpoint
    'load':            True,
    'loadFile':        "/home/slundquist/mountData/tfLCA/saved/cifar_batch16_stride2.ckpt",
    #Device to run on
    'device':          '/gpu:1',
    #####ISTA PARAMS######
    'numIterations':   100000,
    'displayPeriod':   200,
    #Batch size
    'batchSize':       32,
    #Learning rate for optimizer
    'learningRateA':   1e-3,
    'learningRateW':   1e-4,
    #Lambda in energy function
    'thresh':          .015,
    #Number of features in V1
    'numV':            128,
    #Stride of V1
    'VStrideY':        2,
    'VStrideX':        2,
    #Patch size
    'patchSizeY':      12,
    'patchSizeX':      12,
}

#Allocate tensorflow object
tfObj = ISTA(params, trainDataObj)
print "Done init"
tfObj.evalSet(trainDataObj, "/home/slundquist/mountData/tfLCA/cifar_eval/train_data_nosparse/cifar_")
print "Done run"

tfObj.closeSess()

