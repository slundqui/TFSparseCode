import matplotlib
matplotlib.use('Agg')
from dataObj.image import cifarObj
from tf.adam import AdamSP
#from plot.roc import makeRocCurve
import numpy as np
import pdb

#trainImageLists =  "/home/slundquist/mountData/datasets/cifar/images/train.txt"
testImageLists = "/home/slundquist/mountData/datasets/cifar/images/test.txt"
randImageSeed = None
#Get object from which tensorflow will pull data from
#trainDataObj = cifarObj(trainImageLists, resizeMethod="pad", shuffle=False, seed=randImageSeed)
testDataObj = cifarObj(testImageLists, resizeMethod="pad", shuffle=False, seed=randImageSeed)

#ISTA params
params = {
    #Base output directory
    'outDir':          "/home/slundquist/mountData/tfSparseCode/",
    #Inner run directory
    'runDir':          "/adam_cifar_nf256_eval/",
    'tfDir':           "/tfout",
    #Save parameters
    'ckptDir':         "/checkpoints/",
    'saveFile':        "/save-model",
    'savePeriod':      100, #In terms of displayPeriod
    #output plots directory
    'plotDir':         "plots/",
    'plotPeriod':      20, #With respect to displayPeriod
    #Progress step
    'progress':        200,
    #Controls how often to write out to tensorboard
    'writeStep':       200, #300,
    #Threshold
    'zeroThresh':      1e-3,
    #Flag for loading weights from checkpoint
    'load':            True,
    'loadFile':        "/home/slundquist/mountData/tfSparseCode/saved/adam_cifar_nf256.ckpt",
    #Device to run on
    'device':          '/gpu:1',
    #####ISTA PARAMS######
    'numIterations':   6250,
    'displayPeriod':   200,
    #Batch size
    'batchSize':       8,
    #Learning rate for optimizer
    'learningRateA':   1e-3,
    'learningRateW':   1,
    #Lambda in energy function
    'thresh':          .02,
    #Number of features in V1
    'numV':            256,
    #Stride of V1
    'VStrideY':        2,
    'VStrideX':        2,
    #Patch size
    'patchSizeY':      12,
    'patchSizeX':      12,
}

#Allocate tensorflow object
tfObj = AdamSP(params, testDataObj)
print "Done init"
outFilename = params["outDir"] + params["runDir"] + "test_adam_cifar_256_eval.pvp"
tfObj.evalSet(testDataObj, outFilename)
print "Done run"

tfObj.closeSess()

