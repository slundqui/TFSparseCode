import matplotlib
matplotlib.use('Agg')
from dataObj.image import cifarObj
from tf.adam import AdamSP
import numpy as np
import pdb

trainImageLists =  "/home/slundquist/mountData/datasets/cifar/images/train.txt"
#testImageLists = "/home/slundquist/mountData/datasets/cifar/images/test.txt"
randImageSeed = None
#Get object from which tensorflow will pull data from
trainDataObj = cifarObj(trainImageLists, resizeMethod="crop", shuffle=True, seed=randImageSeed)
#testDataObj = cifarObj(testImageLists, resizeMethod="pad")

#ISTA params
params = {
    #Base output directory
    'outDir':          "/home/slundquist/mountData/tfSparseCode/",
    #Inner run directory
    'runDir':          "/adadelta_cifar_nf256/",
    'tfDir':           "/tfout",
    #Save parameters
    'ckptDir':         "/checkpoints/",
    'saveFile':        "/save-model",
    'savePeriod':      200, #In terms of displayPeriod
    #output plots directory
    'plotDir':         "plots/",
    'plotPeriod':      100, #With respect to displayPeriod
    #Progress step
    'progress':        100,
    #Controls how often to write out to tensorboard
    'writeStep':       200,
    #Threshold
    'zeroThresh':      1e-3,
    #Flag for loading weights from checkpoint
    'load':            False,
    'loadFile':        "/home/slundquist/mountData/tfLCA/saved/cifar_nf128.ckpt",
    #Device to run on
    'device':          '/gpu:1',
    #####ISTA PARAMS######
    'numIterations':   1000000,
    'displayPeriod':   200,
    #Batch size
    'batchSize':       32,
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
tfObj = AdamSP(params, trainDataObj)
print "Done init"

tfObj.runModel()
print "Done run"

tfObj.closeSess()

