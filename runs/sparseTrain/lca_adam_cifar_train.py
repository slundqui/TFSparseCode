import matplotlib
matplotlib.use('Agg')
from dataObj.image import cifarObj
from tf.lca_adam import LCA_ADAM
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
    'runDir':          "/lca_adam_cifar_nf256/",
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
    'writeStep':       200,
    #Flag for loading weights from checkpoint
    'load':            False,
    'loadFile':        "/home/slundquist/mountData/tfSparseCode/saved/fista_cifar_nf256.ckpt",
    #Device to run on
    'device':          '/gpu:0',
    #####FISTA PARAMS######
    'numIterations':   100000,
    'displayPeriod':   200,
    #Batch size
    'batchSize':       8,
    #Learning rate for optimizer
    'learningRateA':   5e-3,
    'learningRateW':   1,
    #Lambda in energy function
    'thresh':          .01,
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
tfObj = LCA_ADAM(params, trainDataObj)
print "Done init"

tfObj.runModel()
print "Done run"

tfObj.closeSess()

