import matplotlib
matplotlib.use('Agg')
from dataObj.seismic import seismicData
from tf.lca_adam import LCA_ADAM
#from plot.roc import makeRocCurve
import numpy as np
import pdb

#trainImageLists =  "/home/sheng/mountData/datasets/cifar/images/train.txt"
#testImageLists = "/home/slundquist/mountData/datasets/cifar/images/test.txt"
randImageSeed = None

filename = "/home/sheng/mountData/seismic/seismic.txt"
exampleSize = 100
#Get object from which tensorflow will pull data from
trainDataObj = seismicData(filename, exampleSize)

#FISTA params
params = {
    #Base output directory
    'outDir':          "/home/sheng/mountData/tfSparseCode/",
    #Inner run directory
    'runDir':          "/lca_adam_seismic/",
    'tfDir':           "/tfout",
    #Save parameters
    'ckptDir':         "/checkpoints/",
    'saveFile':        "/save-model",
    'savePeriod':      10, #In terms of displayPeriod
    #output plots directory
    'plotDir':         "plots/",
    'plotPeriod':      100, #With respect to displayPeriod
    #Progress step
    'progress':        100,
    #Controls how often to write out to tensorboard
    'writeStep':       100,
    #Flag for loading weights from checkpoint
    'load':            True,
    'loadFile':        "/home/sheng/mountData/tfSparseCode/saved/seismic.ckpt",
    #Device to run on
    'device':          '/gpu:0',
    #####FISTA PARAMS######
    'numIterations':   10000,
    'displayPeriod':   1000,
    #Batch size
    'batchSize':       128,
    #Learning rate for optimizer
    'learningRateA':   5e-3,
    'learningRateW':   .1,
    #Lambda in energy function
    'thresh':          .01,
    #Number of features in V1
    'numV':            64,
    #Stride of V1
    'VStrideY':        1,
    'VStrideX':        1,
    #Patch size
    'patchSizeY':      1,
    'patchSizeX':      15,
}

#Allocate tensorflow object
tfObj = LCA_ADAM(params, trainDataObj)
print "Done init"

tfObj.runModel()
print "Done run"

tfObj.closeSess()

