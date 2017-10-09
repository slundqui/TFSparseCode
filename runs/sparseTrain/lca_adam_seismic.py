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

filename = "/home/slundquist/mountData/datasets/seismic/wf.txt"
settingsFilename = "/home/slundquist/mountData/datasets/seismic/p4681_run1_AE.mat"
exampleSize = 10000
#Get object from which tensorflow will pull data from
trainDataObj = seismicData(filename, settingsFilename, exampleSize, shuffle=True)

#FISTA params
params = {
    #Base output directory
    'outDir':          "/home/slundquist/mountData/tfSparseCode/",
    #Inner run directory
    'runDir':          "/lca_adam_seismic_ps1024_nf256_2/",
    'tfDir':           "/tfout",
    #Save parameters
    'ckptDir':         "/checkpoints/",
    'saveFile':        "/save-model",
    'savePeriod':      400, #In terms of displayPeriod
    #output plots directory
    'plotDir':         "plots/",
    'plotPeriod':      400, #With respect to displayPeriod
    #Progress step
    'progress':        100,
    #Controls how often to write out to tensorboard
    'writeStep':       500,
    #Flag for loading weights from checkpoint
    'load':            True,
    'loadFile':        "/home/slundquist/mountData/tfSparseCode/lca_adam_seismic_ps1024_nf256/checkpoints/save-model-3400500",
    #Device to run on
    'device':          '/gpu:0',
    #####FISTA PARAMS######
    'numIterations':   10001,
    'displayPeriod':   500,
    #Batch size
    'batchSize':       2,
    #Learning rate for optimizer
    'learningRateA':   3e-5,
    'learningRateW':   .1,
    #Lambda in energy function
    'thresh':          .0015,
    #Number of features in V1
    'numV':            256,
    #Stride of V1
    'VStrideY':        1,
    'VStrideX':        2,
    #Patch size
    'patchSizeY':      1,
    'patchSizeX':      1024,
}

#Allocate tensorflow object
tfObj = LCA_ADAM(params, trainDataObj)
print("Done init")

tfObj.runModel()
print("Done run")

tfObj.closeSess()

