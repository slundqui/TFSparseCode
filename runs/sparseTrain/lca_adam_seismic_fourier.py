import matplotlib
matplotlib.use('Agg')
from dataObj.seismic import seismicDataFourier
from tf.lca_adam import LCA_ADAM
#from plot.roc import makeRocCurve
import numpy as np
import pdb

#trainImageLists =  "/home/sheng/mountData/datasets/cifar/images/train.txt"
#testImageLists = "/home/slundquist/mountData/datasets/cifar/images/test.txt"
randImageSeed = None

filename = "/home/slundquist/mountData/datasets/seismic/wf.txt"
settingsFilename = "/home/slundquist/mountData/datasets/seismic/p4681_run1_AE.mat"
exampleSize = 1024
#Get object from which tensorflow will pull data from
trainDataObj = seismicDataFourier(filename, settingsFilename, exampleSize, shuffle=True, scaleByChannel=False)

#FISTA params
params = {
    #Base output directory
    'outDir':          "/home/slundquist/mountData/tfSparseCode/",
    #Inner run directory
    'runDir':          "/lca_adam_seismic_fourier/",
    'tfDir':           "/tfout",
    #Save parameters
    'ckptDir':         "/checkpoints/",
    'saveFile':        "/save-model",
    'savePeriod':      200, #In terms of displayPeriod
    #output plots directory
    'plotDir':         "plots/",
    'plotReconPeriod':  1000*500,   #400*500,
    'plotWeightPeriod': 1000*500,#400*500,
    #Progress step
    'progress':        100,
    #Controls how often to write out to tensorboard
    'writeStep':       1000, #600,
    #Flag for loading weights from checkpoint
    'load':            False,
    'loadFile':        "/home/slundquist/mountData/tfSparseCode/lca_adam_seismic_ps1024_nf256_dyn_scale/checkpoints/save-model-2200500",
    #Device to run on
    'device':          '/gpu:0',
    #####FISTA PARAMS######
    'numIterations':   100001,
    'displayPeriod':   1000,
    #Batch size
    'batchSize':       4,
    #Learning rate for optimizer
    'learningRateA':   5e-5,
    'learningRateW':   .1,
    #Lambda in energy function
    'thresh':          .01,
    #Number of features in V1
    'numV':            4096, #TODO make overcomplete

    #Sets if fully connected or conv
    'conv'    :        False,

    #Stride of V1
    'VStrideY':        1,
    'VStrideX':        2,
    #Patch size
    'patchSizeY':      1,
    'patchSizeX':      1024,
    'inputMult':       1,

    'fourier':         True,
}

#Allocate tensorflow object
tfObj = LCA_ADAM(params, trainDataObj)
print("Done init")

tfObj.runModel()
print("Done run")

tfObj.closeSess()

