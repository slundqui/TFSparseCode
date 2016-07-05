#import matplotlib
#matplotlib.use('Agg')
from dataObj.image import pvpObj
from tf.slp_sparse_code import SLP
import numpy as np
import pdb

#pvp files
trainFileList = "/home/slundquist/mountData/datasets/cifar/pvp/CONCAT_TRAIN.pvp"
trainGtList =  "/home/slundquist/mountData/datasets/cifar/pvp/mixed_cifar_compneuro.txt"

testFileList = "/home/slundquist/mountData/datasets/cifar/pvp/CONCAT_TEST.pvp"
testGtList =  "/home/slundquist/mountData/datasets/cifar/pvp/test_batch_randorder_compneuro.txt"

#Get object from which tensorflow will pull data from
trainDataObj = pvpObj(trainFileList, trainGtList, (16, 16, 256), resizeMethod="crop", shuffle=True, skip=1, seed=None)
testDataObj = pvpObj(testFileList, testGtList, (16, 16, 256), resizeMethod="crop", shuffle=True, skip=1, seed=None)

params = {
    #Base output directory
    'outDir':          "/home/slundquist/mountData/tfLCA/",
    #Inner run directory
    'runDir':          "/cifar_lca_slp/",
    'tfDir':           "/tfout",
    #Save parameters
    'ckptDir':         "/checkpoints/",
    'saveFile':        "/save-model",
    'savePeriod':      100, #In terms of displayPeriod
    #output plots directory
    'plotDir':         "plots/",
    'plotPeriod':      100, #With respect to displayPeriod
    #Progress step
    'progress':        10,
    #Controls how often to write out to tensorboard
    'writeStep':       100, #300,
    #Flag for loading weights from checkpoint
    'load':            False,
    'loadFile':        "/home/slundquist/mountData/DeepGAP/saved/cifar.ckpt",
    #Input vgg file for preloaded weights
    'pvpWeightFile':   "/home/slundquist/mountData/datasets/cifar/pvp/S1ToImageReconS1Error_W.pvp",
    #Device to run on
    'device':          '/gpu:1',
    #Num iterations
    'outerSteps':      10000000, #1000000,
    'innerSteps':      50, #300,
    #Batch size
    'batchSize':       128,
    #Learning rate for optimizer
    'learningRate':    1e-4,
    'numClasses':      10,

    #####ISTA PARAMS######
    'VStrideY':        2,
    'VStrideX':        2,
    'rectify': False,
}

#Allocate tensorflow object
#This will build the graph
tfObj = SLP(params, trainDataObj.inputShape)

print "Done init"
tfObj.runModel(trainDataObj, testDataObj = testDataObj)
print "Done run"

tfObj.closeSess()

