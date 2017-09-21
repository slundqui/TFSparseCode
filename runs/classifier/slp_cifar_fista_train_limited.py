import matplotlib
matplotlib.use('Agg')
from dataObj.image import pvpObj
from tf.slp_sparse_code import SLP
import numpy as np
import pdb

#Paths to list of filenames
trainFileList = "/home/slundquist/mountData/tfSparseCode/fista_cifar_nf256_eval/fista_train_cifar_256_eval.pvp"
trainGtList =  "/home/slundquist/mountData/datasets/cifar/images/train.txt"

testFileList = "/home/slundquist/mountData/tfSparseCode/fista_cifar_nf256_eval/fista_test_cifar_256_eval.pvp"
testGtList =  "/home/slundquist/mountData/datasets/cifar/images/test.txt"

#Get object from which tensorflow will pull data from
trainDataObj = pvpObj(trainFileList, trainGtList, (16, 16, 256), resizeMethod="crop", shuffle=True, skip=1, seed=None, rangeIdx=range(128))
testDataObj = pvpObj(testFileList, testGtList, (16, 16, 256), resizeMethod="crop", shuffle=True, skip=1, seed=None)

params = {
    #Base output directory
    'outDir':          "/home/slundquist/mountData/tfSparseCode/",
    #Inner run directory
    'runDir':          "/cifar_fista_slp_limited/",
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
    'pvpWeightFile':   "/home/slundquist/mountData/tfLCA/cifar_train/pvp/ista.pvp",
    #Device to run on
    'device':          '/gpu:0',
    #Num iterations
    'outerSteps':      10000000, #1000000,
    'innerSteps':      50, #300,
    #Batch size
    'batchSize':       4,
    #Learning rate for optimizer
    'learningRate':    1e-4,
    'numClasses':      10,
    'verifyTrain':     False,
    'verifyTest':      False,

    #####ISTA PARAMS######
    'VStrideY':        2,
    'VStrideX':        2,
    'rectify': False,

}

#Allocate tensorflow object
#This will build the graph
tfObj = SLP(params, trainDataObj.inputShape)

print("Done init")
tfObj.runModel(trainDataObj, testDataObj = testDataObj, numTest=256)
print("Done run")

tfObj.closeSess()

