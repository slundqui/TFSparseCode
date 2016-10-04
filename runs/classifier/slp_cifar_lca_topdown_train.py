import matplotlib
matplotlib.use('Agg')
from dataObj.image import pvpObj
from tf.slp_sparse_code import SLP
import numpy as np
import pdb

#pvp files
#trainFileList = "/home/slundquist/mountData/cifar_pv/cifar_trainset/S4.pvp"
#trainFileList = "/home/slundquist/mountData/cifar_pv/cifar_trainset/S3.pvp"
#trainFileList = "/home/slundquist/mountData/cifar_pv/cifar_trainset/S2.pvp"
trainFileList = "/home/slundquist/mountData/cifar_pv/cifar_trainset/S1.pvp"
trainGtList =  "/home/slundquist/mountData/cifar_pv/cifar_trainset/cifar_train.txt"

#testFileList = "/home/slundquist/mountData/cifar_pv/cifar_testset/S4.pvp"
#testFileList = "/home/slundquist/mountData/cifar_pv/cifar_testset/S3.pvp"
#testFileList = "/home/slundquist/mountData/cifar_pv/cifar_testset/S2.pvp"
testFileList = "/home/slundquist/mountData/cifar_pv/cifar_testset/S1.pvp"
testGtList =  "/home/slundquist/mountData/cifar_pv/cifar_testset/cifar_test.txt"

#Get object from which tensorflow will pull data from
#trainDataObj = pvpObj(trainFileList, trainGtList, (1, 1, 1536), resizeMethod="crop", shuffle=True, skip=1, seed=None)
#testDataObj = pvpObj(testFileList, testGtList, (1, 1, 1536), resizeMethod="crop", shuffle=True, skip=1, seed=None)
#trainDataObj = pvpObj(trainFileList, trainGtList, (4, 4, 384), resizeMethod="crop", shuffle=True, skip=1, seed=None)
#testDataObj = pvpObj(testFileList, testGtList, (4, 4, 384), resizeMethod="crop", shuffle=True, skip=1, seed=None)
#trainDataObj = pvpObj(trainFileList, trainGtList, (8, 8, 96), resizeMethod="crop", shuffle=True, skip=1, seed=None)
#testDataObj = pvpObj(testFileList, testGtList, (8, 8, 96), resizeMethod="crop", shuffle=True, skip=1, seed=None)
trainDataObj = pvpObj(trainFileList, trainGtList, (16, 16, 24), resizeMethod="crop", shuffle=True, skip=1, seed=None)
testDataObj = pvpObj(testFileList, testGtList, (16, 16, 24), resizeMethod="crop", shuffle=True, skip=1, seed=None)

params = {
    #Base output directory
    'outDir':          "/home/slundquist/mountData/tfLCA/",
    #Inner run directory
    'runDir':          "/cifar_lca_topdown_s1only/",
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
    'pvpWeightFile':   "/home/slundquist/mountData/datasets/cifar/pvp/128/CIFAR_128_W.pvp",
    #Device to run on
    'device':          '/gpu:1',
    #Num iterations
    'outerSteps':      10000000, #1000000,
    'innerSteps':      50, #300,
    #Batch size
    'batchSize':       128,
    #Learning rate for optimizer
    'learningRate':    1e-3,
    'numClasses':      10,
    'verifyTrain':     False,
    'verifyTest':      False,

    #####ISTA PARAMS######
    'VStrideY':        2,
    'VStrideX':        2,
    'rectify': False,

    #SLP params
    'pooledY':         8,
    'pooledX':         8,
}

#Allocate tensorflow object
#This will build the graph
tfObj = SLP(params, trainDataObj.inputShape)

print "Done init"
tfObj.runModel(trainDataObj, testDataObj = testDataObj)
print "Done run"

tfObj.closeSess()

