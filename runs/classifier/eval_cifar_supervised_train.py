import matplotlib
matplotlib.use('Agg')
from dataObj.image import cifarObj
from tf.supervised_control import Supervised
import numpy as np
import pdb

#Paths to list of filenames
trainList =  "/home/slundquist/mountData/datasets/cifar/images/train.txt"
testList =  "/home/slundquist/mountData/datasets/cifar/images/test.txt"

#Get object from which tensorflow will pull data from
trainDataObj = cifarObj(trainList, resizeMethod="crop", shuffle=True, skip=1, seed=None, getGT=True)
testDataObj = cifarObj(testList, resizeMethod="crop", shuffle=False, skip=1, seed=None, getGT=True)

params = {
    #Base output directory
    'outDir':          "/home/slundquist/mountData/tfLCA/",
    #Inner run directory
    'runDir':          "/cifar_sup_128/",
    'tfDir':           "/tfout",
    #Save parameters
    'ckptDir':         "/checkpoints/",
    'saveFile':        "/save-model",
    'savePeriod':      100, #In terms of displayPeriod
    #output plots directory
    'plotDir':         "plots/",
    'plotPeriod':      200, #With respect to displayPeriod
    #Progress step
    'progress':        10,
    #Controls how often to write out to tensorboard
    'writeStep':       100, #300,
    #Flag for loading weights from checkpoint
    'load':            False,
    'loadFile':        "/home/slundquist/mountData/DeepGAP/saved/cifar.ckpt",
    #Device to run on
    'device':          '/gpu:1',
    #Num iterations
    'outerSteps':      500, #1000000,
    'innerSteps':      100, #300,
    #Batch size
    'batchSize':       100,
    #Learning rate for optimizer
    'learningRate':    1e-4,
    'numClasses':      10,

    'epsilon': 1e-8,

    'regularizer': 'none',
    'regWeight': .3,

    #####ISTA PARAMS######
    'VStrideY':        2,
    'VStrideX':        2,
    'patchSizeX':      12,
    'patchSizeY':      12,
    'numV':            128,
    #####New encode parapms#####
    'maxPool':         True, #Controls max or avg pool
}

print("Done init")
est = np.zeros((testDataObj.numImages))
gt = np.zeros((testDataObj.numImages))

#Allocate tensorflow object
#This will build the graph
tfObj = Supervised(params, trainDataObj.inputShape)

assert(testDataObj.numImages % params["batchSize"] == 0)

for i in range(testDataObj.numImages/params["batchSize"]):
    print(i*params["batchSize"], "out of", testDataObj.numImages)
    (inImage, inGt) = testDataObj.getData(params["batchSize"])
    outVals = tfObj.evalModel(inImage, inGt = inGt, plot=False)
    tfObj.timestep += 1
    v = np.argmax(outVals, axis=1)

    startIdx = i*batch
    endIdx = startIdx + params["batchSize"]
    est[startIdx:endIdx] = v
    gt[startIdx:endIdx] = inGt

print("Done run")

tfObj.closeSess()

numCorrect = len(np.nonzero(est == gt)[0])
print("Accuracy: ", float(numCorrect)/testDataObj.numImages)

