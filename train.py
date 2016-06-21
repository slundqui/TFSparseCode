import matplotlib
matplotlib.use('Agg')
from dataObj.image import cifarObj
from tf.ista import ISTA
#from plot.roc import makeRocCurve
import numpy as np
import pdb
import os
import matplotlib.pyplot as plt

#Input vgg file for preloaded weights
trainImageLists =  "/home/sheng/mountData/datasets/cifar/images/train.txt"
testImageLists = "/home/sheng/mountData/datasets/cifar/images/test.txt"

#Base output directory
outDir = "/home/sheng/mountData/tfLCA/"
#Inner run directory
runDir = outDir + "/cifar_batch16_stride2/"

#Flag for loading weights from checkpoint
load = True
loadFile = outDir + "/saved/cifar_batch16_stride2.ckpt"

outerSteps = 1000000
innerSteps = 200
saveSteps = 10 #In terms of innerSteps

#output plots directory
plotDir = runDir + "plots/"

if not os.path.exists(runDir):
   os.makedirs(runDir)

if not os.path.exists(plotDir):
   os.makedirs(plotDir)


#Get object from which tensorflow will pull data from
trainDataObj = cifarObj(trainImageLists, resizeMethod="pad")
#testDataObj = cifarObj(testImageLists, resizeMethod="pad")

#testDataObj.setMeanVar(trainDataObj.mean, trainDataObj.std)

#Allocate tensorflow object
tfObj = ISTA(trainDataObj)

#Load checkpoint if flag set
if(load):
   tfObj.loadModel(loadFile)
else:
   tfObj.initSess()

#For tensorboard
tfObj.writeSummary(runDir + "/tfout")

print "Done init"

saveFile = runDir + "/save-model"
#Normalize weights to start
tfObj.normWeights()

#Training
for i in range(outerSteps):
   if(i%saveSteps == 0):
       tfObj.trainA(innerSteps, saveFile)
   else:
       tfObj.trainA(innerSteps)
   #Train
   tfObj.trainW(plotDir)
   tfObj.normWeights()

print "Done run"

tfObj.closeSess()

