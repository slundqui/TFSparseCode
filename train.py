#import matplotlib
#matplotlib.use('Agg')
from dataObj.image import cifarObj
from tf.lca import LCA
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
runDir = outDir + "/cifar_run0/"

#Flag for loading weights from checkpoint
load = False
loadFile = outDir + "/saved/dog-saved.ckpt"

outerSteps = 200
innerSteps = 1000

#output plots directory
plotDir = runDir + "plots/"

if not os.path.exists(runDir):
   os.makedirs(runDir)

if not os.path.exists(plotDir):
   os.makedirs(plotDir)


#Get object from which tensorflow will pull data from
trainDataObj = cifarObj(trainImageLists, resizeMethod="pad")
testDataObj = cifarObj(testImageLists, resizeMethod="pad")

testDataObj.setMeanVar(trainDataObj.mean, trainDataObj.std)



#Allocate tensorflow object
tfObj = LCA(trainDataObj)

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
   #Train
   tfObj.trainW()
   tfObj.normWeights()
   tfObj.trainA(innerSteps, saveFile)

print "Done run"

tfObj.closeSess()

