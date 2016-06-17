import matplotlib
matplotlib.use('Agg')
from dataObj.dogwalker import dataObj
from tf.imageClassification import imageClassification
from plot.roc import makeRocCurve
import numpy as np
import pdb
import os

#Paths to list of filenames
trainImageList = "/home/sheng/mountData/datasets/CropsForObjectClassifier/Dog/train.txt"
testImageList = "/home/sheng/mountData/datasets/CropsForObjectClassifier/Dog/test.txt"

#Base output directory
outDir = "/home/sheng/mountData/dogwalk/"
#Inner run directory
runDir = outDir + "/dogPad_eval0/"
#output plots directory
plotDir = runDir + "plots/"

if not os.path.exists(runDir):
   os.makedirs(runDir)

if not os.path.exists(plotDir):
   os.makedirs(plotDir)

#Flag for loading weights from checkpoint
load = True
loadFile = outDir + "/saved/dogPad.ckpt"

#Get object from which tensorflow will pull data from
trainDataObj = dataObj(trainImageList, resizeMethod="pad")
testDataObj = dataObj(testImageList, resizeMethod="pad")

testDataObj.setMeanVar(trainDataObj.mean, trainDataObj.std)
##If resizeMethod == "max", run this code to match maxDims
#if(trainDataObj.maxDim >= testDataObj.maxDim):
#    testDataObj.setMaxDim(trainDataObj.maxDim)
#else:
#    trainDataObj.setMaxDim(testDataObj.maxDim)

#Allocate tensorflow object
tfObj = imageClassification(trainDataObj.inputShape)

#Load checkpoint if flag set
if(load):
   tfObj.loadModel(loadFile)
else:
   tfObj.initSess()

#For tensorboard
tfObj.writeSummary(runDir + "/tfout")

print "Done init"

#Evaluate test frame
(evalData, gtData) = testDataObj.allImages()
outVals = tfObj.evalModel(evalData)
#outVals is a [batch, 2] numpy array
#We subtract positive (1) from negative(0) to change to scalar
estVals = outVals[:, 1] - outVals[:, 0]
makeRocCurve(estVals, gtData[:, 1], plotsOutDir=plotDir)

tfObj.closeSess()
