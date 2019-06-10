import matplotlib
matplotlib.use('Agg')
from dataObj.image import imageNetObj
from tf.ista import ISTA
#from plot.roc import makeRocCurve
import pdb

#Input vgg file for preloaded weights
trainImageLists =  "/shared/imageNet/vid2015_128x64/imageNetVID_2015_list.txt"
#Get object from which tensorflow will pull data from
trainDataObj = imageNetObj(trainImageLists, resizeMethod="pad")

#ISTA params
params = {
    #Base output directory
    'outDir': "/home/slundquist/mountData/tfLCA/",
    #Inner run directory
    'runDir': "/imagenet/",
    'tfDir': "/tfout",
    'ckptDir': '/checkpoints/',
    'saveFile': '/save-model',
    #Flag for loading weights from checkpoint
    'load': True,
    'loadFile': "/home/slundquist/mountData/tfLCA/saved/imagenet.ckpt",
    'numIterations': 1000000,
    'displayPeriod': 300,
    'savePeriod': 10, #In terms of displayPeriod
    #output plots directory
    'plotDir': "plots/",
    'plotPeriod': 20, #With respect to displayPeriod
    #Progress step (also controls how often to save and write out to tensorboard)
    'progress': 300,
    #####ISTA PARAMS######
    'batchSize': 1,
    #Learning rate for optimizer
    'learningRateA': 1e-4,
    'learningRateW': 1e-4,
    #Lambda in energy function
    'thresh': .0125,
    #Number of features in V1
    'numV': 3072,
    #Stride of V1
    'VStride': 4,
    #Patch size
    'patchSizeY': 16,
    'patchSizeX': 32,
}

#Allocate tensorflow object
#This will build the graph
tfObj = ISTA(params, trainDataObj)

print("Done init")
tfObj.runModel()
print("Done run")

tfObj.closeSess()

