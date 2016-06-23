import matplotlib
matplotlib.use('Agg')
from dataObj.image import imageNetObj
from tf.ista_time import ISTA_Time
#from plot.roc import makeRocCurve
import pdb

#Input vgg file for preloaded weights
trainImageLists =  "/shared/imageNet/vid2015_128x64/imageNetVID_2015_list.txt"
#Get object from which tensorflow will pull data from
trainDataObj = imageNetObj(trainImageLists, resizeMethod="crop")

#ISTA params
params = {
    #Base output directory
    'outDir':          "/home/slundquist/mountData/tfLCA/",
    #Inner run directory
    'runDir':          "/imagenetTime/",
    'tfDir':           "/tfout",
    'ckptDir':         "/checkpoints/",
    'saveFile':        "/save-model",
    #Flag for loading weights from checkpoint
    'load':            False,
    'loadFile':        "/home/slundquist/mountData/tfLCA/saved/imagenet.ckpt",
    'numIterations':   10000000,
    'displayPeriod':   300,
    'savePeriod':      10, #In terms of displayPeriod
    #output plots directory
    'plotDir':         "plots/",
    'plotPeriod':      20, #With respect to displayPeriod
    #Progress step (also controls how often to save and write out to tensorboard)
    'progress':        10,
    'writeStep':       300,
    #####ISTA PARAMS######
    'batchSize':       1,
    #Learning rate for optimizer
    'learningRateA':   1e-3,
    'learningRateW':   1e-3,
    #Lambda in energy function
    'thresh':          .0125,
    #Number of features in V1
    'numV':            192,
    #Time dimension
    'nT':              4,
    #Stride of V1
    'VStrideT':        1,
    'VStrideY':        1,
    'VStrideX':        1,
    #Patch size
    'patchSizeT':      2,
    'patchSizeY':      8,
    'patchSizeX':      8,
}

#Allocate tensorflow object
#This will build the graph
tfObj = ISTA_Time(params, trainDataObj)

print "Done init"
tfObj.runModel()
print "Done run"

tfObj.closeSess()

