import matplotlib
matplotlib.use('Agg')
from dataObj.obspy_seismic import obspySeismicData
from dataObj.multithread import mtWrapper
from tf.lca_adam import LCA_ADAM
#from plot.roc import makeRocCurve
import numpy as np
import pdb

filename = "/home/slundquist/mountData/datasets/CanadianData_feb.txt"
example_size = 10000
#Get object from which tensorflow will pull data from

batch_size = 4

#Event
start_time = "2016-02-24T21:25:00"
end_time = "2016-02-24T21:55:00"
##No event
#start_time = "2016-02-20T00:00:00"
#end_time = "2016-02-20T12:00:00"
trainDataObj = obspySeismicData(filename, example_size, seed=123456, time_range = [start_time, end_time])
#trainDataObj = obspySeismicData(filename, example_size, seed=123456)

#FISTA params
params = {
    #Base output directory
    'outDir':          "/home/slundquist/mountData/tfSparseCode/",
    #Inner run directory
    'runDir':          "/lca_adam_obspy_seismic_train_event/",
    'tfDir':           "/tfout",
    #Save parameters
    'ckptDir':         "/checkpoints/",
    'saveFile':        "/save-model",
    'savePeriod':      400, #In terms of displayPeriod
    #output plots directory
    'plotDir':         "plots/",
    'plotReconPeriod':  1000*500,
    'plotWeightPeriod': 1000*500,
    #Progress step
    'progress':        100,
    #Controls how often to write out to tensorboard
    'writeStep':       500,
    #Flag for loading weights from checkpoint
    'load':            False,
    'loadFile':        "/home/slundquist/mountData/tfSparseCode/lca_adam_obspy_seismic_events/checkpoints/save-model-7800500",
    #Device to run on
    'device':          '/gpu:0',
    #####Sparse coding params######
    #Fully connected sc or conv sc
    'fc':              False,
    #Iteration params
    'numIterations':   10000001,
    'displayPeriod':   500,
    #Batch size
    'batchSize':       batch_size,
    #Learning rate for optimizer
    'learningRateA':   5e-4,
    'learningRateW':   1e-3,
    #Lambda in energy function
    'thresh':          .0025,
    #Number of features in V1
    'numV':            256,
    #Stride of V1
    'VStrideY':        1,
    'VStrideX':        2,
    #Patch size
    'patchSizeY':      1,
    'patchSizeX':      1024,
    'normalize':       True,
    'inputMult':       .4,
}

#Allocate tensorflow object
tfObj = LCA_ADAM(params, trainDataObj)
print("Done init")

tfObj.runModel()
print("Done run")

tfObj.closeSess()

