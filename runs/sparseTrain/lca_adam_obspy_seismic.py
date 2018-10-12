import matplotlib
matplotlib.use('Agg')
from dataObj.obspy_seismic import obspySeismicData
from dataObj.multithread import mtWrapper
from tf.lca_adam import LCA_ADAM
#from plot.roc import makeRocCurve
import numpy as np
import pdb

filename = "/home/slundquist/mountData/datasets/CanadianData_2016.txt"
example_size = 10000
event_filename = "/home/slundquist/mountData/datasets/query_2016.csv"
station_csv = "/home/slundquist/mountData/datasets/station_info.csv"

#Get object from which tensorflow will pull data from

batch_size = 4

trainDataObj = obspySeismicData(filename, example_size, seed=123456, event_csv=event_filename, get_type="event", station_csv=station_csv)

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
    'savePeriod':      1000, #In terms of displayPeriod
    #output plots directory
    'plotDir':         "plots/",
    'plotReconPeriod':  1000*500,
    'plotWeightPeriod': 1000*500,
    #Progress step
    'progress':        500,
    #Controls how often to write out to tensorboard
    'writeStep':       500,
    #Flag for loading weights from checkpoint
    'load':            False,
    'loadFile':        "/home/slundquist/mountData/tfSparseCode/lca_adam_obspy_seismic_events/checkpoints/save-model-7800500",
    #Device to run on
    'device':          '/gpu:1',
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
    'plot_groups':     trainDataObj.station_group,
    'plot_group_title':trainDataObj.station_title,
}

#Allocate tensorflow object
tfObj = LCA_ADAM(params, trainDataObj)
print("Done init")

tfObj.runModel()
print("Done run")

tfObj.closeSess()

