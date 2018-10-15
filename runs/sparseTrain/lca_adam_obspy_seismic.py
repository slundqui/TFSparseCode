import matplotlib
matplotlib.use('Agg')
from data.obspy_seismic import obspySeismicData
from models.sparseCode import sparseCode
#from plot.roc import makeRocCurve
import numpy as np
import pdb
import os

home_dir = os.getenv("HOME")
filename = "/home/slundquist/mountData/datasets/CanadianData_feb.txt"
example_size = 10000

#Get object from which tensorflow will pull data from
trainDataObj = obspySeismicData(filename, example_size, seed=123456)

class Params(object):
    #Base output directory
    out_dir = home_dir + "/mountData/tfSparseCode/"
    #Inner run directory
    run_dir = out_dir + "/deep_lca_adam_obspy_seismic/"
    save_period  = 10000
    #output plots directory
    plot_period = 1000
    eval_period = 1000
    #Progress step
    progress  = 10
    #Controls how often to write out to tensorboard
    write_step = 10
    #Flag for loading weights from checkpoint
    load = False
    load_file = ""
    #Device to run on
    device = '/gpu:0'
    num_steps = 10000000

    #####Sparse coding params######
    batch_size = 8
    input_shape = [example_size, trainDataObj.num_channels]

    sc_iter = 500
    sc_verbose = True
    sc_lr = 2e-3
    D_lr = 5e-4

    num_layers = 1
    dict_patch_size = [1024]
    err_weight = [1]
    act_weight = [1]
    l1_weight = [.01]
    l2_weight = 0
    stride = [2]
    dict_size = [256]
    layer_type = ["sc_conv"]
    normalize_act = [False]

    target_norm_std = .4
    norm_input = True

    feature_labels = trainDataObj.feature_labels

params = Params()

#Allocate tensorflow object
tfObj = sparseCode(params)
print("Done init")

tfObj.trainModel(trainDataObj)
print("Done run")

tfObj.closeSess()

