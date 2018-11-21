import matplotlib
matplotlib.use('Agg')
from data.obspy_seismic import obspySeismicData
from models.sparseCode import sparseCode
#from plot.roc import makeRocCurve
import numpy as np
import pdb
import os

home_dir = os.getenv("HOME")

#filename = "/home/slundquist/mountData/datasets/CanadianData_2016.txt"
filename = "/home/slundquist/mountData/datasets/CanadianData_feb.txt"
example_size = 10000
#event_filename = "/home/slundquist/mountData/datasets/query_2016.csv"
station_csv = "/home/slundquist/mountData/datasets/station_info.csv"

trainDataObj = obspySeismicData(filename, example_size, seed=123456, station_csv=station_csv)

class Params(object):
    #Base output directory
    out_dir = home_dir + "/mountData/tfSparseCode/"
    #Inner run directory
    run_dir = out_dir + "/deep_lca_adam_obspy_seismic_pot/"
    save_period  = 1000
    #output plots directory
    plot_period = 100
    eval_period = 100
    #Progress step
    progress  = 10
    #Controls how often to write out to tensorboard
    write_step = 10
    #Flag for loading weights from checkpoint
    load = False
    load_file = ""
    #Device to run on
    device = '/gpu:1'
    num_steps = 10000000

    #####Sparse coding params######
    batch_size = 4
    input_shape = [example_size, trainDataObj.num_channels]

    sc_iter = 1000
    sc_verbose = True
    sc_lr = 5e-4
    D_lr = 5e-4

    num_layers = 3
    dict_patch_size = [64, 64, 64]
    err_weight = [1, 1, 1]
    act_weight = [1, 1, 1]
    top_down_weight = [0.1, 0.1, None]
    l1_weight = [.02, .008, .003]
    l2_weight = 0
    stride = [2, 1, 1]
    dict_size = [256, 256, 256]
    layer_type = ["sc_conv", "sc_conv", "sc_conv"]
    normalize_act = [False, False, False]

    target_norm_std = .1
    norm_input = True

    feature_labels = trainDataObj.feature_labels

    plot_groups = trainDataObj.station_group
    plot_group_title = trainDataObj.station_title

params = Params()

#Allocate tensorflow object
tfObj = sparseCode(params)
print("Done init")

tfObj.trainModel(trainDataObj)
print("Done run")

tfObj.closeSess()

