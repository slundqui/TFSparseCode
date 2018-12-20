import matplotlib
matplotlib.use('Agg')
from data.seismic_hdf5 import SeismicDataHdf5
from models.sparseCode import sparseCode
import numpy as np
import pdb
import os

home_dir = os.getenv("HOME")

filename = "/mnt/drive1/DataFile.hdf5"

example_size = 10000

trainDataObj = SeismicDataHdf5(filename, example_size, seed=123456)

class Params(object):
    #Base output directory
    out_dir = home_dir + "/mountData/tfSparseCode/"
    #Inner run directory
    run_dir = out_dir + "/sc_hdf5_seismic_pre_norm_less/"
    save_period  = 1000
    #output plots directory
    plot_period = 500
    eval_period = 500
    #Progress step
    progress  = 10
    #Controls how often to write out to tensorboard
    write_step = 10
    #Flag for loading weights from checkpoint
    load = False
    load_file = home_dir + "/mountData/tfSparseCode/sc_hdf5_seismic/checkpoints/save-model-5000"
    #Device to run on
    device = '/gpu:1'
    num_steps = 10000000

    legend = ["HHE", "HHN", "HHZ"]
    num_plot_weights = 10
    num_plot_recon = 3
    plot_groups = trainDataObj.station_group
    plot_group_title = trainDataObj.station_title

    #####Sparse coding params######
    batch_size = 4
    input_shape = [example_size, trainDataObj.num_features]

    sc_iter = 500
    sc_verbose = True
    sc_lr = 1e-3
    D_lr = 1e-3

    num_layers = 1
    dict_patch_size = [1024]
    err_weight = [1]
    act_weight = [1]
    top_down_weight = [None]
    l1_weight = [.003]
    l2_weight = 0
    stride = [8]
    dict_size = [256]
    layer_type = ["sc_conv", "sc_conv", "sc_conv"]
    normalize_act = [False]

    target_norm_std = .1
    norm_input = False
    #Normalize features independently
    norm_ind_features = False


params = Params()

#Allocate tensorflow object
tfObj = sparseCode(params)
print("Done init")

tfObj.trainModel(trainDataObj)
print("Done run")

tfObj.closeSess()

