import matplotlib
matplotlib.use('Agg')
from data.seismic_hdf5 import SeismicDataHdf5
from models.mlp import mlp
import numpy as np
import pdb
import os

home_dir = os.getenv("HOME")

filename = "/mnt/drive1/large_seismic_hdf5/data.txt"

example_size = 2000

#seed = 123456
#Train continue seed
#seed = 89673937
seed = None
downsample_stride = 5

#channel_idx = None
#station_idx = [0]

#channel_idx = [0]
channel_idx = None
station_idx = None

data_save_fn = "/home/slundquist/Work/Datasets/seismic/class_data_loc.npy"
#None means load data from files
#data_save_fn = None

trainDataObj = SeismicDataHdf5(filename, example_size, seed=123456, normalize=False,
  downsample_stride=downsample_stride, channel_idx=channel_idx, station_idx=station_idx,
  data_save_fn=data_save_fn, loc_filter=True, sample_for_class=True, prediction=True)

class Params(object):
    #Base output directory
    out_dir = home_dir + "/mountData/tfSparseCode/"
    #Inner run directory
    run_dir = out_dir + "/mlp_hdf5_seismic_prediction/"
    save_period  = 1000
    #output plots directory
    plot_period = 1000
    eval_period = 1000
    #Progress step
    progress  = 10
    #Controls how often to write out to tensorboard
    write_step = 10
    #Flag for loading weights from checkpoint
    load = False
    load_file = home_dir + "/mountData/tfSparseCode/sc_hdf5_seismic_single_loc_large_mag/checkpoints/save-model-4000"
    #Device to run on
    device = '/gpu:0'
    #num_steps = 10000000
    num_steps = 10000

    legend = ["HHE", "HHN", "HHZ"]
    num_plot_weights = 10
    num_plot_recon = 1
    plot_groups = trainDataObj.station_group
    plot_group_title = trainDataObj.station_title

    #####MLP params######
    batch_size = 32
    input_shape = [example_size, trainDataObj.num_features]

    dict_patch_size = 1024
    stride = 2
    dict_size = 768

    class_lr = 5e-4

    target_norm_std = .1

    norm_input = True
    #Normalize features independently
    norm_ind_features = False


params = Params()

#Allocate tensorflow object
tfObj = mlp(params)
print("Done init")

tfObj.trainModel(trainDataObj)
print("Done run")

tfObj.closeSess()

