import matplotlib
matplotlib.use('Agg')
from data.seismic_hdf5 import SeismicDataHdf5
from models.sparseCode import sparseCode
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
    data_save_fn=data_save_fn, loc_filter=True, sample_for_class=True, prediction=False)

class Params(object):
    #Base output directory
    out_dir = home_dir + "/mountData/tfSparseCode/"
    #Inner run directory
    run_dir = out_dir + "/sc_hdf5_seismic_single_loc_classifier_detection_less_batch/"
    save_period  = 1000
    #output plots directory
    plot_period = 100
    eval_period = 100
    #Progress step
    progress  = 10
    #Controls how often to write out to tensorboard
    write_step = 10
    #Flag for loading weights from checkpoint
    load = True
    load_file = home_dir + "/mountData/tfSparseCode/sc_hdf5_seismic_single_loc_ind_norm_fixnorm/checkpoints/save-model-7000"
    #Device to run on
    device = '/gpu:0'
    num_steps = 10000000

    legend = ["HHE", "HHN", "HHZ"]
    num_plot_weights = 10
    num_plot_recon = 1
    plot_groups = trainDataObj.station_group
    plot_group_title = trainDataObj.station_title

    #####Sparse coding params######
    batch_size = 4
    input_shape = [example_size, trainDataObj.num_features]

    sc_iter = 1000
    sc_verbose = True
    sc_lr = 5e-4
    D_lr = 5e-4

    dict_patch_size = 1024
    #l1_weight = .01 #Single
    l2_weight = 0
    stride = 2
    dict_size = 768
    layer_type = "sc_conv"

    use_gan = False
    load_gan = False
    gan_weight = 0.01

    target_norm_std = .1
    l1_weight = .05 #Full

    norm_input = True
    #Normalize features independently
    norm_ind_features = False

    ###Classifier params
    load_classifier=False
    use_classifier=True
    class_lr = 3e-3



params = Params()

#Allocate tensorflow object
tfObj = sparseCode(params)
print("Done init")

tfObj.trainClassifier(trainDataObj)
print("Done run")

tfObj.closeSess()

