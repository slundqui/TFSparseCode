import matplotlib
matplotlib.use('Agg')
from data.seismic_hdf5 import SeismicDataHdf5
from models.sparseCode import sparseCode
import numpy as np
import pdb
import os

from plots import plotRecon, plotWeights

home_dir = os.getenv("HOME")

filename = "/mnt/drive1/DataFile.hdf5"

example_size = 2000

#seed = 123456
#Train continue seed
#seed = 89673937
seed = None
downsample_stride = 5

#channel_idx = None
#station_idx = [0]

channel_idx = [0]
station_idx = None

trainDataObj = SeismicDataHdf5(filename, example_size, seed=123456, downsample_stride=downsample_stride, channel_idx=channel_idx, station_idx=station_idx)

class Params(object):
    #Base output directory
    out_dir = home_dir + "/mountData/tfSparseCode/"
    #Inner run directory
    run_dir = out_dir + "/sc_hdf5_seismic_single_channel_wider_eval/"
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
    load_file = home_dir + "/mountData/tfSparseCode/sc_hdf5_seismic_single_channel_wider_no_gan_4/checkpoints/save-model-39000"
    #Device to run on
    device = '/gpu:1'
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

    dict_patch_size = 2000
    #l1_weight = .0005 #Full
    l1_weight = .01 #Single
    l2_weight = 0
    stride = 2
    dict_size = 768
    layer_type = "sc_conv"

    use_gan = False
    load_gan = False
    gan_weight = 0.01

    #target_norm_std = .01
    target_norm_std = 1
    norm_input = False
    #Normalize features independently
    norm_ind_features = False


params = Params()

#Allocate tensorflow object
tfObj = sparseCode(params)
print("Done init")

np_dict = tfObj.sess.run(tfObj.scObj.model["dictionary"])

from sklearn.decomposition import PCA

#np_dict = np.transpose(np_dict, [2, 1, 0])
##np_dict now in [features, station, time_samples]
#[n_features, n_stations, n_samples] = np_dict.shape
#np_dict = np.reshape(np_dict, [n_features, -1])
#
#pca = PCA().fit(np_dict)
#pca_components = np.reshape(pca.components_, [n_features, n_stations, n_samples])

out_fn = params.run_dir + "eval_out_sc.npy"
if(os.path.exists(out_fn)):
  out_act = np.load(out_fn)
else:
  out_act = tfObj.evalSet(trainDataObj.data[:, :example_size, :])
  np.save(out_fn, out_act)

#out_act is [num_samples, time, features]
[num_samples, num_time, num_features] = out_act.shape

out_act_flat = np.reshape(out_act, [num_samples, -1])
pca = PCA().fit(out_act_flat)
components = np.reshape(pca.components_, [-1, num_time, num_features])

#Plot and reconstruct

recon = tfObj.calcRecon(components[0])

prefix=params.run_dir + "pca_0"

plotRecon.plotRecon1D(recon, recon, prefix,
        num_plot = params.num_plot_recon,
        groups=params.plot_groups, group_title=params.plot_group_title,
        legend=params.legend)

tfObj.closeSess()

