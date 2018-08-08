#import matplotlib
#matplotlib.use('Agg')
from models.deepSparseCode import deepSparseCode
import pdb
import os
import const

home_dir = os.getenv("HOME")
base_ds_dir = home_dir + "/mountData/dataset/netApData/indicators_summary/"

#TODO add data obj here

#Parameter class
class Params(object):
    ##Bookkeeping params
    #Base output directory
    out_dir            = home_dir + "/mountData/"
    #Inner run directory
    run_dir            = out_dir + "/deep_sc_cifar/"

    #Save parameters
    save_period        = 10000
    #output plots directory
    plot_period        = 10000
    eval_period        = 1000 # 1 epoch
    #Progress step
    progress           = 10
    #Controls how often to write out to tensorboard
    write_step         = 100
    #Flag for loading weights from checkpoint
    load               = False
    load_file          = ""
    #Device to run on
    device             = "/gpu:0"
    num_steps          = 1000000

    #Model params
    batch_size = 8
    image_shape = dataObj.image_shape
    num_features = dataObj.num_features

    #SC params
    sc_iter = 1000
    sc_verbose = False
    sc_lr = 5e-3
    D_lr = 1e-3

    num_layers = 7
    dict_patch_size=[13, 7, 7, 5, None, None, None] #make sure patchsize is odd
    err_weight = [1, None, 1, None, 1, None, 1]
    act_weight = [1, .5, 1, .5, 1, .5, 1]
    l1_weight = [.1, None, .1, None, .1, None, .01]
    l2_weight = 0
    stride = [4, 2, 4, 1, None, None, None]
    dict_size = [256, 16, 256, 16, 512, 16, 4] #dict_size / (stride * num_input_features) = overcompletness
    layer_type = ["sc_conv", "conv", "sc_conv", "conv", "sc_fc", "fc", "sc_fc"]
    normalize_act = [False, True, False, True, False, True, False]

    target_norm_std = .5

    norm_input = False

#Initialize params
params = Params()
modelObj = deepSparseCode(params)
modelObj.trainModel(dataObj)


