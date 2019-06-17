import matplotlib
matplotlib.use('Agg')
from data.seismic_hdf5 import SeismicDataHdf5
from models.sparseCode import sparseCode
import numpy as np
import matplotlib.pyplot as plt
from plots.plotRecon import plotRecon1D
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
    data_save_fn=data_save_fn, loc_filter=True, sample_for_class=False, prediction=False)

class Params(object):
    #Base output directory
    out_dir = home_dir + "/mountData/tfSparseCode/"
    #Inner run directory
    run_dir = out_dir + "/sc_hdf5_seismic_single_loc_classifier_detection_eval/"
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
    load_file = home_dir + "/mountData/tfSparseCode/sc_hdf5_seismic_single_loc_classifier_detection_less_batch/checkpoints/save-model-9000"
    #Device to run on
    device = '/gpu:0'
    num_steps = 10000000

    legend = ["HHE", "HHN", "HHZ"]
    num_plot_weights = 10
    num_plot_recon = 4
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
    load_classifier=True
    use_classifier=True
    class_lr = 1e-3



params = Params()

#Allocate tensorflow object
tfObj = sparseCode(params)
print("Done init")

class_w = tfObj.sess.run(tfObj.classifier_weights)
#0 class is negative, 1 class is positive
#Grab positive weights
#TODO should we grab negative weights here?
pos_w = class_w[:, 1]

##Make histogram of these weights
#plt.hist(pos_w)
#filename = params.run_dir + "/pos_w_hist.png"
#plt.savefig(filename)

#Grab weights of interest
#Positive only TODO filter in different ways
pos_dict_idxs = np.nonzero(pos_w > 0)
neg_dict_idxs = np.nonzero(pos_w < 0)

#Get sample
data = trainDataObj.getData(params.batch_size)
gt = data["gt"]
print(gt)
input_feed_dict = tfObj.getEvalFeedDict(data["data"])
#Calculate activations
act = tfObj.evalModel(input_feed_dict)
norm_input = tfObj.sess.run(tfObj.norm_input, input_feed_dict)
recons = tfObj.sess.run(tfObj.scObj.model["recon"])

#TODO filter activity here
pos_filter_act = np.copy(act)
pos_filter_act[:, :, neg_dict_idxs] = 0
neg_filter_act = np.copy(act)
neg_filter_act[:, :, pos_dict_idxs] = 0

feed_dict = {}
feed_dict[tfObj.scObj.model["inject_act_placeholder"]] = pos_filter_act
pos_recon = tfObj.sess.run(tfObj.scObj.model["recon_from_act"], feed_dict)

feed_dict[tfObj.scObj.model["inject_act_placeholder"]] = neg_filter_act
neg_recon = tfObj.sess.run(tfObj.scObj.model["recon_from_act"], feed_dict)

#Recon must be in (batch, time, features)
def plotRecons(recon_matrix, img_matrix, pos_recon_matrix, neg_recon_matrix, outPrefix, num_plot=None, groups=None, group_title=None, legend=None):

    colors=[[0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [.5, .5, .5]]

    (batch, nt, nf) = recon_matrix.shape

    if(groups is None):
        groups = [range(nf)]
    if(group_title is None):
        group_title= ["station_" + str(g[0]) for g in groups]

    if num_plot == None:
        r = range(batch)
    elif type(num_plot) == int:
        r = range(num_plot)
    else:
        r = num_plot

    for b in r:
        recon = recon_matrix[b]
        pos_recon = pos_recon_matrix[b]
        neg_recon = neg_recon_matrix[b]
        img = img_matrix[b]

        for i_g, g in enumerate(groups):
            fig, axarr = plt.subplots(4, 1)
            fig.suptitle(group_title[i_g])
            axarr[0].set_title("orig")
            axarr[1].set_title("recon")
            axarr[2].set_title("pos recon")
            axarr[3].set_title("neg recon")
            outGroupPrefix = outPrefix+"_"+group_title[i_g]+"_batch"+str(b)

            #Plot each feature as a different color
            legend_lines = []
            for i_f, f in enumerate(g):
                #Find max/min of each plot
                val_max = np.max([img[:, f], recon[:, f]])
                val_min = np.min([img[:, f], recon[:, f]])

                l = axarr[0].plot(img[:, f], color=colors[i_f%8])
                axarr[1].plot(recon[:, f], color=colors[i_f%8])
                axarr[2].plot(pos_recon[:, f], color=colors[i_f%8])
                axarr[3].plot(neg_recon[:, f], color=colors[i_f%8])

                axarr[0].set_ylim([val_min, val_max])
                axarr[1].set_ylim([val_min, val_max])
                axarr[2].set_ylim([val_min, val_max])
                axarr[3].set_ylim([val_min, val_max])

                legend_lines.append(l[0])

            if(legend is not None):
                lgd = fig.legend(legend_lines, legend, "upper right")
                plt.savefig(outGroupPrefix+"_scaled.png", additional_artists=[lgd])
            else:
                plt.savefig(outGroupPrefix+"_scaled.png")
            plt.close('all')


fn_prefix = params.run_dir + "/plots/"
#Plot
plotRecons(recons, norm_input, pos_recon, neg_recon, fn_prefix+"pos_recon",
        num_plot = params.num_plot_recon,
        groups=params.plot_groups, group_title=params.plot_group_title,
        legend=params.legend)


tfObj.closeSess()

