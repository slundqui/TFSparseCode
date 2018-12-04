import numpy as np
import matplotlib.pyplot as plt
import pdb
from scipy import stats
import pandas as pd

def plotRecon(recon_matrix, img_matrix, outPrefix, r=None):
    (batch, ny, nx, nf) = recon_matrix.shape
    (batchImg, nyImg, nxImg, nfImg) = img_matrix.shape
    assert(batch == batchImg)
    assert(nf == nfImg)

    if r == None:
        r = range(batch)

    for b in r:
        img = img_matrix[b, :, :, :]
        r_img = (img-img.min())/(img.max()-img.min()+1e-6)
        recon = recon_matrix[b, :, :, :]
        #Plot recon with img stats
        r_recon = (recon-img.min())/(img.max()-img.min()+1e-6)
        #Clamp values to not be out of bounds
        r_recon = np.clip(r_recon, 0.0, 1.0)
        if(nf == 3):
            fig, axarr = plt.subplots(2, 1)
            axarr[0].imshow(r_img)
            axarr[0].set_title("orig")
            axarr[1].imshow(r_recon)
            axarr[1].set_title("recon")
            plt.savefig(outPrefix+"_b"+str(b)+".png")
            plt.close(fig)
        else:
            for f in range(nf):
                fig, axarr = plt.subplots(2, 1)
                axarr[0].imshow(r_img[:, :, f], cmap="gray")
                axarr[0].set_title("orig")
                axarr[1].imshow(r_recon[:, :, f], cmap="gray")
                axarr[1].set_title("recon")
                plt.savefig(outPrefix+"_f" + str(f) + "_b"+str(b)+".png")
                plt.close(fig)


colors=[[0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [.5, .5, .5]]

# define window function:
#data is defined to be [time, features]
def sliding_window(data, window):
    current_pos = 0
    left_pos = 0
    win_size = window
    right_pos = left_pos + win_size
    vdata = []
    mdata = []
    while current_pos < len(data-win_size):
        left_pos = current_pos
        right_pos = left_pos + win_size
        mean = np.mean(np.var(data[left_pos:right_pos,:], axis=0))
        var = np.var(data[left_pos:right_pos,:], axis=0)
        vdata.append(var)
        mdata.append(mean)
        current_pos += 1
    return np.array(vdata), np.array(mdata)

#Recon must be in (batch, time, features)
def plotRecon1D(recon_matrix, img_matrix, outPrefix, num_plot=None, x_range=None, unscaled_img_matrix=None, unscaled_recon_matrix=None, var_window=20, mask_matrix=None, groups=None, group_title=None, legend=None):
    (batch, nt, nf) = recon_matrix.shape
    (batchImg, ntImg, nfImg) = img_matrix.shape

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
        img = img_matrix[b]
        if(mask_matrix is not None):
            mask = mask_matrix[b]
        else:
            mask = np.zeros(recon.shape)
        if(x_range is not None):
            recon = recon[x_range[0]:x_range[1], :]
            img = img[x_range[0]:x_range[1], :]
        error = img - recon

        for i_g, g in enumerate(groups):
            fig, axarr = plt.subplots(4, 1)
            fig.suptitle(group_title[i_g])
            axarr[0].set_title("orig")
            axarr[1].set_title("recon")
            axarr[2].set_title("diff")
            axarr[3].set_title("mask")
            outGroupPrefix = outPrefix+"_"+group_title[i_g]+"_batch"+str(b)

            #Plot each feature as a different color
            legend_lines = []
            for i_f, f in enumerate(g):
                l = axarr[0].plot(img[:, f], color=colors[i_f%8])
                axarr[1].plot(recon[:, f], color=colors[i_f%8])
                axarr[2].plot(error[:, f], color=colors[i_f%8])
                axarr[3].plot(mask[:, f], color=colors[i_f%8])

                legend_lines.append(l[0])

            if(legend is not None):
                lgd = fig.legend(legend_lines, legend, "upper right")
                plt.savefig(outGroupPrefix+"_scaled.png", additional_artists=[lgd])
            else:
                plt.savefig(outGroupPrefix+"_scaled.png")
            plt.close('all')

            if(unscaled_img_matrix is not None):
                unscaled_img = unscaled_img_matrix[b]
                unscaled_recon = unscaled_recon_matrix[b]
                unscaled_error = unscaled_img - unscaled_recon
                fig, axarr = plt.subplots(4, 1)
                fig.suptitle(group_title[i_g])
                axarr[0].set_title("unscaled_orig")
                axarr[1].set_title("unscaled_recon")
                axarr[2].set_title("unscaled_diff")
                axarr[3].set_title("mask")
                #Plot each feature as a different color
                for i_f, f in enumerate(g):
                    axarr[0].plot(unscaled_img[:, f], color=colors[i_f%8])
                    axarr[1].plot(unscaled_recon[:, f], color=colors[i_f%8])
                    axarr[2].plot(unscaled_error[:, f], color=colors[i_f%8])
                    axarr[3].plot(mask[:, f], color=colors[i_f%8])
                if(legend is not None):
                    lgd = fig.legend(legend_lines, legend, "upper right")
                    plt.savefig(outGroupPrefix+"_unscaled.png", additional_artists=[lgd])
                else:
                    plt.savefig(outGroupPrefix+"_unscaled.png")
                plt.close('all')

                #Plot variance of img and recon with sliding window
                [img_variance, img_mean] = sliding_window(unscaled_img, var_window)
                [recon_variance, recon_mean] = sliding_window(unscaled_recon, var_window)
                [error_variance, error_mean] = sliding_window(unscaled_error, var_window)
                fig, axarr = plt.subplots(3, 1)
                fig.suptitle(group_title[i_g])
                axarr[0].set_title("unscaled_orig_var")
                axarr[1].set_title("unscaled_recon_var")
                axarr[2].set_title("unscaled_diff_var")
                #Plot each feature as a different color
                for i_f, f in enumerate(g):
                    axarr[0].plot(img_variance[:, f], color=colors[i_f%8])
                    axarr[1].plot(recon_variance[:, f], color=colors[i_f%8])
                    axarr[2].plot(error_variance[:, f], color=colors[i_f%8])
                if(legend is not None):
                    lgd = fig.legend(legend_lines, legend, "upper right")
                    plt.savefig(outGroupPrefix+"_var.png", additional_artists=[lgd])
                else:
                    plt.savefig(outGroupPrefix+"_var.png")
                plt.close('all')



