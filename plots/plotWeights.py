import numpy as np
import matplotlib.pyplot as plt
import pdb

#Order defines the order in weights_matrix for num_weights, t, y, x, f
def plot_weights_time(weights_matrix, outPrefix, order=[0, 1, 2, 3, 4]):
    assert(weights_matrix.ndim == 5)
    num_weights = weights_matrix.shape[order[0]]
    patch_t = weights_matrix.shape[order[1]]
    patch_y = weights_matrix.shape[order[2]]
    patch_x = weights_matrix.shape[order[3]]
    patch_f = weights_matrix.shape[order[4]]

    if(order == [0, 1, 2, 3, 4]):
        permute_weights = weights_matrix
    else:
        permute_weights = weights_matrix.copy()
        permute_weights = np.transpose(weights_matrix, order)

    for t in range(patch_t):
        outFilename = outPrefix + "_frame" + str(t) + ".png"
        plot_weights(permute_weights[:, t, :, :, :], outFilename)

#Order defines the order in weights_matrix for num_weights, y, x, f
def plot_weights(weights_matrix, outFilename, order=[0, 1, 2, 3]):
    print("Creating plot")
    assert(weights_matrix.ndim == 4)
    num_weights = weights_matrix.shape[order[0]]
    patch_y = weights_matrix.shape[order[1]]
    patch_x = weights_matrix.shape[order[2]]
    patch_f = weights_matrix.shape[order[3]]

    if(order == [0, 1, 2, 3]):
        permute_weights = weights_matrix
    else:
        permute_weights = weights_matrix.copy()
        permute_weights = np.transpose(weights_matrix, order)

    subplot_x = int(np.ceil(np.sqrt(num_weights)))
    subplot_y = int(np.ceil(num_weights/float(subplot_x)))

    outWeightMat = np.zeros((patch_y*subplot_y, patch_x*subplot_x, patch_f)).astype(np.float32)

    #Normalize each patch individually
    for weight in range(num_weights):
        weight_y = int(weight/subplot_x)
        weight_x = weight%subplot_x

        startIdx_x = weight_x*patch_x
        endIdx_x = startIdx_x+patch_x

        startIdx_y = weight_y*patch_y
        endIdx_y = startIdx_y+patch_y

        weight_patch = permute_weights[weight, :, :, :].astype(np.float32)

        #Set mean to 0
        weight_patch = weight_patch - np.mean(weight_patch)
        scaleVal = np.max([np.fabs(weight_patch.max()), np.fabs(weight_patch.min())])
        weight_patch = weight_patch / (scaleVal+1e-6)
        weight_patch = (weight_patch + 1)/2

        outWeightMat[startIdx_y:endIdx_y, startIdx_x:endIdx_x, :] = weight_patch

    if(patch_f == 1 or patch_f == 3):
        fig = plt.figure()
        plt.imshow(outWeightMat)
        plt.savefig(outFilename)
        plt.close(fig)
    else:
        for f in range(patch_f):
            fig = plt.figure()
            outMat = outWeightMat[:, :, f]
            plt.imshow(outMat, cmap="gray")
            plt.savefig(outFilename + "f_" + str(f) + ".png")
            plt.close(fig)

    fig = plt.figure()
    plt.hist(weights_matrix.flatten(), 50)
    plt.savefig(outFilename + ".hist.png")
    plt.close(fig)

COLORS=[[0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [.5, .5, .5]]

#Order defines the order in weights_matrix for num_weights, v
#Order should be in [num_weights, patch_size, num_f]
#Activity has to be in (batch, x, f)
def plotWeights1D(weights_matrix, out_prefix, order=[0, 1, 2], activity_count=None, group_policy="group", groups=None, group_title=None, num_plot=10, legend=None):

    if(groups is None or group_policy != "group"):
        groups = [range(nf)]
    if(group_title is None):
        group_title= ["station_" + str(g[0]) for g in groups]

    num_weights = weights_matrix.shape[order[0]]
    num_weights = min(num_weights, num_plot)
    patch_size = weights_matrix.shape[order[1]]
    numf = weights_matrix.shape[order[2]]

    if(order == [0, 1, 2, 3]):
        permute_weights = weights_matrix
    else:
        permute_weights = weights_matrix.copy()
        permute_weights = np.transpose(weights_matrix, order)

    if(activity_count is not None):
        sort_idxs = np.argsort(activity_count)
        #This sorts from smallest to largest, reverse
        sort_idxs = sort_idxs[::-1]
        sort_activity_count = activity_count[sort_idxs]
        #make bar chart of act count
        ind = np.arange(1, weights_matrix.shape[order[0]]+1)
        fig = plt.figure()
        plt.bar(ind, sort_activity_count)
        outfn = out_prefix + "_act_count.png"
        plt.savefig(outfn)
        plt.close("all")
    else:
        sort_idxs = range(num_weights)

    for weight in range(num_weights):
        weightIdx = sort_idxs[weight]
        if(group_policy=="single"):
            fig, axs = plt.subplots(numf, 1, sharex=True, sharey=True, figsize=(8, 12))
            for f in range(numf):
                axs[f].plot(permute_weights[weightIdx, :, f], color=COLORS[f%len(COLORS)])
                axs[f].set_title(group_title[f])
            plt.subplots_adjust(wspace=0, hspace=0)
            plt.savefig(out_prefix + "_weight"+str(weight)+".png")
        elif(group_policy=="all"):
            fig = plt.figure()
            for f in range(numf):
                plt.plot(permute_weights[weightIdx, :, f], color=COLORS[f%len(COLORS)])
            lgd = plt.legend(group_title, loc=1, bbox_to_anchor=(1.6, 1))
            plt.savefig(out_prefix + "_weight"+str(weight)+".png", additional_artists=[lgd])
        elif(group_policy=="group"):
            num_groups = len(groups)
            num_y_axes = int(np.ceil(num_groups/2))
            fig, axs = plt.subplots(num_y_axes, 2, sharex=True, sharey=True, squeeze=False, figsize=(16, 3*num_y_axes))
            for i_g, g in enumerate(groups):
                y_idx = i_g//2
                x_idx = i_g % 2
                legend_lines = []
                for i_f, f in enumerate(g):
                    l = axs[y_idx, x_idx].plot(permute_weights[weightIdx, :, f], color=COLORS[i_f%len(COLORS)])
                    legend_lines.append(l[0])
                axs[y_idx, x_idx].set_title(group_title[i_g])
            if(legend is not None):
                lgd = fig.legend(legend_lines, legend, "upper right")
                plt.savefig(out_prefix + "_weight"+str(weight)+".png", additional_artists=[lgd])
            else:
                plt.savefig(out_prefix + "_weight"+str(weight)+".png")

        plt.close("all")
        #np.savetxt(outFilename + "weight"+str(weight)+".txt", permute_weights[weightIdx, :], delimiter=",")

