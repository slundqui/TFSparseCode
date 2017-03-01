import numpy as np
import matplotlib.pyplot as plt
import scipy.misc as spmisc
import pdb
from tf.utils import makeDir

#Order defines the order in weights_matrix for num_weights, t, y, x, f
def plot_weights_time(weights_matrix, outPrefix, order=[0, 1, 2, 3, 4], v1Rank=None, plotInd=False):
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
        outFilename = outPrefix + "_frame" + str(t)
        plot_weights(permute_weights[:, t, :, :, :], outFilename, v1Rank)

#Order defines the order in weights_matrix for num_weights, y, x, f
def plot_weights(weights_matrix, outFilename, order=[0, 1, 2, 3], v1Rank=None, plotInd = False):
    print "Creating plot"
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

    if v1Rank is not None:
        #Flatten and count v1
        axis = () #empty tuple
        for i in range(v1Rank.ndim-1):
            axis += (i,)
        rankVals = np.sum(v1Rank, axis=axis)
        rankIdx = np.argsort(rankVals)[::-1]
        #Make val histogram
        x = range(num_weights)
        y = [rankVals[i] for i in rankIdx]
        fig = plt.figure()
        plt.bar(x, y)
        plt.savefig(outFilename+"_v1Hist.png")
        plt.close(fig)

    subplot_x = int(np.ceil(np.sqrt(num_weights)))
    subplot_y = int(np.ceil(num_weights/float(subplot_x)))

    outWeightMat = np.zeros((patch_y*subplot_y, patch_x*subplot_x, patch_f)).astype(np.float32)

    if v1Rank is not None:
        rangeWeight = rankIdx;
    else:
        rangeWeight = range(num_weights)

    if(plotInd):
        indOutDir = outFilename + "_ind/"
        makeDir(indOutDir)

    #Normalize each patch individually
    for weight in range(num_weights):
        weightIdx = rangeWeight[weight]
        weight_y = weight/subplot_x
        weight_x = weight%subplot_x

        startIdx_x = weight_x*patch_x
        endIdx_x = startIdx_x+patch_x

        startIdx_y = weight_y*patch_y
        endIdx_y = startIdx_y+patch_y

        weight_patch = permute_weights[weightIdx, :, :, :].astype(np.float32)

        #weight_patch = weight_patch - np.mean(weight_patch)
        #Find max magnitude
        scaleVal = np.max([np.fabs(weight_patch.max()), np.fabs(weight_patch.min())])
        #Scale to be between -1 and 1
        weight_patch = weight_patch / scaleVal
        #Set scale to be between 0 and 1, with 0 in orig weight_patch to be .5
        weight_patch = (weight_patch + 1)/2

        if plotInd:
            if v1Rank is None:
                saveStr = indOutDir + "weight_" + str(weightIdx) + ".png"
            else:
                saveStr = indOutDir + "rank_" + str(weight) + "_weight_" + str(weightIdx) + ".png"
            print saveStr
            spmisc.imsave(saveStr, weight_patch)
            #fig = plt.figure()
            #plt.imshow(weight_patch)
            #plt.savefig(saveStr)
            #plt.close(fig)

        outWeightMat[startIdx_y:endIdx_y, startIdx_x:endIdx_x, :] = weight_patch

    fig = plt.figure()
    plt.imshow(outWeightMat)
    plt.savefig(outFilename + ".png")
    plt.close(fig)

    fig = plt.figure()
    plt.hist(weights_matrix.flatten(), 50)
    plt.savefig(outFilename + "_hist.png")
    plt.close(fig)


