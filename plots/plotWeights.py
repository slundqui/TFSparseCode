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

    subplot_x = int(np.ceil(np.sqrt(num_weights)))
    subplot_y = int(np.ceil(num_weights/float(subplot_x)))

    outWeightMat = np.zeros((patch_y*subplot_y, patch_x*subplot_x, patch_f)).astype(np.float32)

    #Normalize each patch individually
    for weight in range(num_weights):
        weight_y = weight/subplot_x
        weight_x = weight%subplot_x

        startIdx_x = weight_x*patch_x
        endIdx_x = startIdx_x+patch_x

        startIdx_y = weight_y*patch_y
        endIdx_y = startIdx_y+patch_y

        weight_patch = permute_weights[weight, :, :, :].astype(np.float32)

        #Set mean to 0
        weight_patch = weight_patch - np.mean(weight_patch)
        scaleVal = np.max([np.fabs(weight_patch.max()), np.fabs(weight_patch.min())])
        weight_patch = weight_patch / scaleVal
        weight_patch = (weight_patch + 1)/2

        outWeightMat[startIdx_y:endIdx_y, startIdx_x:endIdx_x, :] = weight_patch

    fig = plt.figure()
    plt.imshow(outWeightMat)
    plt.savefig(outFilename)
    plt.close(fig)

    fig = plt.figure()
    plt.hist(weights_matrix.flatten(), 50)
    plt.savefig(outFilename + ".hist.png")
    plt.close(fig)

#Order defines the order in weights_matrix for num_weights, v
#Activity has to be in (batch, y, x, f)
def plot_1d_weights(weights_matrix, outFilename, order=[0, 1], activity=None):
    numWeights = weights_matrix.shape[order[0]]
    patchSize = weights_matrix.shape[order[1]]
    if(order == [0, 1, 2, 3]):
        permute_weights = weights_matrix
    else:
        permute_weights = weights_matrix.copy()
        permute_weights = np.transpose(weights_matrix, order)

    if(activity is not None):
        c_activity = activity.copy()
        c_activity[np.nonzero(activity > 0)] = 1
        count = np.sum(c_activity, axis=(0, 1, 2))
        sortIdxs = np.argsort(count)
        #This sorts from smallest to largest, reverse
        sortIdxs = sortIdxs[::-1]
    else:
        sortIdxs = range(numWeights)

    for weight in range(numWeights):
        weightIdx = sortIdxs[weight]
        fig = plt.figure()
        plt.plot(permute_weights[weightIdx, :])
        plt.savefig(outFilename + "weight"+str(weight)+".png")
        plt.close(fig)
        np.savetxt(outFilename + "weight"+str(weight)+".txt", permute_weights[weightIdx, :], delimiter=",")

