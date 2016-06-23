import numpy as np
import matplotlib.pyplot as plt
import pdb

numY = 4

def make_plot_time(weights_matrix, outFilename):
    print "Creating plot"
    assert(weights_matrix.ndim == 5)
    num_weights = weights_matrix.shape[3]
    patch_t = weights_matrix.shape[0]
    patch_y = weights_matrix.shape[1]
    patch_x = weights_matrix.shape[2]
    patch_f = weights_matrix.shape[4]

    subplot_y = numY
    subplot_x = np.ceil(num_weights/numY)

    outWeightMat = np.zeros((patch_y*subplot_y*patch_t, patch_x*subplot_x, patch_f))
    #Normalize each patch individually
    for weight in range(num_weights):
        weight_y = weight/subplot_x
        weight_x = weight%subplot_x

        #X here doesn't change
        startIdx_x = weight_x*patch_x
        endIdx_x = startIdx_x+patch_x

        for t in range(patch_t):
            startIdx_y = (weight_y+t)*patch_y
            endIdx_y = startIdx_y+patch_y
            weight_patch = weights_matrix[t, :, :, weight, :]
            scale_factor = (weight_patch.max() - weight_patch.min())
            weight_patch = (weight_patch - weight_patch.min()) / scale_factor
            outWeightMat[startIdx_y:endIdx_y, startIdx_x:endIdx_x, :] = weight_patch

    plt.imshow((outWeightMat * 255).astype(np.uint8))
    plt.savefig(outFilename)



def make_plot(weights_matrix, outFilename):
    print "Creating plot"
    assert(weights_matrix.ndim == 4)
    num_weights = weights_matrix.shape[3]
    patch_y = weights_matrix.shape[0]
    patch_x = weights_matrix.shape[1]
    patch_f = weights_matrix.shape[2]

    subplot_x = int(np.ceil(np.sqrt(num_weights)))
    subplot_y = int(np.ceil(num_weights/float(subplot_x)))
    weights_list = list()



    outWeightMat = np.zeros((patch_y*subplot_y, patch_x*subplot_x, patch_f))

    #Normalize each patch individually
    for weight in range(num_weights):
        weight_y = weight/subplot_x
        weight_x = weight%subplot_x

        startIdx_x = weight_x*patch_x
        endIdx_x = startIdx_x+patch_x

        startIdx_y = weight_y*patch_y
        endIdx_y = startIdx_y+patch_y

        weight_patch = weights_matrix[:, :, :, weight]

        scale_factor = (weight_patch.max() - weight_patch.min())
        weight_patch = (weight_patch - weight_patch.min()) / scale_factor
        outWeightMat[startIdx_y:endIdx_y, startIdx_x:endIdx_x, :] = weight_patch

    plt.imshow((outWeightMat * 255).astype(np.uint8))
    plt.savefig(outFilename)
