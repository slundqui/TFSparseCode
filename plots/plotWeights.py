import numpy as np
import matplotlib.pyplot as plt
import pdb

def make_plot(weights_matrix, outFilename):
    print "Creating plot"
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
