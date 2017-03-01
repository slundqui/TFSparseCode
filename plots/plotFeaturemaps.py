import numpy as np
import matplotlib.pyplot as plt
import pdb
from scipy.misc import imresize, imsave

#activation is [batch, y, x, features]
#image is [batch, y, x, colors]
def plotFeaturemaps(activation, image, prefix, r=None):
    #make dir if not there
    #Plot parameters
    sliceWeight = .7
    baseWeight = .8

    (batch, nyAct, nxAct, nfAct) = activation.shape
    (batchImg, nyImg, nxImg, nfImg) = image.shape

    assert(batch == batchImg)

    if r == None:
        r = range(batch)

    for b in r:
        filename = prefix + "_batch_" + str(b)
        singleImage = image[b, :, :, :]
        rescaleImg = ((singleImage / (singleImage.max() - singleImage.min())) + 1) / 2
        for n in range(nfAct):
            singleAct = np.zeros([nyAct, nxAct, 3])
            #Mark activations as green
            singleAct[:, :, 1] = activation[b, :, :, n]
            #imresize also rescales the image
            resizeAct = imresize(singleAct, [nyImg, nxImg])
            #Change back to 0-1 range
            rescaleAct = resizeAct.astype(np.float64)/255

            #Scale image
            #scaleAct = singleAct / (singleAct.max() - singleAct.min())

            #Weight and scale
            finalImage = baseWeight * rescaleImg + sliceWeight * rescaleAct
            #Clamp max value
            finalImage[np.nonzero(finalImage > 1)] = 1
            #Save image
            saveStr = filename + "_weight_" + str(n) + ".png"
            print saveStr
            imsave(saveStr, finalImage)
