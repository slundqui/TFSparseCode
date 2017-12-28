import numpy as np
import matplotlib.pyplot as plt
import pdb
from plots.utils import tensorToComplex

def plotRecon(recon_matrix, img_matrix, outPrefix, r=None):
    (batch, ny, nx, nf) = recon_matrix.shape
    (batchImg, nyImg, nxImg, nfImg) = img_matrix.shape
    assert(batch == batchImg)
    if r == None:
        r = range(batch)

    for b in r:
        recon = recon_matrix[b, :, :, :]
        r_recon = (recon-recon.min())/(recon.max()-recon.min())
        img = img_matrix[b, :, :, :]
        r_img = (img-img.min())/(img.max()-img.min())
        f, axarr = plt.subplots(2, 1)
        axarr[0].imshow(r_img)
        axarr[0].set_title("orig")
        axarr[1].imshow(r_recon)
        axarr[1].set_title("recon")
        plt.savefig(outPrefix+"_"+str(b)+".png")
        plt.close(f)

colors=[[0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [.5, .5, .5]]




#Recon must be in (batch, time)
def plotRecon1d(recon_matrix, img_matrix, outPrefix, r=None, fourier=False, origImg=None):
    if(fourier):
        recon_matrix = tensorToComplex(recon_matrix)
        #real part of ifft
        recon_matrix = np.real(np.fft.ifft(recon_matrix, axis=1))

        img_matrix = tensorToComplex(img_matrix)
        #real part of ifft
        img_matrix = np.real(np.fft.ifft(img_matrix, axis=1))

    (batch, nt, nf) = recon_matrix.shape
    (batchImg, ntImg, nfImg) = img_matrix.shape

    if r == None:
        r = range(batch)

    for b in r:
        recon = recon_matrix[b]
        img = img_matrix[b]
        if(origImg is not None):
            o_img = origImg[b]
            f, axarr = plt.subplots(3, 1)
            axarr[0].set_title("orig")
            axarr[1].set_title("ifft")
            axarr[2].set_title("recon")
        else:
            f, axarr = plt.subplots(2, 1)
            axarr[0].set_title("orig")
            axarr[1].set_title("recon")

        #Plot each feature as a different color
        for f in range(nf):
            if(origImg is not None):
                axarr[0].plot(o_img[:, f], color=colors[f%8])
                axarr[1].plot(img[:, f], color=colors[f%8])
                axarr[2].plot(recon[:, f], color=colors[f%8])
            else:
                axarr[0].plot(img[:, f], color=colors[f%8])
                axarr[1].plot(recon[:, f], color=colors[f%8])
        plt.savefig(outPrefix+"_"+str(b)+".png")
        plt.close(f)
        #if(origImg is not None):
        #    np.savetxt(outPrefix + "o_orig"+str(b)+".txt", o_img, delimiter=",")
        #np.savetxt(outPrefix + "orig"+str(b)+".txt", img, delimiter=",")
        #np.savetxt(outPrefix + "recon"+str(b)+".txt", recon, delimiter=",")




