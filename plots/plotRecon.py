import numpy as np
import matplotlib.pyplot as plt
import pdb

def plotRecon(recon_matrix, img_matrix, outPrefix):
    (batch, ny, nx, nf) = recon_matrix.shape
    (batchImg, nyImg, nxImg, nfImg) = img_matrix.shape
    assert(batch == batchImg)

    for b in range(batch):
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

