import numpy as np
import matplotlib.pyplot as plt
import pdb

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

#Recon must be in (batch, time)
def plotRecon1d(recon_matrix, img_matrix, outPrefix, r=None):
    (batch, nt, nf) = recon_matrix.shape
    (batchImg, ntImg, nfImg) = img_matrix.shape

    if r == None:
        r = range(batch)

    for b in r:
        recon = recon_matrix[b]
        img = img_matrix[b]
        f, axarr = plt.subplots(2, 1)
        axarr[0].set_title("orig")
        axarr[1].set_title("recon")
        #Plot each feature as a different color
        for f in range(nf):
            axarr[0].plot(img[:, f], color=colors[f%8])
            axarr[1].plot(recon[:, f], color=colors[f%8])
        plt.savefig(outPrefix+"_"+str(b)+".png")
        plt.close(f)
        np.savetxt(outPrefix + "orig"+str(b)+".txt", img, delimiter=",")
        np.savetxt(outPrefix + "recon"+str(b)+".txt", recon, delimiter=",")




