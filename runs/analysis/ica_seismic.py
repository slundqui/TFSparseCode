import matplotlib
matplotlib.use('Agg')
from dataObj.seismic import seismicData
from plots.plotWeights import plot_1d_weights
#from plot.roc import makeRocCurve
import numpy as np
import pdb
from sklearn.decomposition import FastICA, PCA
import os

#trainImageLists =  "/home/sheng/mountData/datasets/cifar/images/train.txt"
#testImageLists = "/home/slundquist/mountData/datasets/cifar/images/test.txt"
randImageSeed = None

filename = "/home/slundquist/mountData/datasets/seismic/wf.txt"
settingsFilename = "/home/slundquist/mountData/datasets/seismic/p4681_run1_AE.mat"

exampleSize = 1024
numSamples = 10000
numComponents = 256

outDir=          "/home/slundquist/mountData/tfSparseCode/"
#Inner run directory
runDir=          outDir + "/icapca_seismic_ps1024_nf256_inputmult/"
#output plots directory
plotDir=         runDir + "plots/"

if not os.path.exists(runDir):
   os.makedirs(runDir)

if not os.path.exists(plotDir):
   os.makedirs(plotDir)

print("Building data")
#Get object from which tensorflow will pull data from
trainDataObj = seismicData(filename, settingsFilename, exampleSize, shuffle=True)

#Build data obj
#TODO see if we can do this incrementally
data = trainDataObj.getData(numSamples)

#Match magnitude of lca
data = data / 90.5

print("Done")

[drop, drop, sampleSize, numFeatures] = data.shape

r_data = np.reshape(data, [numSamples, -1])

print("Running ICA")
ica = FastICA(n_components = numComponents)
ica.fit(r_data)
r_ica_weights = ica.components_
ica_weights = np.reshape(r_ica_weights, [numComponents, sampleSize, numFeatures])
print("Done")

print("Running PCA")
pca=PCA(n_components=numComponents)
pca.fit(r_data)
r_pca_weights = pca.components_
pca_weights = np.reshape(r_pca_weights, [numComponents, sampleSize, numFeatures])
print("Done")

print("Plotting")
icaDir = plotDir + "/ica/"
if not os.path.exists(icaDir):
   os.makedirs(icaDir)
plot_1d_weights(ica_weights, icaDir, sepFeatures=True)

pcaDir = plotDir + "/pca/"
if not os.path.exists(pcaDir):
   os.makedirs(pcaDir)
plot_1d_weights(pca_weights, pcaDir, sepFeatures=True)
print("Done")

