import matplotlib
matplotlib.use('Agg')
from dataObj.obspy_seismic import obspySeismicData
from plots.plotWeights import plot_1d_weights
#from plot.roc import makeRocCurve
import numpy as np
import pdb
from sklearn.decomposition import FastICA, PCA
import os

filename = "/home/slundquist/mountData/datasets/CanadianData_tiny.txt"
example_size = 1024
num_samples = 50000
numComponents = 256

outDir=          "/home/slundquist/mountData/tfSparseCode/"
#Inner run directory
runDir=          outDir + "/icapca_obspy_seismic/"
#output plots directory
plotDir=         runDir + "plots/"

if not os.path.exists(runDir):
   os.makedirs(runDir)

if not os.path.exists(plotDir):
   os.makedirs(plotDir)

#Get object from which tensorflow will pull data from
trainDataObj = obspySeismicData(filename, example_size)

print("Getting data")
#Build data obj
#TODO see if we can do this incrementally
data = trainDataObj.getData(num_samples)
print("Done")

#Match magnitude of lca
#.4 / sqrt(patchsize)
data = data * 0.004419417382


[drop, drop, sampleSize, numFeatures] = data.shape

r_data = np.reshape(data, [num_samples, -1])

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

