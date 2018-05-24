import random
import pdb
import numpy as np
from scipy.io import loadmat
import os
import TFSparseCode.dataObj.utils as utils

class seismicData(object):
    #filename is a list of filenames that contain seismic data
    def __init__(self, filename, settingsFn, exampleSize, shuffle, seed=None, scaleByChannel=False):
        if seed is not None:
            random.seed(seed)
        #TODO fix to allow for multiple streams
        self.exampleSize = exampleSize
        settings = loadmat(settingsFn)
        self.numFrames = int(settings['numFrames'][0,0]/2)
        self.numChannels = settings['channels2save'].size
        self.inputShape = [1, exampleSize, self.numChannels]
        #Read list of filenames and store into member variable
        self.fnList = utils.readList(filename)
        self.numFiles = len(self.fnList)
        #Shuffle files
        self.doShuffle = shuffle
        self.shuffleFnIdx = list(range(len(self.fnList)))
        self.fnIdx = 0
        self.scaleByChannel = scaleByChannel

        if(self.doShuffle):
            random.shuffle(self.shuffleFnIdx)

    def getExample(self):
        numSamples = -np.inf
        sampleAgain = False
        #Only take files that are >= exampleSize
        while(numSamples < self.exampleSize):
            if(sampleAgain):
                print("Skipping file", self.current_filename, "as it only contains", numSamples, "samples in the file")
            self.current_filename = self.fnList[self.shuffleFnIdx[self.fnIdx]]
            #filename = "/media/data/jamal/p4681/p4681ac/run1/WF_100.ac"
            self.fnIdx += 1
            if(self.fnIdx >= len(self.fnList)):
                self.fnIdx = 0
                print("Rewinding")

            #Load file
            data = np.fromfile(self.current_filename, dtype=np.int16)
            data = np.reshape(data, [self.numFrames, self.numChannels, -1])
            data = np.transpose(data, [1, 0, 2])
            data = np.reshape(data, [self.numChannels, -1])
            data = np.transpose(data, [1, 0])

            (numSamples, drop) = data.shape
            #Use all data if exampleSize < 0
            if(self.exampleSize < 0):
                break
            sampleAgain = True

        #Grab a chunk from exampleSize
        if(self.exampleSize < 0):
            outData = data.astype(np.float32)
        else:
            if(self.doShuffle):
                beg_idx = np.random.randint(0, numSamples - self.exampleSize)
            else:
                beg_idx = 0
            outData = data[beg_idx:beg_idx+self.exampleSize, :].astype(np.float32)

        #Normalize data
        #outData = data[beg_idx:beg_idx + self.exampleSize, :]

        if(self.scaleByChannel):
            std = outData.std(axis=0, keepdims=True)
        else:
            std = outData.std()
        outData = outData.astype(np.float32)/std

        #outData = outData * .04

        return outData

    def getData(self, batchSize):
        if(batchSize > 1):
            assert(self.exampleSize > 0)

        outData = np.zeros((batchSize, self.inputShape[0], self.inputShape[1], self.inputShape[2]))
        for b in range(batchSize):
            outData[b, 0, :, :] = self.getExample()
        return outData

if __name__=="__main__":
    #List of filenames
    filename = "/home/slundquist/mountData/datasets/seismic/wf.txt"
    #Settings file
    settingsFilename = "/home/slundquist/mountData/datasets/seismic/p4681_run1_AE.mat"
    #Output directory
    outDir = "/home/slundquist/mountData/datasets/seismicpvp/"

    if not os.path.exists(outDir):
        os.makedirs(outDir)
    #How many timesteps to store as one exmple
    #-1 means all data
    exampleSize = -1
    obj = seismicData(filename, settingsFilename, exampleSize, shuffle=False)
    for f in range(obj.numFiles):
        data = obj.getExample()
        data = data[np.newaxis, np.newaxis, :, :]
        outFilename = obj.current_filename[:-3] + ".pvp"
        #parse out filename
        fnSuffix = outFilename.split("/")[-1]
        pvpData = {"values": data, "time": np.array([0])}
        #writepvpfile(outDir+fnSuffix, pvpData)
