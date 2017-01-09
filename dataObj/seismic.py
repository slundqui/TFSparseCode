import random
import pdb
import numpy as np

def readFile(filename):
    fn = open(filename, 'r')
    content = fn.readlines()
    fn.close()
    return [f[:-1] for f in content] #Remove new lines

class seismicData(object):
    #filename is a list of filenames that contain seismic data
    def __init__(self, filename, exampleSize, seed=None):
        if seed is not None:
            random.seed(seed)
        #TODO fix to allow for multiple streams
        self.exampleSize = exampleSize
        self.inputShape = [1, exampleSize, 1]
        #Read list of filenames and store into member variable
        self.fnList = readFile(filename)

    def getExample(self):
        #Get random file from fnList
        #If length of data in file is less than example size, skip
        numData = 0
        while numData < self.exampleSize:
            filename = random.choice(self.fnList)
            data = readFile(filename)
            numData = len(data)
        #Generate starting point of data
        #Must contain a full example
        startIdx = random.randint(0, numData-self.exampleSize)
        dataVals = [float(s.split(',')[1]) for s in data[startIdx:startIdx+self.exampleSize]]
        dataVals = np.array(dataVals)
        #Make sure there are no 0s in the seismic data (for log scale)
        dataVals[np.nonzero(np.logical_and(dataVals>-1, dataVals<1))] = 1
        #Set to log scale
        logDataVals = np.log(np.abs(dataVals)) * np.sign(dataVals)
        return logDataVals

    def getData(self, batchSize):
        outVals = np.zeros((batchSize, self.inputShape[0], self.inputShape[1], self.inputShape[2]))
        for b in range(batchSize):
            outVals[b, 0, :, 0] = self.getExample()
        return outVals

if __name__=="__main__":
    listOfFn = "/home/sheng/mountData/seismic/seismic.txt"
    #How many timesteps to store as one exmple
    exampleSize = 100
    obj = seismicData(listOfFn, exampleSize, "asdf")
    e = obj.getData(10)
    pdb.set_trace()
