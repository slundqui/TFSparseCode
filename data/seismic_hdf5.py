import h5py
import numpy as np
import pdb
import random
import TFSparseCode.data.utils as utils

class SeismicDataHdf5(object):
    def __init__(self, filename, example_size=None, seed=None, normalize=True):
        if seed is not None:
            random.seed(seed)

        self.example_size = example_size
        print("Loading Data")
        self.data = self.loadStream(filename)
        print("Done")
        (self.num_samples, self.num_channels, self.num_stations, self.num_events) = self.data.shape
        self.data = np.transpose(self.data, [3, 0, 2, 1])
        self.num_features = self.num_channels * self.num_stations
        #Data is now [num_events, num_samples, num_stations * num_channels]
        self.data = np.reshape(self.data, [self.num_events, self.num_samples, self.num_features])

        #Normalize by entire dataset, ind by feature
        if(normalize):
            mean = np.mean(self.data, axis=(0, 1), keepdims = True)
            std = np.std(self.data, axis=(0, 1), keepdims = True)
            self.data = (self.data - mean)/std

        self.station_group = np.reshape(np.array(range(self.num_features)), [self.num_stations, self.num_channels])
        self.station_title = ["station_" + str(g[0]) for g in self.station_group]

        self.inputShape = [self.example_size, self.num_features]

        assert(self.example_size > 0 and self.example_size <= self.num_samples)

    def loadStream(self, filename, verbose=True):
        self.file = h5py.File(filename, 'r')
        #Using only first 5 for now, TODO param
        data = []
        max_idx = 5
        for idx in range(max_idx):
            if(verbose):
                print(str(idx + 1) + " out of " + str(max_idx))
            data.append(np.array(self.file[str(idx)]))
        data = np.concatenate(data, axis=3)

        #Find nans, which means that station doesn't have data
        has_data = np.isfinite(data)
        pos_station = np.all(has_data, axis=(0, 1, 3))
        pos_station_idx = np.nonzero(pos_station)

        #Remove these stations
        parse_data = data[:, :, pos_station_idx[0], :]
        return parse_data

    def getExample(self):
        #Randomly select an event
        event_idx = random.randint(0, self.num_events-1)
        sample_idx = random.randint(0, self.num_samples - self.example_size)

        outData = self.data[event_idx, sample_idx:sample_idx+self.example_size, :]
        return outData

    def getData(self, batchSize, dataset = "all"):
        outData = np.zeros([batchSize,] + self.inputShape)
        for b in range(batchSize):
            data = self.getExample()
            outData[b, :, :] = data

        outDict = {}
        outDict["data"] = outData
        return outDict

if __name__=="__main__":
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from plots.plotRecon import plotRecon1D

    seed=12345678

    #parameters
    filename = "/mnt/drive1/DataFile.hdf5"
    example_size = 12500

    dataObj = SeismicDataHdf5(filename, example_size, seed=seed)

    outDict = dataObj.getData(2)
    data = outDict['data']
    pdb.set_trace()
    prefix='/home/slundquist/mountData/tfSparseCode/test/test_plot'
    plotRecon1D(data, data, prefix, groups=dataObj.station_group)

