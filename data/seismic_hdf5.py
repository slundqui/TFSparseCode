import h5py
import numpy as np
import pdb
import random
import TFSparseCode.data.utils as utils
import os
from scipy.signal import butter, lfilter, freqz
import pickle

class SeismicDataHdf5(object):
    def butter_lowpass(self, cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    def butter_lowpass_filter(self, data, cutoff, fs, axis = -1, order=5 ):
        b, a = self.butter_lowpass(cutoff, fs, order=order)
        y = lfilter(b, a, data, axis=axis)
        return y

    def __init__(self, filename, example_size=None, seed=None, normalize=True, downsample_stride = 1, channel_idx=None, station_idx=None, arrival_filename=None, data_save_fn=None, loc_filter=True, sample_for_class=False, prediction=False):
        if seed is not None:
            random.seed(seed)

        self.example_size = example_size
        self.sampling_rate = 50 #hz
        self.downsample_stride = downsample_stride
        self.cutoff = self.sampling_rate/self.downsample_stride
        self.loc_filter = loc_filter
        self.sample_for_class = sample_for_class
        self.prediction = prediction

        print("Loading Data")
        load_npy_data = False
        if(data_save_fn is not None):
          if(os.path.isfile(data_save_fn)):
            load_npy_data = True
          else:
            load_npy_data = False
        else:
          load_npy_data = False

        #TODO need to load data w/ gt
        if(load_npy_data):
          out_array = np.load(data_save_fn)
          out_dict = {}
          out_dict["data"] = out_array[0]
          out_dict["gt"] = out_array[1]
        else:
          out_dict = self.loadStream(filename, arrival_filename=arrival_filename)
          if(data_save_fn is not None):
            out_array = np.array([out_dict["data"], out_dict["gt"]])
            np.save(data_save_fn, out_array)
        self.data = out_dict["data"]
        self.gt = out_dict["gt"]

        print("Done")

        #Save data if it exists, otherwise load it

        if(channel_idx is not None):
          self.data = self.data[:, np.array(channel_idx), :, :]
        if(station_idx is not None):
          self.data = self.data[:, :, np.array(station_idx), :]

        (self.num_samples, self.num_channels, self.num_stations, self.num_events) = self.data.shape
        print("Final data shape:")
        print(self.data.shape)

        self.data = np.transpose(self.data, [3, 0, 2, 1])
        self.num_features = self.num_channels * self.num_stations
        #Data is now [num_events, num_samples, num_stations * num_channels]
        self.data = np.reshape(self.data, [self.num_events, self.num_samples, self.num_features])

        #Split into train/test
        #This is okay since we shuffle idxs when reading
        num_test = int(self.num_events * .1)
        self.train_data = self.data[:-num_test]
        self.train_gt = self.gt[:-num_test]
        self.num_train_events = self.train_data.shape[0]
        self.test_data = self.data[-num_test:]
        self.test_gt = self.gt[-num_test:]
        self.num_test_events = self.test_data.shape[0]

        #Normalize by entire dataset, ind by feature
        if(normalize):
            mean = np.mean(self.data, axis=(0, 1), keepdims = True)
            std = np.std(self.data, axis=(0, 1), keepdims = True)
            self.data = (self.data - mean)/std

        self.station_group = np.reshape(np.array(range(self.num_features)), [self.num_stations, self.num_channels])
        self.station_title = ["station_" + str(g[0]) for g in self.station_group]

        self.inputShape = [self.example_size, self.num_features]

        assert(self.example_size > 0 and self.example_size <= self.num_samples)

    def loadStream(self, filename, verbose=True, arrival_filename=None):
        #filename is a list of filenames
        with open(filename, 'r') as f:
            list_of_filenames = f.readlines()
        list_of_filenames = [fn[:-1] for fn in list_of_filenames]

        all_data = []
        all_gt = []

        #Hard coded
        if(not self.loc_filter):
          list_of_filenames = list_of_filenames[:30]
        else:
          list_of_filenames = list_of_filenames[:40]

        for fn in list_of_filenames:
          if(verbose):
            print(fn)
          #TODO filename can be list of filenames
          self.file = h5py.File(fn, 'r')

          #Data is in num_samples, num_channels, num_stations, num_events
          full_data = self.file['data']
          srcs = self.file['srcs']

          if(self.loc_filter):
            #Find srcs within area of interest
            lat_filter = np.logical_and(srcs[:, 0] > -26, srcs[:, 1] < -24)
            lon_filter = np.logical_and(srcs[:, 1] > -71, srcs[:, 1] < -69)
            depth_filter = np.logical_and(srcs[:, 2] > -100000, srcs[:, 2] < -25000)
            loc_filter = np.logical_and(np.logical_and(lat_filter, lon_filter), depth_filter)

            #mag_filter = srcs[:, 5] > 2
            #total_filter = np.logical_and(mag_filter, loc_filter)
            #filter_idx = np.nonzero(total_filter)
            filter_idx = np.nonzero(loc_filter)

            if(self.prediction):
              data = full_data[40000:50000, :, :, filter_idx[0]]
            else:
              data = full_data[45000:55000, :, :, filter_idx[0]]
          else:
            if(self.prediction):
              data = full_data[40000:50000]
            else:
              data = full_data[45000:55000]

          num_data = data.shape[-1]
          num_full_data = full_data.shape[-1]
          gt = np.ones((num_data))

          #Sample noise
          if(self.sample_for_class):
            if(self.loc_filter):
              sample_idxs = filter_idx[0]
            else:
              sample_idxs = np.random.choice(num_full_data, num_data, replace=False)

            #Randomly select a time period, half before half after event
            #Sort because hdf5 files requires it to be in increasing order
            noise_data_pre = full_data[10000:20000, :, :, np.sort(sample_idxs[:int(num_data/2)])]
            noise_data_post = full_data[80000:90000, :, :, np.sort(sample_idxs[int(num_data/2):])]
            noise_data = np.concatenate((noise_data_pre, noise_data_post), axis=3)
            data = np.concatenate((data, noise_data), axis=3)
            gt = np.concatenate((gt, np.zeros((num_data))))

          #Downsample and filter data
          if(self.downsample_stride > 1):
            data = self.butter_lowpass_filter(data, self.cutoff, self.sampling_rate, axis=0)
            data = data[::self.downsample_stride, ...]

          all_data.append(data)
          all_gt.append(gt)

        data = np.concatenate(all_data, axis=3)
        gt = np.concatenate(all_gt)
        #Find nans, which means that station doesn't have data
        has_data = np.isfinite(data)
        pos_station = np.all(has_data, axis=(0, 1, 3))
        #pos_station_idx = np.nonzero(pos_station)
        #Hard coded
        #TODO explicitly set pos_station_idx
        #pos_station_idx = np.nonzero(pos_station)
        pos_station_idx = (np.array([1, 2, 3, 4, 5, 7, 11, 13, 16]),)

        #Remove stations which has nans
        parse_data = data[:, :, pos_station_idx[0], :]

        num_examples = parse_data.shape[-1]

        #Shuffle example idxs
        shuffle_idx = np.arange(num_examples)
        #This does shuffle in place
        np.random.shuffle(shuffle_idx)

        out = {}
        out["data"] = parse_data[:, :, :, shuffle_idx]
        out["gt"] = gt[shuffle_idx]

        return out

    def getExample(self, dataset):

        if(dataset == "all"):
          data = self.data
          gt = self.gt
          num_events = self.num_events
        elif(dataset == "train"):
          data = self.train_data
          gt = self.train_gt
          num_events = self.num_train_events
        elif(dataset == "test"):
          data = self.test_data
          gt = self.test_gt
          num_events = self.num_test_events

        #Randomly select an event
        event_idx = random.randint(0, num_events-1)
        sample_idx = random.randint(0, self.num_samples - self.example_size)


        outData = data[event_idx, sample_idx:sample_idx+self.example_size, :]
        outGt = gt[event_idx]
        return outData, outGt

    def getData(self, batchSize, dataset = "all"):
        outData = np.zeros([batchSize,] + self.inputShape)
        outGt = np.zeros([batchSize])
        for b in range(batchSize):
            data, gt = self.getExample(dataset)
            outData[b, :, :] = data
            outGt[b] = gt

        outDict = {}
        outDict["data"] = outData
        outDict["gt"] = outGt
        return outDict

if __name__=="__main__":
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from plots.plotRecon import plotRecon1D

    seed=12345678

    #parameters
    #filename = "/mnt/drive1/small_seismic_hdf5/DataFile.hdf5"
    #arrival_filename = "/mnt/drive1/small_seismic_hdf5/ArrivalFile.hdf5"
    filename = "/mnt/drive1/large_seismic_hdf5/data.txt"
    example_size = 10000

    dataObj = SeismicDataHdf5(filename, example_size, seed=seed)

    outDict = dataObj.getData(1000)
    data = outDict['data']

    [batch, time, features] = data.shape

    #Move features to batch
    data = np.transpose(data, [0, 2, 1])
    data = np.reshape(data, [batch * features, time])
    f = np.fft.fft(data, axis=-1)
    #Don't need phase information
    freq = np.fft.fftfreq(time)
    pdb.set_trace()
    plt.plot(freq, np.mean(f.real, axis=0))

    plt.savefig('freq_data.png')




    pdb.set_trace()



    #prefix='/home/slundquist/mountData/tfSparseCode/test/test_plot'
    #plotRecon1D(data, data, prefix, groups=dataObj.station_group)

