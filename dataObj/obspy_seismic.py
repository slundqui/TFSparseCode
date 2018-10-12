import obspy as ob
import numpy as np
import pdb
import random
import TFSparseCode.dataObj.utils as utils
import datetime
from matplotlib.dates import datestr2num
import csv

class obspySeismicData(object):
    def __init__(self, filename, example_size, target_rate=40, seed=None, time_range=None, event_csv=None, get_type=None, event_window = 1800, station_csv=None):
        if seed is not None:
            random.seed(seed)
        self.get_type = get_type
        self.event_window = event_window
        self.example_size = example_size
        assert(self.example_size > 0)
        self.target_rate = target_rate
        self.time_range = time_range
        (self.stream, self.trace_dict, self.start_time, self.end_time) = self.loadStream(filename)
        print(self.trace_dict)
        self.num_channels = len(self.trace_dict.keys())
        self.time_window = self.example_size / self.target_rate
        self.delta_time = self.end_time - self.start_time
        self.inputShape = [1, self.example_size, self.num_channels]

        if(event_csv is not None):
            with open(event_csv, 'r') as file:
                csv_txt = file.readlines()
            self.event_times = [v.split(',')[0] for v in csv_txt]
            #Remove first entry
            self.event_times.pop(0)
            self.event_times = [ob.UTCDateTime(v) for v in self.event_times]
            #Only select times in time range
            self.event_times = [v for v in self.event_times if v >= self.start_time
                    and v <= self.end_time]
            print("Number of events:", len(self.event_times))
            if(len(self.event_times) == 0):
                print("Error: no events between ", self.start_time, "and", self.end_time)
                pdb.set_trace()
        else:
            self.event_times = None

        if(station_csv is not None):
            with open(station_csv, 'r') as file:
                csv_txt = file.readlines()
            #Remove first entry
            csv_txt.pop(0)

            #Get station info from traces
            trace_stations = []
            station_idxs = []
            for k in self.trace_dict.keys():
                station_name = k.split('.')[1]
                trace_stations.append(station_name)
                station_idxs.append(self.trace_dict[k])

            self.station_info = []
            self.station_title = []
            self.station_group = []
            for line in csv_txt:
                split_line = line.split(',')
                station_key = split_line[1]
                name = split_line[2]
                lat = split_line[3]
                lon = split_line[4]

                idx_list = []
                for i, trace_s in enumerate(trace_stations):
                    if station_key == trace_s:
                        idx_list.append(station_idxs[i])
                if(len(idx_list) != 0):
                    self.station_title.append(station_key)
                    self.station_info.append([name, lat, lon])
                    self.station_group.append(idx_list)
        else:
            self.station_info = None

    def loadStream(self, filename):
        fns = utils.readList(filename)
        st = ob.Stream()
        print("Reading files")
        #Read into stream
        for i in range(len(fns)):
            current_st = ob.read(fns[i])
            print(fns[i], np.round(100*i/len(fns)), "%", end='\r', flush=True)
            for trace in current_st:
                if(trace.stats['sampling_rate'] - self.target_rate < 1e-4):
                    st.append(trace)

        if self.time_range is not None:
            start_time = ob.UTCDateTime(self.time_range[0])
            end_time = ob.UTCDateTime(self.time_range[1])
            st = st.slice(start_time, end_time, nearest_sample=False)
        else:
            start_time = None
            end_time = None

        print("Done")
        print("Processing stream")
        #Load each stream into a dictionary to determine features for data
        #Stores channel index in each trace dict
        trace_dict = {}
        #stores list of stds for each trace in dictionary
        #max_dict = {}
        #std_dict = {}

        count = 0

        #Store max amplitude for each trace in stream
        self.trace_stats= {}

        #Iterate through traces in stream
        for (i, trace) in enumerate(st):
            #print(100*i/len(st), "%", end='\r', flush=True)
            id_val = trace.get_id()

            #trace_max = trace.max()
            #trace_std = trace.std()
            #trace_n = trace.stats['npts']

            stats_tuple = (trace.stats['starttime'], trace.stats['endtime'], trace.max())
            if(id_val not in trace_dict):
                trace_dict[id_val] = count
                self.trace_stats[id_val] = [stats_tuple,]
                count += 1
            else:
                self.trace_stats[id_val].append(stats_tuple)

            #    std_dict[id_val] = [[trace_n, trace_std]]
            #    max_dict[id_val] = trace_max
            #else:
            #    std_dict[id_val].append([trace_n, trace_std])
            #    if(trace_max > max_dict[id_val]):
            #        max_dict[id_val] = trace_max

            #Update start and end time
            trace_st = trace.stats['starttime']
            trace_et = trace.stats['endtime']
            if(start_time is None or trace_st < start_time):
                start_time = trace_st
            if(end_time is None or trace_et > end_time):
                end_time = trace_et


        #Go through traces
        print("Done")

        return(st, trace_dict, start_time, end_time)

    def getExample(self):
        outData = np.zeros([self.example_size, self.num_channels])
        #Mask where 1 is invalid
        outMask = np.ones([self.example_size, self.num_channels])

        #Define random offset
        if(self.get_type == "event"):
            #Pick random event time
            event_t = random.choice(self.event_times)
            half_event_window = self.event_window/2
            offset = random.uniform(0, self.event_window - self.time_window)
            sample_st = event_t - half_event_window + offset
            sample_et = sample_st + self.time_window
        elif(self.get_type == 'no_event'):
            half_event_window = self.event_window/2
            in_event = True
            while in_event:
                offset = random.uniform(0, self.delta_time - self.time_window)
                sample_st = self.start_time + offset
                sample_et = sample_st + self.time_window
                in_event = False
                for e_time in self.event_times:
                    e_start_time = e_time - half_event_window
                    e_end_time = e_time + half_event_window
                    latest_start = max(e_start_time, sample_st)
                    earliest_end = min(e_end_time, sample_et)
                    delta = (earliest_end - latest_start)
                    #If overlap
                    if(delta > 0):
                        in_event = True
                        break

        elif(self.get_type is None):
            offset = random.uniform(0, self.delta_time - self.time_window)
            sample_st = self.start_time + offset
            sample_et = sample_st + self.time_window
        else:
            print("Error: get_type", self.get_type, "not recognized")

        #Get data from all traces between start_time and end_time
        slice_st = self.stream.slice(sample_st, sample_et, nearest_sample=False)
        #Split masked traces into contiguous unmasked arrays
        #slice_st = slice_st.split()

        for trace in slice_st:
            trace_id = trace.get_id()
            channelIdx = self.trace_dict[trace_id]
            #Get times for trace and convert to indices
            float_sampleIdx = (self.target_rate * (trace.times(type='utcdatetime') - sample_st))
            sampleIdx = float_sampleIdx.astype(np.int32)
            #Make sure in bounds
            valid_indices = np.nonzero(np.logical_and(sampleIdx >= 0, sampleIdx < self.example_size))
            sampleIdx = sampleIdx[valid_indices]
            trace_data = trace.data[valid_indices]
            #Store values and mark mask
            outData[sampleIdx, channelIdx] = trace_data
            outMask[sampleIdx, channelIdx] = 0

        ##Normalize by mean/std
        ##Find mean/std by hand in case of missing data
        #n = np.sum(1-outMask, axis=0)
        ##Avoid divide by 0
        #n[np.nonzero(n == 0)] = 1
        #mean = np.sum(outData, axis=0)/n
        #std = np.sqrt(np.sum(np.square(outData - mean), axis=0)/n)
        ##Avoid divide by 0
        #std[np.nonzero(std == 0)] = 1
        #outData = (outData - mean) / std
        return(outData, outMask)

    def getData(self, batchSize):
        outData = np.zeros((batchSize, self.inputShape[0], self.inputShape[1], self.inputShape[2]))
        #TODO
        outMask = np.zeros((batchSize, self.inputShape[0], self.inputShape[1], self.inputShape[2]))
        for b in range(batchSize):
            (data, mask) = self.getExample()
            while(np.sum(mask) == mask.size):
                (data, mask) = self.getExample()
            outData[b, 0, :, :] = data
            outMask[b, 0, :, :] = mask
        return(outData, outMask)
        #return outData

if __name__=="__main__":
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from plots.plotRecon import plotRecon1d

    random.seed(12345678)

    #parameters
    filename = "/home/slundquist/mountData/datasets/CanadianData_jun.txt"
    event_filename = "/home/slundquist/mountData/datasets/query_2016.csv"
    station_csv = "/home/slundquist/mountData/datasets/station_info.csv"
    example_size = 100000

    #start_time = "2016-02-24T20:00:00"
    #end_time = "2016-02-25T02:00:00"
    #dataObj = obspySeismicData(filename, example_size, time_range=[start_time, end_time])
    dataObj = obspySeismicData(filename, example_size, event_csv=event_filename, get_type="event", station_csv=station_csv)
    print(dataObj.station_info)
    print(dataObj.station_group)

    (data, mask) = dataObj.getData(10)
    prefix='/home/slundquist/mountData/tfSparseCode/test/test_plot'
    plotRecon1d(mask[:, 0, :, :], data[:, 0, :, :], prefix+"1", groups=dataObj.station_group, group_title=dataObj.station_title)

