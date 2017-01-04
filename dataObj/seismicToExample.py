import tensorflow as tf
import numpy as np
import os
import pdb


listOfFn = "/home/sheng/mountData/seismic/seismic.txt";
name = "smallSeismic"
fn = open(listOfFn, 'r')
fnList = fn.readlines()
fn.close()
fnList = [f[:-1] for f in fnList]

#How many timesteps to store as one exmple
exampleSize = 100
#How much to stride a file
exampleStride = 4

tf.app.flags.DEFINE_string('directory', '/home/sheng/mountData/seismicData',
                           'Directory for results')

FLAGS = tf.app.flags.FLAGS

def _float32_feature(listValue):
    return tf.train.Feature(float_list=tf.train.FloatList(value=listValue))

#dataSum = 0.0
#count = 0
#
#for f in fnList:
#    data = np.genfromtxt(f, delimiter=',')
#    (numData, drop) = data.shape
#    dataSum += np.sum(data[:, 1])
#    count += numData
#
#dataMean = dataSum / count
#
#stdSum = 0.0
#
#for f in fnList:
#    data = np.genfromtxt(f, delimiter=',')
#    (numData, drop) = data.shape
#    stdSum += np.sum(np.power(data[:, 1] - dataMean, 2))
#
#stdMean = stdSum / count
#stdVal = np.sqrt(stdMean)

filename = os.path.join(FLAGS.directory, name + '.tfrecords')
writer = tf.python_io.TFRecordWriter(filename)

print 'Writing', filename
for f in fnList:
    data = np.genfromtxt(f, delimiter=',')
    (numData, drop) = data.shape
    offsetCount = 0
    while(offsetCount + exampleSize < numData):
        dataWindow = data[offsetCount:offsetCount+exampleSize, 1]
        dataWindow[np.nonzero(np.logical_and(dataWindow > -1, dataWindow < 1))] = 1
        logDataWindow = np.log(np.abs(dataWindow)) * np.sign(dataWindow)
        assert(len(np.nonzero(np.isnan(logDataWindow))[0]) == 0)

        example = tf.train.Example(features=tf.train.Features(feature={
            'data': _float32_feature(logDataWindow)
            }))
        writer.write(example.SerializeToString())
        offsetCount += exampleStride
writer.close()

pdb.set_trace()




