import tensorflow as tf
import pdb
import numpy as np


class seismicData(object):
    exampleSize = 100
    inputShape = [1, 100, 1]
    def read_and_decode_single_example(self, filename):
        # first construct a queue containing a list of filenames.
        # this lets a user split up there dataset in multiple files to keep
        # size down
        filename_queue = tf.train.string_input_producer([filename],
                                                        num_epochs=None)
        # Unlike the TFRecordWriter, the TFRecordReader is symbolic
        reader = tf.TFRecordReader()
        # One can read a single serialized example from a filename
        # serialized_example is a Tensor of type string.
        _, serialized_example = reader.read(filename_queue)
        # The serialized example is converted back to actual values.
        # One needs to describe the format of the objects to be returned
        features = tf.parse_single_example(
            serialized_example,
            features={
                # We know the length of both fields. If not the
                # tf.VarLenFeature could be used
                'data': tf.FixedLenFeature([self.exampleSize], tf.float32)
            })
        # now return the converted data
        data = features['data']
        return data

    def __init__(self, filename):
        # get single examples
        data = self.read_and_decode_single_example(filename)
        # groups examples into batches randomly
        self.data_batch = tf.train.shuffle_batch(
                    [data], batch_size=128,
                        capacity=2000,
                            min_after_dequeue=1000)
        self.sess = tf.Session()
        init = tf.initialize_all_variables()
        self.sess.run(init)
        tf.train.start_queue_runners(sess=self.sess)

    def getData(self, batchSize):
        assert(batchSize == 128)
        data = self.sess.run(self.data_batch)
        data = np.expand_dims(data, 1)
        data = np.expand_dims(data, 3)
        return data




