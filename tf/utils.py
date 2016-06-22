import numpy as np
import tensorflow as tf

#Helper functions for initializing weights
def weight_variable_fromnp(inNp, inName):
    shape = inNp.shape
    return tf.Variable(inNp, name=inName)

def sparse_weight_variable(shape, inName, sparsePercent = .9):
    inNp = np.random.uniform(-1.0, 1.0, shape).astype(np.float32)
    inNp[np.nonzero(np.abs(inNp) < sparsePercent)] = 0
    return tf.Variable(inNp, inName)

def weight_variable(shape, inName, inStd):
    initial = tf.truncated_normal_initializer(stddev=inStd)
    return tf.get_variable(inName, shape, initializer=initial)


def bias_variable(shape, inName, biasInitConst=.01):
   initial = tf.constant(biasInitConst, shape=shape, name=inName)
   return tf.Variable(initial)

def weight_variable_xavier(shape, inName, conv=False):
   #initial = tf.truncated_normal(shape, stddev=weightInitStd, name=inName)
   if conv:
       initial = tf.contrib.layers.xavier_initializer_conv2d()
   else:
       initial = tf.contrib.layers.xavier_initializer()
   return tf.get_variable(inName, shape, initializer=initial)

#Helper functions for creating input nodes
def node_variable(shape, inName):
   return tf.placeholder("float", shape=shape, name=inName)

#Helper functions for creating convolutions and pooling
def conv2d(x, W, inName, stride = None):
    if stride:
        return tf.nn.conv2d(x, W, strides=stride, padding='SAME', name=inName)
    else:
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', name=inName)

def conv2d_oneToMany(x, W, outShape, inName, tStride):
    return tf.nn.conv2d_transpose(x, W, outShape, [1, tStride, tStride, 1], padding='SAME', name=inName)

def maxpool_2x2(x, inName):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1], padding='SAME', name=inName)

def conv2d_oneToMany_test(x, xShape, w, wShape, strideX, strideY, inName):
    [nyp, nxp, nifp, nofp] = wShape
    [nb, ny, nx, nf] = xShape
    assert(inF == nf)
    targetShape = [nb, ny*strideY, nx*strideX, outF]

    #Build gather indices for weights
    #Adding kernel number to end of features
    for iyp in range(nyp):
        for ixp in range(nxp):
            for iifp in range(nifp):
                for iofp in range(nofp):
                    pass

if __name__ == "__main__":
    weightShape = (4, 4, 1, 1)
    stride = 2
    inputShape = (1, 8, 8, 1)

    outputShape = (1, inputShape[1]*stride, inputShape[2]*stride, 1)

    npWeightArray = np.zeros(weightShape)
    npWeightArray[0, 0, 0, 0] = 1
    npWeightArray[0, 1, 0, 0] = 2
    npWeightArray[0, 2, 0, 0] = 1
    npWeightArray[0, 3, 0, 0] = 2
    npWeightArray[1, 0, 0, 0] = 3
    npWeightArray[1, 1, 0, 0] = 4
    npWeightArray[1, 2, 0, 0] = 3
    npWeightArray[1, 3, 0, 0] = 4
    npWeightArray[2, 0, 0, 0] = 1
    npWeightArray[2, 1, 0, 0] = 2
    npWeightArray[2, 2, 0, 0] = 1
    npWeightArray[2, 3, 0, 0] = 2
    npWeightArray[3, 0, 0, 0] = 3
    npWeightArray[3, 1, 0, 0] = 4
    npWeightArray[3, 2, 0, 0] = 3
    npWeightArray[3, 3, 0, 0] = 4

    npInputArray = np.zeros(inputShape)
    npInputArray[0, 3, 3, 0] = 1

