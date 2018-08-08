import numpy as np
import tensorflow as tf
import pdb
from scipy import sparse
#from pvtools import readpvpfile

def convertToSparse5d(m):
    [nb, nt, ny, nx, nf] = m.shape
    mreshape = np.reshape(m, (nb, nt*ny*nx*nf))
    return sparse.csr_matrix(mreshape)

def convertToSparse4d(m):
    [nb, ny, nx, nf] = m.shape
    mreshape = np.reshape(m, (nb, ny*nx*nf))
    return sparse.csr_matrix(mreshape)

def save_sparse_csr(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )

#def load_sparse_csr(filename):
#    loader = np.load(filename)
#    return csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
#                         shape = loader['shape'])

#def load_pvp_weights(filename):
#    data = readpvpfile(filename)
#    vals = data["values"]
#    outVals = vals[0, 0, :, :, :, :].transpose((1, 2, 3, 0)).astype(np.float32)
#    return outVals

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

def uniform_weight_variable(shape, inName, minVal=0, maxVal=None):
    initial = tf.random_uniform_initializer(minVal, maxVal)
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

def conv2d_oneToMany(x, W, outShape, inName, yStride, xStride, padding='SAME'):
    return tf.nn.conv2d_transpose(x, W, outShape, [1, yStride, xStride, 1], padding=padding, name=inName)

def maxpool_2x2(x, inName):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1], padding='SAME', name=inName)

def conv3d(x, w, inName):
    return tf.nn.conv3d(x, w, strides=[1, 1, 1, 1, 1], padding='SAME', name=inName)

#Transposes data to permute strides to the output feature dimension
def transpose5dData(x, xShape, strideT, strideY, strideX):
    [nb, nt, ny, nx, nf] = xShape
    print("Building output indices for conv3d")
    #Build gather indices for output
    #Must be in shape of target output data
    dataIdxs = np.zeros((nb, nt/strideT, ny/strideY, nx/strideX, nf*strideT*strideY*strideX, 5)).astype(np.int32)
    for iib in range(nb):
        for iit in range(nt):
            for iiy in range(ny):
                for iix in range(nx):
                    for iif in range(nf):
                        #Calculate input indices given output indices
                        oob = iib
                        oot = iit/strideT
                        ooy = iiy/strideY
                        oox = iix/strideX
                        kernelIdx = (iit%strideT)*strideY*strideX + (iiy%strideY)*strideX + (iix%strideX)
                        oof = iif + nf*kernelIdx
                        dataIdxs[oob, oot, ooy, oox, oof, :] = [iib, iit, iiy, iix, iif]
    return tf.gather_nd(x, dataIdxs)

#Undo transepost5dData
def undoTranspose5dData(x, xShape, strideT, strideY, strideX):
    #These shapes are in terms of the orig image
    [nb, nt, ny, nx, nf] = xShape
    print("Building output indices for conv3d")
    #Build gather indices for output
    #Must be in shape of target output data
    dataIdxs = np.zeros((nb, nt, ny, nx, nf, 5)).astype(np.int32)
    for oob in range(nb):
        for oot in range(nt):
            for ooy in range(ny):
                for oox in range(nx):
                    for oof in range(nf):
                        #Calculate input indices given output indices
                        iib = oob
                        iit = oot/strideT
                        iiy = ooy/strideY
                        iix = oox/strideX
                        kernelIdx = (oot%strideT)*strideY*strideX + (ooy%strideY)*strideX + (oox%strideX)
                        iif = oof + nf*kernelIdx
                        dataIdxs[oob, oot, ooy, oox, oof, :] = [iib, iit, iiy, iix, iif]
    return tf.gather_nd(x, dataIdxs)

#Transposes weight data for viewing
def transpose5dWeight(w, wShape, strideT, strideY, strideX):
    print("Building weight indices for conv3d")
    #These shapes are in terms of the already strided values
    [ntp, nyp, nxp, nifp, nofp] = wShape
    #Translate to target output shape
    ntp *= strideT
    nyp *= strideY
    nxp *= strideX
    nofp = nofp/(strideT*strideX*strideY)

    #Build gather indices for weights
    #Must be in shape of target output weights
    weightIdxs = np.zeros((ntp, nyp, nxp, nifp, nofp, 5)).astype(np.int32)
    #Adding kernel number to end of features
    for otp in range(ntp):
        for oyp in range(nyp):
            for oxp in range(nxp):
                for oifp in range(nifp):
                    for oofp in range(nofp):
                        #Calculate output indices given input indices
                        #Must reverse, as we're using conv2d as transpose conv2d
                        #otp = int((ntp-itp-1)/strideT)
                        #oyp = int((nyp-iyp-1)/strideY)
                        #oxp = int((nxp-ixp-1)/strideX)
                        #oifp = iifp #Input features stay the same
                        itp = int((ntp - otp-1)/strideT)
                        iyp = int((nyp - oyp-1)/strideY)
                        ixp = int((nxp - oxp-1)/strideX)
                        iifp=oifp
                        #oofp uses iofp as offset, plus an nf stride based on which kernel it belongs to
                        kernelIdx = (otp%strideT)*strideY*strideX + (oyp%strideY)*strideX + (oxp%strideX)
                        iofp = oofp + nofp * kernelIdx
                        weightIdxs[otp, oyp, oxp, oifp, oofp, :] = [itp, iyp, ixp, iifp, iofp]
    return tf.gather_nd(w, weightIdxs)

def conv3d_oneToMany(x, xShape, w, wShape, strideT, strideY, strideX, inName):
    [ntp, nyp, nxp, nifp, nofp] = wShape
    [nb, nt, ny, nx, nf] = xShape

    #stride must be divisible by both weights and input
    assert(ntp%strideT == 0)
    assert(nyp%strideY == 0)
    assert(nxp%strideX == 0)
    assert(nt%strideT == 0)
    assert(ny%strideY == 0)
    assert(nx%strideX == 0)

    assert(nifp == nf)

    print("Building weight indices for conv3d")
    #Build gather indices for weights
    #Must be in shape of target output weights
    weightIdxs = np.zeros((int(ntp/strideT), int(nyp/strideY), int(nxp/strideX), nifp, nofp*strideT*strideX*strideY, 5)).astype(np.int32)
    #Adding kernel number to end of features
    for itp in range(ntp):
        for iyp in range(nyp):
            for ixp in range(nxp):
                for iifp in range(nifp):
                    for iofp in range(nofp):
                        #Calculate output indices given input indices
                        #Must reverse, as we're using conv2d as transpose conv2d
                        otp = int((ntp-itp-1)/strideT)
                        oyp = int((nyp-iyp-1)/strideY)
                        oxp = int((nxp-ixp-1)/strideX)
                        oifp = iifp #Input features stay the same
                        #oofp uses iofp as offset, plus an nf stride based on which kernel it belongs to
                        kernelIdx = (itp%strideT)*strideY*strideX + (iyp%strideY)*strideX + (ixp%strideX)
                        oofp = iofp + nofp * kernelIdx
                        weightIdxs[otp, oyp, oxp, oifp, oofp, :] = [itp, iyp, ixp, iifp, iofp]


    print("Building output indices for conv3d")
    #Build gather indices for output
    #Must be in shape of target output data
    dataIdxs = np.zeros((nb, nt*strideT, ny*strideY, nx*strideX, nofp, 5)).astype(np.int32)
    for oob in range(nb):
        for oot in range(nt*strideT):
            for ooy in range(ny*strideY):
                for oox in range(nx*strideX):
                    for oof in range(nofp):
                        #Calculate input indices given output indices
                        iib = oob
                        iit = oot/strideT
                        iiy = ooy/strideY
                        iix = oox/strideX
                        kernelIdx = (oot%strideT)*strideY*strideX + (ooy%strideY)*strideX + (oox%strideX)
                        iif = oof + nofp*kernelIdx
                        dataIdxs[oob, oot, ooy, oox, oof, :] = [iib, iit, iiy, iix, iif]

    #Build convolution structure
    w_reshape = tf.gather_nd(w, weightIdxs)
    o_reshape = tf.nn.conv3d(x, w_reshape, strides=[1, 1, 1, 1, 1], padding='SAME', name=inName)
    o = tf.gather_nd(o_reshape, dataIdxs)
    return o

if __name__ == "__main__":
    #For conv2d
    weightShapeOrig = (6, 6, 6, 1, 1)
    stride = 2
    inputShape = (1, 8, 8, 8, 1)

    npWeightArray = np.zeros(weightShapeOrig).astype(np.float32)
    for itp in range(6):
        for iyp in range(6):
            for ixp in range(6):
                idx = itp*36 + iyp*6 + ixp
                npWeightArray[itp, iyp, ixp, 0, 0] = idx

    npInputArray = np.zeros(inputShape).astype(np.float32)
    npInputArray[0, 3, 3, 3, 0] = 1

    #Tensorflow test
    sess=tf.InteractiveSession()
    W = tf.Variable(npWeightArray)
    I = tf.Variable(npInputArray)
    O = conv3d_oneToMany(I, inputShape, W, weightShapeOrig, 2, 2, 2, "test")
    sess.run(tf.initialize_all_variables())

    npI = I.eval()[0, :, :, :, 0]
    npW = W.eval()[:, :, :, 0, 0]
    npO = O.eval()[0, :, :, :, 0]

    pdb.set_trace()
