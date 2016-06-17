import pdb
import numpy as np
import tensorflow as tf
#import matplotlib.pyplot as plt

#Helper functions for initializing weights
def weight_variable_fromnp(inNp, inName):
    shape = inNp.shape
    return tf.Variable(inNp, name=inName)

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

def maxpool_2x2(x, inName):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1], padding='SAME', name=inName)

class LCA:
    #Initialize tf parameters here
    #Learning rate for optimizer
    #TODO make this parameter in train method
    learningRate = 1e-4
    thresh = .015
    numV = 128
    patchSize = 12
    #Progress interval
    progress = 1

    #Global timestep
    timestep = 0

    #Constructor takes inputShape, which is a 3 tuple (ny, nx, nf) based on the size of the image being fed in
    def __init__(self, dataObj, vggFile = None, batchSize = 1):
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        self.dataObj = dataObj
        self.batchSize = batchSize
        self.buildModel(self.dataObj.inputShape)
        (self.currImg, drop) = self.dataObj.getData(self.batchSize)

    #Builds the model. inMatFilename should be the vgg file
    def buildModel(self, inputShape):
        #Running on GPU
        with tf.device('gpu:0'):
            with tf.name_scope("inputOps"):
                #Get convolution variables as placeholders
                self.inputImage = node_variable([self.batchSize, inputShape[0], inputShape[1], inputShape[2]], "inputImage")
                #self.gt = node_variable([None, 2], "gt")
                #Model variables for convolutions

            with tf.name_scope("Dictionary"):
                self.V1_W = weight_variable((self.patchSize, self.patchSize, self.numV, 3), "V1_W", 1e-3)

            with tf.name_scope("weightNorm"):
                self.normVals = tf.reduce_sum(tf.square(self.V1_W), reduction_indices=[0, 1, 3], keep_dims=True)
                self.normalize_W = self.V1_W.assign(self.V1_W/self.normVals)

            with tf.name_scope("LCA"):
                #Soft threshold
                #self.V1_A= weight_variable((self.batchSize, inputShape[0]/2, inputShape[1]/2, self.numV), "V1_U", 1e-3)
                self.V1_A= weight_variable((self.batchSize, inputShape[0], inputShape[1], self.numV), "V1_U", .01)
                #self.LCA_U = conv2d(self.error, self.V1_W, "u", stride=[1, 2, 2, 1])
                #self.LCA_A = tf.nn.relu(self.LCA_U - self.thresh)

            with tf.name_scope("Recon"):
                #TODO this conv needs to be a "one to many" convolution
                #See OpenPV cudnn implementation
                self.recon = conv2d(self.V1_A, self.V1_W, "recon", [1, 1, 1, 1])

            with tf.name_scope("Error"):
                self.error = self.inputImage - self.recon

            with tf.name_scope("Loss"):
                self.reconError = tf.reduce_mean(tf.square(self.error))
                self.l1Sparsity = tf.reduce_mean(tf.abs(self.V1_A))
                #Define loss
                self.loss = self.reconError/2 + self.thresh * self.l1Sparsity

            with tf.name_scope("Opt"):
                #Define optimizer
                #self.optimizerAll = tf.train.AdagradOptimizer(self.learningRate).minimize(self.loss)
                #self.optimizerFC = tf.train.AdagradOptimizer(self.learningRate).minimize(self.loss,
                self.optimizerA = tf.train.GradientDescentOptimizer(self.learningRate).minimize(self.loss,
                        var_list=[
                            self.V1_A
                        ])
                self.optimizerW = tf.train.GradientDescentOptimizer(self.learningRate).minimize(self.loss,
                        var_list=[
                            self.V1_W
                        ])

        #Summaries
        self.s_loss = tf.scalar_summary('loss', self.loss, name="lossSum")
        self.s_recon = tf.scalar_summary('recon error', self.reconError, name="reconError")
        self.s_l1= tf.scalar_summary('l1 sparsity', self.l1Sparsity, name="l1Sparsity")
        self.s_l0 = tf.scalar_summary('l0 sparsity', tf.nn.zero_fraction(self.V1_A), name="l0Sparsity")

        self.h_input = tf.histogram_summary('input', self.inputImage, name="input")
        self.h_recon = tf.histogram_summary('recon', self.recon, name="recon")
        self.h_v1_w = tf.histogram_summary('V1_W', self.V1_W, name="V1_W")
        self.h_v1_a = tf.histogram_summary('V1_A', self.V1_A, name="V1_A")

        #Images
        self.i_w = tf.image_summary("weights", tf.transpose(self.V1_W, [2, 0, 1, 3]), max_images=self.numV)
        self.i_orig = tf.image_summary("orig", self.inputImage)
        self.i_recon = tf.image_summary("recon", self.recon)

        #Define saver
        self.saver = tf.train.Saver()

    #Initializes session.
    def initSess(self):
        self.sess.run(tf.initialize_all_variables())

    #Allocates and specifies the output directory for tensorboard summaries
    def writeSummary(self, summaryDir):
        self.mergedSummary = tf.merge_summary([
            #'loss', 'recon error', 'l1Sparsity', 'l0Sparsity', 'input', 'recon', 'V1_W', 'V1_A'
            self.s_loss, self.s_recon, self.s_l1, self.s_l0, self.h_input, self.h_recon, self.h_v1_w, self.h_v1_a
            ])
        self.imageSummary = tf.merge_summary([
            self.i_w, self.i_orig, self.i_recon
            ])
        self.train_writer = tf.train.SummaryWriter(summaryDir + "/train", self.sess.graph)
        #self.test_writer = tf.train.SummaryWriter(summaryDir + "/test")

    def closeSess(self):
        self.sess.close()

    #Trains model for numSteps
    #If pre is False, will train entire network
    #If pre is True, will train only fully connected network
    def trainA(self, numSteps, saveFile):
        #Define session
        for i in range(numSteps):
            #feedDict = {self.inputImage: data[0], self.gt: data[1], self.keep_prob: .5}
            feedDict = {self.inputImage: self.currImg}
            #Run optimizer
            self.sess.run(self.optimizerA, feed_dict=feedDict)
            summary = self.sess.run(self.mergedSummary, feed_dict=feedDict)
            self.train_writer.add_summary(summary, self.timestep)
            self.timestep+=1
            if(i%self.progress == 0):
                print "Timestep ", self.timestep

        save_path = self.saver.save(self.sess, saveFile, global_step=self.timestep, write_meta_graph=False)
        print("Model saved in file: %s" % save_path)

    def normWeights(self):
        #Normalize weights
        self.sess.run(self.normalize_W)

    def trainW(self):
        feedDict = {self.inputImage: self.currImg}
        #Update weights
        self.sess.run(self.optimizerW, feed_dict=feedDict)
        #Write summary
        summary = self.sess.run(self.imageSummary, feed_dict=feedDict)
        self.train_writer.add_summary(summary, self.timestep)
        #New image
        (self.currImg, drop) = self.dataObj.getData(self.batchSize)

    ##Evaluates all of inData at once
    ##If an inGt is provided, will calculate summary as test set
    #def evalModel(self, inData, inGt = None):
    #    (numData, ny, nx, nf) = inData.shape
    #    if(inGt != None):
    #        (numGt, drop) = inGt.shape
    #        assert(numData == numGt)
    #        feedDict = {self.inputImage: inData, self.gt: inGt, self.keep_prob: 1}
    #    else:
    #        feedDict = {self.inputImage: inData, self.keep_prob: 1}

    #    outVals = self.est.eval(feed_dict=feedDict, session=self.sess)
    #    if(inGt != None):
    #        summary = self.sess.run(self.mergedSummary, feed_dict=feedDict)
    #        self.test_writer.add_summary(summary, self.timestep)
    #    return outVals

    ##Evaluates inData, but in miniBatchSize batches for memory efficiency
    ##If an inGt is provided, will calculate summary as test set
    #def evalModelBatch(self, miniBatchSize, inData, inGt=None):
    #    (numData, ny, nx, nf) = inData.shape
    #    if(inGt != None):
    #        (numGt, drop) = inGt.shape
    #        assert(numData == numGt)

    #    #Split up numData into miniBatchSize and evaluate est data
    #    tfInVals = np.zeros((miniBatchSize, ny, nx, nf))
    #    outData = np.zeros((numData, 1))

    #    #Ceil of numData/batchSize
    #    numIt = int(numData/miniBatchSize) + 1

    #    #Only write summary on first it

    #    startOffset = 0
    #    for it in range(numIt):
    #        print it, " out of ", numIt
    #        #Calculate indices
    #        startDataIdx = startOffset
    #        endDataIdx = startOffset + miniBatchSize
    #        startTfValIdx = 0
    #        endTfValIdx = miniBatchSize

    #        #If out of bounds
    #        if(endDataIdx >= numData):
    #            #Calculate offset
    #            offset = endDataIdx - numData
    #            #Set endDataIdx to max value
    #            endDataIdx = numData
    #            #Set endTfValIdx to less than max value
    #            endTfValIdx -= offset

    #        tfInVals[startTfValIdx:endTfValIdx, :, :, :] = inData[startDataIdx:endDataIdx, :, :, :]
    #        feedDict = {self.inputImage: tfInVals, self.keep_prob: 1}
    #        tfOutVals = self.est.eval(feed_dict=feedDict, session=self.sess)
    #        outData[startDataIdx:endDataIdx, :] = tfOutVals[startTfValIdx:endTfValIdx, :]

    #        if(inGt != None and it == 0):
    #            tfInGt = inGt[startDataIdx:endDataIdx, :]
    #            summary = self.sess.run(self.mergedSummary, feed_dict={self.inputImage: tfInVals, self.gt: tfInGt, self.keep_prob: 1})
    #            self.test_writer.add_summary(summary, self.timestep)

    #        startOffset += miniBatchSize

    #    #Return output data
    #    return outData

    #Loads a tf checkpoint
    def loadModel(self, loadFile):
        self.saver.restore(self.sess, loadFile)
        print("Model %s loaded" % loadFile)

