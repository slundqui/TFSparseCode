#BROKEN DO NOT USE




import pdb
import numpy as np
import tensorflow as tf
from loadVgg import loadWeights
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

class dcnn:
    #Initialize tf parameters here
    #Learning rate for optimizer
    #TODO make this parameter in train method
    learningRate = 1e-4
    #Progress interval
    progress = 1

    #Global timestep
    timestep = 0

    #Constructor takes inputShape, which is a 3 tuple (ny, nx, nf) based on the size of the image being fed in
    def __init__(self, inputShape, vggFile = None):
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        self.buildModel(inputShape, vggFile)

    #Builds the model. inMatFilename should be the vgg file
    def buildModel(self, inputShape, inMatFilename):
        if(inMatFilename):
            npWeights = loadWeights(inMatFilename)

        #Running on GPU
        with tf.device('gpu:0'):
            with tf.name_scope("inputOps"):
                #Get convolution variables as placeholders
                self.inputImage = node_variable([None, inputShape[0], inputShape[1], inputShape[2]], "inputImage")
                self.gt = node_variable([None, 2], "gt")
                #Model variables for convolutions

            #We match ISTA params on CIFAR here
            with tf.name_scope("Conv1Ops"):
                if(inMatFilename):
                    self.W_conv1 = weight_variable_fromnp(npWeights["conv1_w"], "w_conv1")
                    self.B_conv1 = weight_variable_fromnp(npWeights["conv1_b"], "b_conv1")
                else:
                    self.W_conv1 = weight_variable_fromnp(np.zeros((11, 11, 3, 64), dtype=np.float32), "w_conv1")
                    self.B_conv1 = weight_variable_fromnp(np.zeros((64), dtype=np.float32), "b_conv1")
                    #self.W_conv1 = weight_variable_xavier([11, 11, 3, 64], "w_conv1", conv=True)
                    #self.B_conv1 = bias_variable([64], "b_conv1")
                self.h_conv1 = tf.nn.relu(conv2d(self.inputImage, self.W_conv1, "conv1", stride=[1, 4, 4, 1]) + self.B_conv1)
                self.h_norm1 = tf.nn.local_response_normalization(self.h_conv1, name="LRN1")
                self.h_pool1 = maxpool_2x2(self.h_norm1, "pool1")

            with tf.name_scope("Conv2Ops"):
                if(inMatFilename):
                    self.W_conv2 = weight_variable_fromnp(npWeights["conv2_w"], "w_conv2")
                    self.B_conv2 = weight_variable_fromnp(npWeights["conv2_b"], "b_conv2")
                else:
                    self.W_conv2 = weight_variable_fromnp(np.zeros((5, 5, 64, 256), dtype=np.float32), "w_conv2")
                    self.B_conv2 = weight_variable_fromnp(np.zeros((256), dtype=np.float32), "b_conv2")
                    #self.W_conv2 = weight_variable_xavier([5, 5, 64, 256], "w_conv2", conv=True)
                    #self.B_conv2 = bias_variable([256], "b_conv2")
                self.h_conv2 = tf.nn.relu(conv2d(self.h_pool1, self.W_conv2, "conv2") + self.B_conv2)
                self.h_norm2 = tf.nn.local_response_normalization(self.h_conv2, name="LRN2")
                self.h_pool2 = maxpool_2x2(self.h_norm2, "pool2")

            with tf.name_scope("Conv3Ops"):
                if(inMatFilename):
                    self.W_conv3 = weight_variable_fromnp(npWeights["conv3_w"], "w_conv3")
                    self.B_conv3 = weight_variable_fromnp(npWeights["conv3_b"], "b_conv3")
                else:
                    self.W_conv3 = weight_variable_fromnp(np.zeros((3, 3, 256, 256), dtype=np.float32), "w_conv3")
                    self.B_conv3 = weight_variable_fromnp(np.zeros((256), dtype=np.float32), "b_conv3")
                    #self.W_conv3 = weight_variable_xavier([3, 3, 256, 256], "w_conv3", conv=True)
                    #self.B_conv3 = bias_variable([256], "b_conv3")
                self.h_conv3 = tf.nn.relu(conv2d(self.h_pool2, self.W_conv3, "conv3") + self.B_conv3, name="relu3")

            with tf.name_scope("Conv4Ops"):
                if(inMatFilename):
                    self.W_conv4 = weight_variable_fromnp(npWeights["conv4_w"], "w_conv4")
                    self.B_conv4 = weight_variable_fromnp(npWeights["conv4_b"], "b_conv4")
                else:
                    self.W_conv4 = weight_variable_fromnp(np.zeros((3, 3, 256, 256), dtype=np.float32), "w_conv4")
                    self.B_conv4 = weight_variable_fromnp(np.zeros((256), dtype=np.float32), "b_conv4")
                    #self.W_conv4 = weight_variable_xavier([3, 3, 256, 256], "w_conv4", conv=True)
                    #self.B_conv4 = bias_variable([256], "b_conv4")
                self.h_conv4 = tf.nn.relu(conv2d(self.h_conv3, self.W_conv4, "conv4") + self.B_conv4, name="relu4")

            with tf.name_scope("Conv5Ops"):
                if(inMatFilename):
                    self.W_conv5 = weight_variable_fromnp(npWeights["conv5_w"], "w_conv5")
                    self.B_conv5 = weight_variable_fromnp(npWeights["conv5_b"], "b_conv5")
                else:
                    self.W_conv5 = weight_variable_fromnp(np.zeros((3, 3, 256, 256), dtype=np.float32), "w_conv5")
                    self.B_conv5 = weight_variable_fromnp(np.zeros((256), dtype = np.float32), "b_conv5")
                    #self.W_conv5 = weight_variable_xavier([3, 3, 256, 256], "w_conv5", conv=True)
                    #self.B_conv5 = bias_variable([256], "b_conv5")
                self.h_conv5 = tf.nn.relu(conv2d(self.h_conv4, self.W_conv5, "conv5") + self.B_conv5)
                self.h_pool5 = maxpool_2x2(self.h_conv5, "pool5")

            #placeholder for specifying dropout
            self.keep_prob = tf.placeholder(tf.float32)

            #32 comes from 4 stride in conv1, 2 stride in pool1, 2 stride in pool2, 2 stride in pool5
            numInputs = (inputShape[0]/32) * (inputShape[1]/32) * 256
            with tf.name_scope("FC1"):
                #if(inMatFilename):
                #    self.W_conv5 = weight_variable_fromnp(npWeights["fc1_w"], "w_fc1")
                #    self.B_conv5 = weight_variable_fromnp(npWeights["fc1_b"], "b_fc1")
                #else:
                #    self.W_conv5 = weight_variable_fromnp(np.zeros((6*6*256, 4096), dtype=np.float32), "w_fc1")
                #    self.B_conv5 = weight_variable_fromnp(np.zeros((4096), dtype = np.float32), "b_fc1")
                self.W_fc1 = weight_variable_xavier([numInputs, 4096], "w_fc1")
                self.B_fc1 = bias_variable([4096], "b_fc1")
                h_pool5_flat = tf.reshape(self.h_pool5, [-1, numInputs], name="pool5_flat")
                self.h_fc1 = tf.nn.relu(tf.matmul(h_pool5_flat, self.W_fc1, name="fc1") + self.B_fc1, "fc1_relu")
                self.h_fc1_drop = tf.nn.dropout(self.h_fc1, self.keep_prob)

            with tf.name_scope("FC2"):
                #if(inMatFilename):
                #    self.W_conv5 = weight_variable_fromnp(npWeights["fc2_w"], "w_fc2")
                #    self.B_conv5 = weight_variable_fromnp(npWeights["fc2_b"], "b_fc2")
                #else:
                #    self.W_conv5 = weight_variable_fromnp(np.zeros((4096, 4096), dtype=np.float32), "w_fc2")
                #    self.B_conv5 = weight_variable_fromnp(np.zeros((4096), dtype = np.float32), "b_fc2")
                self.W_fc2 = weight_variable_xavier([4096, 4096], "w_fc2")
                self.B_fc2 = bias_variable([4096], "b_fc2")
                self.h_fc2 = tf.nn.relu(tf.matmul(self.h_fc1_drop, self.W_fc2, name="fc2") + self.B_fc2, "fc2_relu")
                self.h_fc2_drop = tf.nn.dropout(self.h_fc2, self.keep_prob)

            #fc3 should have 16 channels
            #fc3 also uses a sigmoid function
            #We change it to tanh
            with tf.name_scope("FC3"):
                #if(inMatFilename):
                #    self.W_conv5 = weight_variable_fromnp(npWeights["fc3_w"], "w_fc3")
                #    self.B_conv5 = weight_variable_fromnp(npWeights["fc3_b"], "b_fc3")
                #else:
                #    self.W_conv5 = weight_variable_fromnp(np.zeros((4096, 2), dtype=np.float32), "w_fc3")
                #    self.B_conv5 = weight_variable_fromnp(np.zeros((2), dtype = np.float32), "b_fc3")
                self.W_fc3 = weight_variable_xavier([4096, 2], "w_fc3")
                self.B_fc3 = bias_variable([2], "b_fc3")
                self.est = tf.nn.softmax(tf.matmul(self.h_fc2_drop, self.W_fc3, name="fc3") + self.B_fc3, "fc3_softmax")

            with tf.name_scope("Loss"):
                #Define loss
                #self.loss = tf.reduce_mean(-tf.reduce_sum(self.gt * tf.log(self.est), reduction_indices=[1]))
                self.loss = tf.reduce_mean(-(self.gt[:, 1]*.8* tf.log(self.est[:, 1]) + self.gt[:, 0]*.2*tf.log(self.est[:, 0])))

            with tf.name_scope("Opt"):
                #Define optimizer
                #self.optimizerAll = tf.train.AdagradOptimizer(self.learningRate).minimize(self.loss)
                #self.optimizerFC = tf.train.AdagradOptimizer(self.learningRate).minimize(self.loss,
                self.optimizerAll = tf.train.AdamOptimizer(self.learningRate).minimize(self.loss)
                self.optimizerFC = tf.train.AdamOptimizer(self.learningRate).minimize(self.loss,
                        var_list=[
                            self.W_fc1,
                            self.B_fc1,
                            self.W_fc2,
                            self.B_fc2,
                            self.W_fc3,
                            self.B_fc3]
                        )

            with tf.name_scope("Metric"):
                self.gtIdx = tf.argmax(self.gt, 1)
                self.estIdx = tf.argmax(self.est, 1)
                boolGtIdx = tf.cast(self.gtIdx, tf.bool)
                boolEstIdx = tf.cast(self.estIdx, tf.bool)

                #Logical and for true positive
                lAnd = tf.logical_and(boolGtIdx, boolEstIdx)
                self.tp = tf.reduce_sum(tf.cast(lAnd, tf.float32))
                #Logical nor for true negatives
                lNor = tf.logical_not(tf.logical_or(boolGtIdx, boolEstIdx))
                self.tn = tf.reduce_sum(tf.cast(lNor, tf.float32))

                #Subtraction and comparison for others
                lSub = self.gtIdx - self.estIdx
                Ones = tf.cast(tf.ones(tf.shape(lSub)), tf.int64)
                self.fn = tf.reduce_sum(tf.cast(tf.equal(lSub, Ones), tf.float32))
                self.fp = tf.reduce_sum(tf.cast(tf.equal(lSub, -Ones), tf.float32))

                #Accuracy, precision, and recall calculations
                self.accuracy = (self.tp + self.tn)/(self.tp+self.tn+self.fp+self.fn)
                self.precision = self.tp/(self.tp+self.fp)
                self.recall = self.tp/(self.tp+self.fn)

        #Summaries
        tf.scalar_summary('loss', self.loss, name="lossSum")
        tf.scalar_summary('accuracy', self.accuracy, name="accSum")
        tf.scalar_summary('precision', self.precision, name="precSum")
        tf.scalar_summary('recall', self.recall, name="recallSum")
        tf.scalar_summary('tp', self.tp, name="tp")
        tf.scalar_summary('fp', self.fp, name="fp")
        tf.scalar_summary('tn', self.tn, name="tn")
        tf.scalar_summary('fn', self.fn, name="fn")

        tf.histogram_summary('input', self.inputImage, name="image")
        tf.histogram_summary('gt', self.gt, name="gt")
        tf.histogram_summary('conv1', self.h_pool1, name="conv1")
        tf.histogram_summary('conv2', self.h_pool2, name="conv2")
        tf.histogram_summary('conv3', self.h_conv3, name="conv3")
        tf.histogram_summary('conv4', self.h_conv4, name="conv4")
        tf.histogram_summary('conv5', self.h_pool5, name="conv5")
        tf.histogram_summary('fc1', self.h_fc1, name="fc1")
        tf.histogram_summary('fc2', self.h_fc2, name="fc2")
        tf.histogram_summary('est', self.est, name="fc3")
        tf.histogram_summary('w_conv1', self.W_conv1, name="w_conv1")
        tf.histogram_summary('b_conv1', self.B_conv1, name="b_conv1")
        tf.histogram_summary('w_conv2', self.W_conv2, name="w_conv2")
        tf.histogram_summary('b_conv2', self.B_conv2, name="b_conv2")
        tf.histogram_summary('w_conv3', self.W_conv3, name="w_conv3")
        tf.histogram_summary('b_conv3', self.B_conv3, name="b_conv3")
        tf.histogram_summary('w_conv4', self.W_conv4, name="w_conv4")
        tf.histogram_summary('b_conv4', self.B_conv4, name="b_conv4")
        tf.histogram_summary('w_conv5', self.W_conv5, name="w_conv5")
        tf.histogram_summary('b_conv5', self.B_conv5, name="b_conv5")
        tf.histogram_summary('w_fc1', self.W_fc1, name="w_fc1")
        tf.histogram_summary('b_fc1', self.B_fc1, name="b_fc1")
        tf.histogram_summary('w_fc2', self.W_fc2, name="w_fc2")
        tf.histogram_summary('b_fc2', self.B_fc2, name="b_fc2")
        tf.histogram_summary('w_fc3', self.W_fc3, name="w_fc3")
        tf.histogram_summary('b_fc3', self.B_fc3, name="b_fc3")

        #Define saver
        self.saver = tf.train.Saver()

    #Initializes session.
    def initSess(self):
        self.sess.run(tf.initialize_all_variables())

    #Allocates and specifies the output directory for tensorboard summaries
    def writeSummary(self, summaryDir):
        self.mergedSummary = tf.merge_all_summaries()
        self.train_writer = tf.train.SummaryWriter(summaryDir + "/train", self.sess.graph)
        self.test_writer = tf.train.SummaryWriter(summaryDir + "/test")

    def closeSess(self):
        self.sess.close()

    #Trains model for numSteps
    #If pre is False, will train entire network
    #If pre is True, will train only fully connected network
    def trainModel(self, dataObj, numSteps, saveFile, pre=False, miniBatchSize = 128):
        #Define session
        for i in range(numSteps):
            #Get data from dataObj
            data = dataObj.getData(miniBatchSize)
            feedDict = {self.inputImage: data[0], self.gt: data[1], self.keep_prob: .5}
            #Run optimizer
            if(pre):
                self.sess.run(self.optimizerFC, feed_dict=feedDict)
            else:
                self.sess.run(self.optimizerAll, feed_dict=feedDict)
            summary = self.sess.run(self.mergedSummary, feed_dict=feedDict)
            self.train_writer.add_summary(summary, self.timestep)
            self.timestep+=1
            if(i%self.progress == 0):
                print "Timestep ", self.timestep

        save_path = self.saver.save(self.sess, saveFile, global_step=self.timestep, write_meta_graph=False)
        print("Model saved in file: %s" % save_path)

    #Evaluates all of inData at once
    #If an inGt is provided, will calculate summary as test set
    def evalModel(self, inData, inGt = None):
        (numData, ny, nx, nf) = inData.shape
        if(inGt != None):
            (numGt, drop) = inGt.shape
            assert(numData == numGt)
            feedDict = {self.inputImage: inData, self.gt: inGt, self.keep_prob: 1}
        else:
            feedDict = {self.inputImage: inData, self.keep_prob: 1}

        outVals = self.est.eval(feed_dict=feedDict, session=self.sess)
        if(inGt != None):
            summary = self.sess.run(self.mergedSummary, feed_dict=feedDict)
            self.test_writer.add_summary(summary, self.timestep)
        return outVals

    #Evaluates inData, but in miniBatchSize batches for memory efficiency
    #If an inGt is provided, will calculate summary as test set
    def evalModelBatch(self, miniBatchSize, inData, inGt=None):
        (numData, ny, nx, nf) = inData.shape
        if(inGt != None):
            (numGt, drop) = inGt.shape
            assert(numData == numGt)

        #Split up numData into miniBatchSize and evaluate est data
        tfInVals = np.zeros((miniBatchSize, ny, nx, nf))
        outData = np.zeros((numData, 1))

        #Ceil of numData/batchSize
        numIt = int(numData/miniBatchSize) + 1

        #Only write summary on first it

        startOffset = 0
        for it in range(numIt):
            print it, " out of ", numIt
            #Calculate indices
            startDataIdx = startOffset
            endDataIdx = startOffset + miniBatchSize
            startTfValIdx = 0
            endTfValIdx = miniBatchSize

            #If out of bounds
            if(endDataIdx >= numData):
                #Calculate offset
                offset = endDataIdx - numData
                #Set endDataIdx to max value
                endDataIdx = numData
                #Set endTfValIdx to less than max value
                endTfValIdx -= offset

            tfInVals[startTfValIdx:endTfValIdx, :, :, :] = inData[startDataIdx:endDataIdx, :, :, :]
            feedDict = {self.inputImage: tfInVals, self.keep_prob: 1}
            tfOutVals = self.est.eval(feed_dict=feedDict, session=self.sess)
            outData[startDataIdx:endDataIdx, :] = tfOutVals[startTfValIdx:endTfValIdx, :]

            if(inGt != None and it == 0):
                tfInGt = inGt[startDataIdx:endDataIdx, :]
                summary = self.sess.run(self.mergedSummary, feed_dict={self.inputImage: tfInVals, self.gt: tfInGt, self.keep_prob: 1})
                self.test_writer.add_summary(summary, self.timestep)

            startOffset += miniBatchSize

        #Return output data
        return outData

    #Loads a tf checkpoint
    def loadModel(self, loadFile):
        self.saver.restore(self.sess, loadFile)
        print("Model %s loaded" % loadFile)

