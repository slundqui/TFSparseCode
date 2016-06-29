import pdb
import numpy as np
import tensorflow as tf
from utils import *
import os
import matplotlib.pyplot as plt

class SLP:
    #Global timestep
    timestep = 0
    plotTimestep = 0

    #Constructor takes inputShape, which is a 3 tuple (ny, nx, nf) based on the size of the image being fed in
    def __init__(self, params, inputShape):
        self.loadParams(params)
        self.makeDirs()
        #self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        self.sess = tf.Session()
        self.buildModel(inputShape)

    #Sets dictionary of params to member variables
    def loadParams(self, params):
        #Initialize tf parameters here
        self.outDir = params['outDir']
        self.runDir = self.outDir + params['runDir']
        self.ckptDir = self.runDir + params['ckptDir']
        self.plotDir = self.runDir + params['plotDir']
        self.tfDir = self.runDir + params['tfDir']
        self.saveFile = self.ckptDir + params['saveFile']

        self.load = params['load']
        self.loadFile = params['loadFile']
        self.outerSteps = params['outerSteps']
        self.innerSteps = params['innerSteps']
        self.savePeriod = params['savePeriod']
        self.plotPeriod = params['plotPeriod']
        self.writeStep = params['writeStep']

        self.VStrideY = params['VStrideY']
        self.VStrideX = params['VStrideX']

        self.device = params['device']
        self.batchSize = params['batchSize']
        self.learningRate = params['learningRate']
        self.numClasses = params['numClasses']

        self.pvpWeightFile = params['pvpWeightFile']

        self.progress = params['progress']

    #Make approperiate directories if they don't exist
    def makeDirs(self):
        if not os.path.exists(self.runDir):
           os.makedirs(self.runDir)
        if not os.path.exists(self.plotDir):
           os.makedirs(self.plotDir)
        if not os.path.exists(self.ckptDir):
           os.makedirs(self.ckptDir)

    def runModel(self, trainDataObj, testDataObj=None):
        #Load summary
        self.writeSummary()
        for i in range(self.outerSteps):
           #Plot flag
           if(i%self.plotPeriod == 0):
               plot = True
           else:
               plot=False
           if(testDataObj):
               #Evaluate test frame, providing gt so that it writes to summary
               (evalData, gtData) = testDataObj.getData(self.batchSize)
               self.evalModel(evalData, gtData, plot=plot)
               print "Done test eval"
           #Train
           if(i%self.savePeriod == 0):
               self.trainModel(trainDataObj, save=True, plot=plot)
           else:
               self.trainModel(trainDataObj, save=False, plot=plot)


    #Builds the model. inMatFilename should be the vgg file
    def buildModel(self, inputShape):
        if(self.pvpWeightFile):
            npWeights = load_pvp_weights(self.pvpWeightFile)
        else:
            print "Must load from weights"
            assert(0)

        #Running on GPU
        with tf.device(self.device):
            self.imageShape = (self.batchSize, inputShape[0]*self.VStrideY, inputShape[1]*self.VStrideX, 3)
            with tf.name_scope("inputOps"):
                #Get convolution variables as placeholders
                self.input = node_variable([self.batchSize, inputShape[0], inputShape[1], inputShape[2]], "inputImage")
                self.gt = node_variable([self.batchSize, self.numClasses], "gt")
                #Model variables for convolutions

            with tf.name_scope("SLP"):
                #Max pooled values
                self.W_slp = weight_variable_xavier((inputShape[2], self.numClasses), "slp_w", False)
                self.B_slp = bias_variable([self.numClasses], "slp_b")
                self.pooled = tf.reduce_max(self.input, reduction_indices=[1, 2])
                self.est = tf.nn.softmax(tf.matmul(self.pooled, self.W_slp) + self.B_slp)

            with tf.name_scope("recon"):
                self.W_dict = weight_variable_fromnp(npWeights, "W_dict")
                self.recon = conv2d_oneToMany(self.input, self.W_dict, self.imageShape, "recon", self.VStrideY, self.VStrideX)

            with tf.name_scope("Loss"):
                #Define loss
                self.loss = tf.reduce_mean(-tf.reduce_sum(self.gt * tf.log(self.est), reduction_indices=[1]))

            with tf.name_scope("Opt"):
                #Define optimizer
                self.optimizer = tf.train.AdamOptimizer(self.learningRate).minimize(self.loss)

            with tf.name_scope("Metric"):
                self.correct = tf.equal(tf.argmax(self.gt, 1), tf.argmax(self.est, 1))
                self.accuracy = tf.reduce_mean(tf.cast(self.correct, tf.float32))

        #Summaries
        tf.scalar_summary('loss', self.loss, name="loss")
        tf.scalar_summary('accuracy', self.accuracy, name="accuracy")

        tf.histogram_summary('input', self.input, name="image")
        tf.histogram_summary('pooled', self.pooled, name="pooled")
        tf.histogram_summary('gt', self.gt, name="gt")
        #Conv layer histograms
        tf.histogram_summary('est', self.est, name="est")
        #Weight and bias hists
        tf.histogram_summary('w_slp', self.W_slp, name="w_slp")
        tf.histogram_summary('b_slp', self.B_slp, name="b_slp")

        #tf.image_summary("cam", self.sortedCamImg, max_images=10)
        #tf.image_summary("input", self.inputImage, max_images=1)

        #Define saver
        self.saver = tf.train.Saver()

        #Initialize
        #Load checkpoint if flag set
        if(self.load):
           self.loadModel()
           ##We only load weights, so we need to initialize A
           #un_vars = list(tf.get_variable(name) for name in self.sess.run(tf.report_uninitialized_variables(tf.all_variables())))
           #tf.initialize_variables(un_vars)
        else:
           self.initSess()

    #Initializes session.
    def initSess(self):
        self.sess.run(tf.initialize_all_variables())

    #Allocates and specifies the output directory for tensorboard summaries
    def writeSummary(self):
        self.mergedSummary = tf.merge_all_summaries()
        self.train_writer = tf.train.SummaryWriter(self.tfDir + "/train", self.sess.graph)
        self.test_writer = tf.train.SummaryWriter(self.tfDir + "/test")

    def closeSess(self):
        self.sess.close()

    #Trains model for numSteps
    #If pre is False, will train entire network
    #If pre is True, will train only fully connected network
    def trainModel(self, dataObj, save, plot):
        #Define session
        for i in range(self.innerSteps):
            #Get data from dataObj
            data = dataObj.getData(self.batchSize)
            feedDict = {self.input: data[0], self.gt: data[1]}
            #Run optimizer
            self.sess.run(self.optimizer, feed_dict=feedDict)
            if(i%self.writeStep == 0):
                summary = self.sess.run(self.mergedSummary, feed_dict=feedDict)
                self.train_writer.add_summary(summary, self.timestep)
            if(i%self.progress == 0):
                print "Timestep ", self.timestep
            pdb.set_trace()
            self.timestep+=1
        if(save):
            save_path = self.saver.save(self.sess, self.saveFile, global_step=self.timestep, write_meta_graph=False)
            print("Model saved in file: %s" % save_path)
        #if(plot):
        #    filename = self.plotDir + "train_" + str(self.timestep) + ".png"
        #    self.evalAndPlotCam(feedDict, filename)

    #Evaluates all of inData at once
    #If an inGt is provided, will calculate summary as test set
    def evalModel(self, inData, inGt = None, plot=True):
        (numData, ny, nx, nf) = inData.shape
        if(inGt != None):
            (numGt, drop) = inGt.shape
            assert(numData == numGt)
            feedDict = {self.input: inData, self.gt: inGt}
        else:
            feedDict = {self.input: inData}

        outVals = self.est.eval(feed_dict=feedDict, session=self.sess)
        if(inGt != None):
            summary = self.sess.run(self.mergedSummary, feed_dict=feedDict)
            self.test_writer.add_summary(summary, self.timestep)
        pdb.set_trace()
        #if(plot and inGt != None):
        #    filename = self.plotDir + "test_" + str(self.timestep) + ".png"
        #    self.evalAndPlotCam(feedDict, filename)

        return outVals

    ##Evaluates inData, but in miniBatchSize batches for memory efficiency
    ##If an inGt is provided, will calculate summary as test set
    #def evalModelBatch(self, inData, inGt=None):
    #    (numData, ny, nx, nf) = inData.shape
    #    if(inGt != None):
    #        (numGt, drop) = inGt.shape
    #        assert(numData == numGt)

    #    #Split up numData into miniBatchSize and evaluate est data
    #    tfInVals = np.zeros((self.batchSize, ny, nx, nf))
    #    outData = np.zeros((numData, 1))

    #    #Ceil of numData/batchSize
    #    numIt = int(numData/self.batchSize) + 1

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
    #        feedDict = {self.inputImage: tfInVals}
    #        tfOutVals = self.est.eval(feed_dict=feedDict, session=self.sess)
    #        outData[startDataIdx:endDataIdx, :] = tfOutVals[startTfValIdx:endTfValIdx, :]

    #        if(inGt != None and it == 0):
    #            tfInGt = inGt[startDataIdx:endDataIdx, :]
    #            summary = self.sess.run(self.mergedSummary, feed_dict={self.inputImage: tfInVals, self.gt: tfInGt})
    #            self.test_writer.add_summary(summary, self.timestep)

    #        startOffset += miniBatchSize

    #    #Return output data
    #    return outData

    #Loads a tf checkpoint
    def loadModel(self):
        self.saver.restore(self.sess, self.loadFile)
        print("Model %s loaded" % self.loadFile)

