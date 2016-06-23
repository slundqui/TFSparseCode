import pdb
import numpy as np
import tensorflow as tf
from plots.plotWeights import make_plot_time
import os
from .utils import sparse_weight_variable, weight_variable, node_variable, conv3d
#import matplotlib.pyplot as plt

class ISTA_Time:
    #Global timestep
    timestep = 0
    plotTimestep = 0

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
        self.numIterations= params['numIterations']
        self.displayPeriod = params['displayPeriod']
        self.savePeriod = params['savePeriod']
        self.plotPeriod = params['plotPeriod']
        self.batchSize = params['batchSize']
        self.learningRateA = params['learningRateA']
        self.learningRateW = params['learningRateW']
        self.thresh = params['thresh']
        self.numV = params['numV']
        self.nT = params['nT']
        self.VStrideT = params['VStrideT']
        self.VStrideY = params['VStrideY']
        self.VStrideX = params['VStrideX']
        self.patchSizeT = params['patchSizeT']
        self.patchSizeY = params['patchSizeY']
        self.patchSizeX = params['patchSizeX']
        self.progress = params['progress']
        self.writeStep = params['writeStep']

    #Make approperiate directories if they don't exist
    def makeDirs(self):
        if not os.path.exists(self.runDir):
           os.makedirs(self.runDir)
        if not os.path.exists(self.plotDir):
           os.makedirs(self.plotDir)
        if not os.path.exists(self.ckptDir):
           os.makedirs(self.ckptDir)

    def runModel(self):
        #Initialize
        #Load checkpoint if flag set
        if(self.load):
           self.loadModel()
        else:
           self.initSess()


        #Load summary
        self.writeSummary()

        #Normalize weights to start
        self.normWeights()

        #Training
        for i in range(self.numIterations):
           if(i%self.savePeriod == 0):
               self.trainA(True)
           else:
               self.trainA(False)
           #Train
           self.trainW()
           self.normWeights()

    #Constructor takes inputShape, which is a 3 tuple (ny, nx, nf) based on the size of the image being fed in
    def __init__(self, params, dataObj):
        self.loadParams(params)
        self.makeDirs()
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        self.dataObj = dataObj
        self.buildModel(self.dataObj.inputShape)
        self.currImg = self.dataObj.getData(self.batchSize, self.nT)

    #Builds the model. inMatFilename should be the vgg file
    def buildModel(self, inputShape):
        assert(self.nT % self.VStrideT == 0)
        assert(inputShape[0] % self.VStrideY == 0)
        assert(inputShape[1] % self.VStrideX == 0)
        V_T = int(self.nT/self.VStrideT)
        V_Y = int(inputShape[0]/self.VStrideY)
        V_X = int(inputShape[1]/self.VStrideX)

        imageShape = (self.batchSize, self.nT, inputShape[0], inputShape[1], inputShape[2])
        WShape = (self.patchSizeT, self.patchSizeY, self.patchSizeX, self.numV, inputShape[2])
        VShape = (self.batchSize, V_T, V_Y, V_X, self.numV)

        #Running on GPU
        #with tf.device('gpu:0'):
        with tf.name_scope("inputOps"):
            #Get convolution variables as placeholders
            self.inputImage = node_variable(imageShape, "inputImage")
            #Scale inputImage
            self.scaled_inputImage = self.inputImage/np.sqrt(self.patchSizeX*self.patchSizeY*inputShape[2])
            #This is what it should be, but for now, we ignore the scaling with nT
            #self.scaled_inputImage = self.inputImage/np.sqrt(self.nT*self.patchSizeX*self.patchSizeY*inputShape[2])

        with tf.name_scope("Dictionary"):
            self.V1_W = sparse_weight_variable(WShape, "V1_W")
            #self.V1_W = sparse_weight_variable((self.patchSizeY, self.patchSizeX, inputShape[2], self.numV), "V1_W")

        with tf.name_scope("weightNorm"):
            self.normVals = tf.sqrt(tf.reduce_sum(tf.square(self.V1_W), reduction_indices=[0, 1, 2, 4], keep_dims=True))
            #self.normVals = tf.sqrt(tf.reduce_sum(tf.square(self.V1_W), reduction_indices=[0, 1, 2], keep_dims=True))
            self.normalize_W = self.V1_W.assign(self.V1_W/self.normVals)

        with tf.name_scope("ISTA"):
            #Soft threshold
            self.V1_A= weight_variable(VShape, "V1_A", 1e-4)
            #self.V1_A= weight_variable((self.batchSize, inputShape[0], inputShape[1], self.numV), "V1_A", .01)

        with tf.name_scope("Recon"):
            assert(self.VStrideT >= 1)
            assert(self.VStrideY >= 1)
            assert(self.VStrideX >= 1)
            #We build index tensor in numpy to gather
            if(self.VStrideT == 1 and self.VStrideY == 1 and self.VStrideX == 1):
                self.recon = conv3d(self.V1_A, self.V1_W, "recon")
            else:
                print "oneToMany 3d convolutions not implemented yet"
                assert(0)
                #self.recon = conv3d_oneToMany(self.V1_A, VShape, self.V1_W, WShape, self.VStrideT, self.VStrideY, self.VStrideX, "recon")

        with tf.name_scope("Error"):
            self.error = self.scaled_inputImage - self.recon

        with tf.name_scope("Loss"):
            self.reconError = tf.reduce_sum(tf.square(self.error))
            self.l1Sparsity = tf.reduce_sum(tf.abs(self.V1_A))
            #Define loss
            self.loss = self.reconError/2 + self.thresh * self.l1Sparsity

        with tf.name_scope("Opt"):
            #Define optimizer
            #self.optimizerA = tf.train.GradientDescentOptimizer(self.learningRateA).minimize(self.loss,
            self.optimizerA = tf.train.AdamOptimizer(self.learningRateA).minimize(self.loss,
                    var_list=[
                        self.V1_A
                    ])
            self.optimizerW = tf.train.AdamOptimizer(self.learningRateW).minimize(self.loss,
            #self.optimizerW = tf.train.GradientDescentOptimizer(self.learningRateW).minimize(self.loss,
                    var_list=[
                        self.V1_W
                    ])

        with tf.name_scope("stats"):
            self.errorStd = tf.sqrt(tf.reduce_mean(tf.square(self.error-tf.reduce_mean(self.error))))*np.sqrt(self.patchSizeY*self.patchSizeX*inputShape[2])
            self.underThresh = tf.reduce_mean(tf.cast(tf.abs(self.V1_A) > self.thresh, tf.float32))
            self.l1_mean = tf.reduce_mean(tf.abs(self.V1_A))
            self.weightImages = tf.reshape(tf.transpose(self.V1_W, [3, 0, 1, 2, 4]), [WShape[3]*WShape[0], WShape[1], WShape[2], WShape[4]])
            self.frameImages = self.inputImage[0, :, :, :, :]
            self.frameRecons = self.recon[0, :, :, :, :]


        #Summaries
        self.s_loss = tf.scalar_summary('loss', self.loss, name="lossSum")
        self.s_recon = tf.scalar_summary('recon error', self.reconError, name="reconError")
        self.s_errorStd= tf.scalar_summary('errorStd', self.errorStd, name="errorStd")
        self.s_l1= tf.scalar_summary('l1 sparsity', self.l1Sparsity, name="l1Sparsity")
        self.s_l1_mean = tf.scalar_summary('l1 mean', self.l1_mean, name="l1Mean")
        self.s_s_nnz = tf.scalar_summary('nnz', self.underThresh, name="nnz")

        self.h_input = tf.histogram_summary('input', self.inputImage, name="input")
        self.h_recon = tf.histogram_summary('recon', self.recon, name="recon")
        self.h_v1_w = tf.histogram_summary('V1_W', self.V1_W, name="V1_W")
        self.h_v1_a = tf.histogram_summary('V1_A', self.V1_A, name="V1_A")
        self.h_normVals = tf.histogram_summary('normVals', self.normVals, name="normVals")

        #Images
        self.i_w = tf.image_summary("weights", self.weightImages, max_images=self.numV)
        self.i_orig = tf.image_summary("orig", self.frameImages, max_images=self.nT)
        self.i_recon = tf.image_summary("recon", self.frameRecons, max_images=self.nT)

        #Define saver
        self.saver = tf.train.Saver()

    #Initializes session.
    def initSess(self):
        self.sess.run(tf.initialize_all_variables())

    #Allocates and specifies the output directory for tensorboard summaries
    def writeSummary(self):
        self.mergedSummary = tf.merge_summary([
            self.s_loss,
            self.s_recon,
            self.s_l1,
            self.s_l1_mean,
            self.h_input,
            self.h_recon,
            self.h_v1_w,
            self.h_v1_a,
            self.h_normVals,
            self.s_errorStd,
            self.s_s_nnz
            ])
        self.imageSummary = tf.merge_summary([
            self.i_w, self.i_orig, self.i_recon
            ])
        self.train_writer = tf.train.SummaryWriter(self.tfDir + "/train", self.sess.graph)
        #self.test_writer = tf.train.SummaryWriter(self.tfDir+ "/test")

    def closeSess(self):
        self.sess.close()

    #Trains model for numSteps
    def trainA(self, save):
        #Define session
        for i in range(self.displayPeriod):
            feedDict = {self.inputImage: self.currImg}
            #Run optimizer
            self.sess.run(self.optimizerA, feed_dict=feedDict)
            self.timestep+=1
            if((i+1)%self.writeStep == 0):
                summary = self.sess.run(self.mergedSummary, feed_dict=feedDict)
                self.train_writer.add_summary(summary, self.timestep)
            if((i+1)%self.progress == 0):
                print "Timestep ", self.timestep

        if(save):
            save_path = self.saver.save(self.sess, self.saveFile, global_step=self.timestep, write_meta_graph=False)
            print("Model saved in file: %s" % save_path)

    def normWeights(self):
        #Normalize weights
        self.sess.run(self.normalize_W)

    def trainW(self):
        if (self.plotTimestep % self.plotPeriod == 0):
            np_V1_W = self.sess.run(self.V1_W)
            make_plot_time(np_V1_W, self.plotDir+"dict_"+str(self.timestep)+".png")

        feedDict = {self.inputImage: self.currImg}
        #Update weights
        self.sess.run(self.optimizerW, feed_dict=feedDict)
        #Write summary
        summary = self.sess.run(self.imageSummary, feed_dict=feedDict)
        self.train_writer.add_summary(summary, self.timestep)
        #New image
        self.currImg = self.dataObj.getData(self.batchSize, self.nT)
        self.plotTimestep += 1


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
    def loadModel(self):
        self.saver.restore(self.sess, self.loadFile)
        print("Model %s loaded" % self.loadFile)

