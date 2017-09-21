import pdb
import numpy as np
import tensorflow as tf
from plots.plotWeights import make_plot_time
import os
from .utils import sparse_weight_variable, weight_variable, node_variable, conv3d, transpose5dData, transpose5dWeight, undoTranspose5dData, convertToSparse5d, save_sparse_csr
#import matplotlib.pyplot as plt
#from pvtools import writepvpfile


class AdamTimeSp(base):
    #Global timestep
    timestep = 0
    plotTimestep = 0

    #Sets dictionary of params to member variables
    def loadParams(self, params):
        super(AdamTimeSP, self).loadParams(params)
        #Initialize tf parameters here
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
        self.zeroThresh = params['zeroThresh']

    def runModel(self):
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
        super(AdamTimeSP, self).__init__(params, dataObj)
        self.currImg = self.dataObj.getData(self.batchSize, self.nT)

    #Builds the model. inMatFilename should be the vgg file
    def buildModel(self, inputShape):
        assert(self.nT % self.VStrideT == 0)
        assert(inputShape[0] % self.VStrideY == 0)
        assert(inputShape[1] % self.VStrideX == 0)
        V_T = int(self.nT/self.VStrideT)
        V_Y = int(inputShape[0]/self.VStrideY)
        V_X = int(inputShape[1]/self.VStrideX)
        V_Tp = int(self.patchSizeT/self.VStrideT)
        V_Yp = int(self.patchSizeY/self.VStrideY)
        V_Xp = int(self.patchSizeX/self.VStrideX)
        V_Ofp = int(inputShape[2]*self.VStrideT*self.VStrideY*self.VStrideX)

        self.imageShape = (self.batchSize, self.nT, inputShape[0], inputShape[1], inputShape[2])
        self.WShape = (V_Tp, V_Yp, V_Xp, self.numV, V_Ofp)
        self.VShape = (self.batchSize, V_T, V_Y, V_X, self.numV)

        #Running on GPU
        with tf.device(self.device):
            with tf.name_scope("inputOps"):
                #Get convolution variables as placeholders
                self.inputImage = node_variable(self.imageShape, "inputImage")
                #Scale inputImage
                self.scaled_inputImage = self.inputImage/np.sqrt(self.patchSizeX*self.patchSizeY*inputShape[2])
                #This is what it should be, but for now, we ignore the scaling with nT
                #self.scaled_inputImage = self.inputImage/np.sqrt(self.nT*self.patchSizeX*self.patchSizeY*inputShape[2])
                self.reshape_inputImage = transpose5dData(self.scaled_inputImage, self.imageShape, self.VStrideT, self.VStrideY, self.VStrideX)

            with tf.name_scope("Dictionary"):
                self.V1_W = sparse_weight_variable(self.WShape, "V1_W")
                #self.V1_W = sparse_weight_variable((self.patchSizeY, self.patchSizeX, inputShape[2], self.numV), "V1_W")

            with tf.name_scope("weightNorm"):
                self.normVals = tf.sqrt(tf.reduce_sum(tf.square(self.V1_W), reduction_indices=[0, 1, 2, 4], keep_dims=True))
                #self.normVals = tf.sqrt(tf.reduce_sum(tf.square(self.V1_W), reduction_indices=[0, 1, 2], keep_dims=True))
                self.normalize_W = self.V1_W.assign(self.V1_W/self.normVals)

            with tf.name_scope("ISTA"):
                #Variable for activity
                self.V1_A = weight_variable(self.VShape, "V1_A", 1e-4)
                self.zeroConst = tf.zeros(self.VShape)
                self.t_V1_A = tf.select(tf.abs(self.V1_A) < self.zeroThresh, self.zeroConst, self.V1_A)

                #self.V1_A= weight_variable((self.batchSize, inputShape[0], inputShape[1], self.numV), "V1_A", .01)

            with tf.name_scope("Recon"):
                assert(self.VStrideT >= 1)
                assert(self.VStrideY >= 1)
                assert(self.VStrideX >= 1)
                #We build index tensor in numpy to gather
                self.recon = conv3d(self.V1_A, self.V1_W, "recon")
                self.t_recon = conv3d(self.t_V1_A, self.V1_W, "recon")

            with tf.name_scope("Error"):
                self.error = self.reshape_inputImage - self.recon
                self.t_error = self.reshape_inputImage - self.t_recon

            with tf.name_scope("Loss"):
                self.reconError = tf.reduce_sum(tf.square(self.error))
                self.l1Sparsity = tf.reduce_sum(tf.abs(self.V1_A))
                #Define loss
                self.loss = self.reconError/2 + self.thresh * self.l1Sparsity

                self.t_reconError = tf.reduce_sum(tf.square(self.t_error))
                self.t_l1Sparsity = tf.reduce_sum(tf.abs(self.t_V1_A))
                #Define loss
                self.t_loss = self.t_reconError/2 + self.thresh * self.t_l1Sparsity

            with tf.name_scope("Opt"):
                #Define optimizer
                #self.optimizerA = tf.train.GradientDescentOptimizer(self.learningRateA).minimize(self.loss,
                self.optimizerA = tf.train.AdamOptimizer(self.learningRateA).minimize(self.loss,
                        var_list=[
                            self.V1_A
                        ])
                #self.optimizerW = tf.train.GradientDescentOptimizer(self.learningRateW).minimize(self.loss,
                self.optimizerW = tf.train.AdamOptimizer(self.learningRateW).minimize(self.loss,
                        var_list=[
                            self.V1_W
                        ])

            with tf.name_scope("stats"):
                self.underThresh = tf.reduce_mean(tf.cast(tf.abs(self.V1_A) > self.zeroThresh, tf.float32))

                self.errorStd = tf.sqrt(tf.reduce_mean(tf.square(self.error-tf.reduce_mean(self.error))))*np.sqrt(self.patchSizeY*self.patchSizeX*inputShape[2])
                self.l1_mean = tf.reduce_mean(tf.abs(self.V1_A))

                self.t_errorStd = tf.sqrt(tf.reduce_mean(tf.square(self.t_error-tf.reduce_mean(self.t_error))))*np.sqrt(self.patchSizeY*self.patchSizeX*inputShape[2])
                self.t_l1_mean = tf.reduce_mean(tf.abs(self.t_V1_A))

                #Reshape weights for viewing
                self.reshape_weight = transpose5dWeight(self.V1_W, self.WShape, self.VStrideT, self.VStrideY, self.VStrideX)
                self.weightImages = tf.reshape(tf.transpose(self.reshape_weight, [3, 0, 1, 2, 4]), [self.numV*self.patchSizeT, self.patchSizeY, self.patchSizeX, inputShape[2]])
                #For image viewing
                self.frameImages = self.inputImage[0, :, :, :, :]
                self.reshaped_recon = undoTranspose5dData(self.recon, self.imageShape, self.VStrideT, self.VStrideY, self.VStrideX)
                self.frameRecons = self.reshaped_recon[0, :, :, :, :]

                self.t_reshaped_recon = undoTranspose5dData(self.t_recon, self.imageShape, self.VStrideT, self.VStrideY, self.VStrideX)
                self.t_frameRecons = self.t_reshaped_recon[0, :, :, :, :]
                #For log of activities
                self.log_V1_A = tf.log(tf.abs(self.V1_A)+1e-15)


        #Summaries
        self.s_loss = tf.scalar_summary('loss', self.loss, name="lossSum")
        self.s_recon = tf.scalar_summary('recon error', self.reconError, name="reconError")
        self.s_errorStd= tf.scalar_summary('errorStd', self.errorStd, name="errorStd")
        self.s_l1= tf.scalar_summary('l1 sparsity', self.l1Sparsity, name="l1Sparsity")
        self.s_l1_mean = tf.scalar_summary('l1 mean', self.l1_mean, name="l1Mean")
        self.s_s_nnz = tf.scalar_summary('nnz', self.underThresh, name="nnz")

        self.s_t_loss = tf.scalar_summary('t loss', self.t_loss, name="t_lossSum")
        self.s_t_recon = tf.scalar_summary('t recon error', self.t_reconError, name="t_reconError")
        self.s_t_errorStd= tf.scalar_summary('t errorStd', self.t_errorStd, name="t_errorStd")
        self.s_t_l1= tf.scalar_summary('t l1 sparsity', self.t_l1Sparsity, name="t_l1Sparsity")
        self.s_t_l1_mean = tf.scalar_summary('t l1 mean', self.t_l1_mean, name="t_l1Mean")

        self.h_input = tf.histogram_summary('input', self.inputImage, name="input")
        self.h_recon = tf.histogram_summary('recon', self.recon, name="recon")
        self.h_v1_w = tf.histogram_summary('V1_W', self.V1_W, name="V1_W")

        self.h_v1_a = tf.histogram_summary('V1_A', self.V1_A, name="V1_A")
        self.h_log_v1_a = tf.histogram_summary('Log_V1_A', self.log_V1_A, name="Log_V1_A")

        self.h_normVals = tf.histogram_summary('normVals', self.normVals, name="normVals")

        #Images
        #self.i_w = tf.image_summary("weights", self.weightImages, max_images=self.numV)
        #self.i_orig = tf.image_summary("orig", self.frameImages, max_images=self.nT)
        #self.i_recon = tf.image_summary("recon", self.frameRecons, max_images=self.nT)
        #self.i_t_recon = tf.image_summary("t_recon", self.t_frameRecons, max_images=self.nT)

    #Trains model for numSteps
    def trainA(self, save):
        #Define session
        feedDict = {self.inputImage: self.currImg}
        for i in range(self.displayPeriod):
            #Run optimizer
            self.sess.run(self.optimizerA, feed_dict=feedDict)
            self.timestep+=1
            if((i+1)%self.writeStep == 0):
                summary = self.sess.run(self.mergedSummary, feed_dict=feedDict)
                self.train_writer.add_summary(summary, self.timestep)
            if((i+1)%self.progress == 0):
                print("Timestep ", self.timestep)

        if(save):
            save_path = self.saver.save(self.sess, self.saveFile, global_step=self.timestep, write_meta_graph=False)
            print("Model saved in file: %s" % save_path)

    def normWeights(self):
        #Normalize weights
        self.sess.run(self.normalize_W)

    def trainW(self):
        feedDict = {self.inputImage: self.currImg}

        #Visualization
        if (self.plotTimestep % self.plotPeriod == 0):
            np_V1_W = self.sess.run(self.reshape_weight)
            make_plot_time(np_V1_W, self.plotDir+"dict_"+str(self.timestep))
            #Write summary
            #summary = self.sess.run(self.imageSummary, feed_dict=feedDict)
            #self.train_writer.add_summary(summary, self.timestep)

        #Update weights
        self.sess.run(self.optimizerW, feed_dict=feedDict)
        #New image
        self.currImg = self.dataObj.getData(self.batchSize, self.nT)
        self.plotTimestep += 1


    #Finds sparse encoding of inData
    #inData must be in the shape of the image
    #[batch, nT, nY, nX, nF]
    def evalData(self, inData, displayPeriod=None):
        (nb, nt, ny, nx, nf) = inData.shape
        #Check size
        assert(nb == self.batchSize)
        assert(nt == self.nT)
        assert(ny == self.inputShape[0])
        assert(nx == self.inputShape[1])
        assert(nf == self.inputShape[2])

        if(not displayPeriod):
            displayPeriod=self.displayPeriod

        feedDict = {self.inputImage: inData}
        #Optimize V for displayPeriod amount
        for i in range(self.displayPeriod):
            self.sess.run(self.optimizerA, feed_dict=feedDict)
            if((i+1)%self.progress == 0):
                print("Timestep ", str(i) , " out of ", str(self.displayPeriod))
        #Get thresholded v1 as an output
        outVals = self.t_V1_A.eval(session=self.sess)
        return outVals

    def evalSet(self, evalDataObj, outPrefix, displayPeriod=None):
        numImages = evalDataObj.numImages
        numIterations = int(numImages/evalDataObj.skip)
        #batchSize must be 1 for now
        assert(self.batchSize == 1)
        #Open h5py file
        for it in range(numIterations):
            print(str((float(it)*100)/numIterations) + "% done (" + str(it) + " out of " + str(numIterations) + ")")
            #Evaluate
            npV1_A = self.evalData(evalDataObj.getData(self.batchSize, self.nT), displayPeriod=displayPeriod)
            v1Sparse = convertToSparse5d(npV1_A)
            save_sparse_csr(outPrefix+str(it), v1Sparse)

    #def writePvpWeights(self, outputPrefix):
    #    npw = self.sess.run(self.reshape_weight)
    #    [ntp, nyp, nxp, numK, nfp] = npw.shape
    #    for itp in range(ntp):
    #        filename = outputPrefix + "_t" + str(itp) + ".pvp"
    #        #We need to get weights into pvp shape
    #        #6D dense numpy array of size [numFrames, numArbors, numKernels, ny, nx, nf]
    #        outWeights = np.zeros((1, 1, numK*2, nyp, nxp, nfp))
    #        weightBuf = np.transpose(npw[itp, :, :, :, :], [2, 0, 1, 3])
    #        outWeights[0, 0, 0:numK, :, :, :] = weightBuf
    #        outWeights[0, 0, numK:2*numK, :, :, :] = weightBuf * -1
    #        pvp = {}
    #        pvp['values'] = outWeights
    #        pvp['time'] = np.array([0])
    #        writepvpfile(filename, pvp)

