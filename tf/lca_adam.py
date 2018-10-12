import pdb
import numpy as np
import tensorflow as tf
from TFSparseCode.tf.base import base
from TFSparseCode.plots.plotWeights import plot_weights, plot_1d_weights
from TFSparseCode.plots.plotRecon import plotRecon1d, plotRecon
from TFSparseCode.tf.utils import *
import os
import time
#Using pvp files for saving
#import pvtools as pv

class LCA_ADAM(base):
    #Sets dictionary of params to member variables
    def loadParams(self, params):
        super(LCA_ADAM, self).loadParams(params)
        self.learningRateA = params['learningRateA']
        self.learningRateW = params['learningRateW']
        self.thresh = params['thresh']
        self.numV = params['numV']
        self.VStrideY = params['VStrideY']
        self.VStrideX = params['VStrideX']
        self.fc         = params['fc']
        self.patchSizeY = params['patchSizeY']
        self.patchSizeX = params['patchSizeX']
        self.inputMult = params['inputMult']
        self.normalize = params['normalize']
        self.plot_groups = params['plot_groups']
        self.plot_group_title = params['plot_group_title']

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
           #This function is responsible for determining when to plot per iteration
           #self.plot()

    #Constructor takes inputShape, which is a 3 tuple (ny, nx, nf) based on the size of the image being fed in
    def __init__(self, params, dataObj):
        super(LCA_ADAM, self).__init__(params, dataObj)
        #TODO make mask optional
        (self.currImg, self.currMask) = self.dataObj.getData(self.batchSize)

    #Builds the model. inMatFilename should be the vgg file
    def buildModel(self, inputShape):
        self.imageShape = (self.batchSize, inputShape[0], inputShape[1], inputShape[2])
        if self.fc:
            self.WShape = (self.imageShape[1]*self.imageShape[2]*self.imageShape[3], self.numV)
            self.VShape = (self.batchSize, self.numV)
        else:
            assert(self.imageShape[1] % self.VStrideY == 0)
            assert(self.imageShape[2] % self.VStrideX == 0)
            V_Y = int(self.imageShape[1]/self.VStrideY)
            V_X = int(self.imageShape[2]/self.VStrideX)
            self.WShape = (self.patchSizeY, self.patchSizeX, self.imageShape[3], self.numV)
            self.VShape = (self.batchSize, V_Y, V_X, self.numV)

        #Running on GPU
        with tf.device(self.device):
            with tf.name_scope("inputOps"):
                #Get convolution variables as placeholders
                self.inputImage = node_variable(self.imageShape, "inputImage")
                defaultMask = tf.zeros(self.imageShape)
                self.inputMask = tf.placeholder_with_default(defaultMask, self.imageShape)

                #Normalize image
                if(self.normalize):
                    n = tf.reduce_sum(1-self.inputMask, axis=[1, 2], keepdims=True)
                    #Avoid divide by 0
                    n = tf.where(tf.equal(n, 0), tf.ones(n.shape), n)
                    self.data_mean = tf.reduce_sum(self.inputImage, axis=[1,2], keepdims=True)/n
                    self.data_std = tf.sqrt(tf.reduce_sum(tf.square(self.inputImage - self.data_mean), axis=[1, 2], keepdims=True)/n)
                    #Avoid divide by 0
                    self.data_std = tf.where(tf.equal(self.data_std, 0), tf.ones(self.data_std.shape), self.data_std)

                    self.scaled_inputImage = (self.inputImage - self.data_mean) / self.data_std

                #Scale inputImage
                if(self.fc):
                    #TODO is this necessary for fc?
                    #self.scaled_inputImage = self.inputImage/(np.sqrt(self.imageShape[1]*self.imageShape[2]*self.imageShape[3]))
                    self.scaled_inputImage = self.scaled_inputImage
                else:
                    self.patch_norm = np.sqrt(self.patchSizeX * self.patchSizeY*self.imageShape[3])
                    self.scaled_inputImage = self.scaled_inputImage/self.patch_norm
                self.scaled_inputImage = self.scaled_inputImage * self.inputMult
                #self.checked_inputImage = tf.check_numerics(self.scaled_inputImage, "scaled_input error", name=None)

            with tf.name_scope("Dictionary"):
                self.V1_W = weight_variable(self.WShape, "V1_W", 1e-3)

            with tf.name_scope("weightNorm"):
                if(self.fc):
                    self.normVals = tf.sqrt(tf.reduce_sum(tf.square(self.V1_W), axis=[0], keepdims=True))
                else:
                    self.normVals = tf.sqrt(tf.reduce_sum(tf.square(self.V1_W), axis=[0, 1, 2], keepdims=True))
                self.normVals = tf.verify_tensor_all_finite(self.normVals, 'V1W error', name=None)
                self.normalize_W = self.V1_W.assign(self.V1_W/(self.normVals + 1e-8))

            with tf.name_scope("LCA_ADAM"):
                self.V1_init = tf.random_uniform(self.VShape, 0, 1.25*self.thresh, dtype=tf.float32)
                self.V1_U = uniform_weight_variable(self.VShape, "V1_U", 0.0, 1.25*self.thresh)
                self.V1_A = weight_variable(self.VShape, "V1_A", 1e-3)

            with tf.name_scope("Recon"):
                if(self.fc):
                    flat_recon = tf.matmul(self.V1_A, self.V1_W, transpose_b=True, a_is_sparse=False)
                    #Reshape recon into image shape
                    self.recon = tf.reshape(flat_recon, self.imageShape)
                else:
                    assert(self.VStrideY >= 1)
                    assert(self.VStrideX >= 1)
                    self.recon = tf.nn.conv2d_transpose(self.V1_A, self.V1_W, self.imageShape, [1, self.VStrideY, self.VStrideX, 1], padding='SAME', name="recon")

                #Unnormalize
                self.unscaled_recon = self.recon/self.inputMult

                if (self.fc):
                    pass
                else:
                    self.unscaled_recon = self.unscaled_recon * self.patch_norm

                if(self.normalize):
                    self.unscaled_recon = (self.unscaled_recon * self.data_std) + self.data_mean
                else:
                    self.unscaled_recoon = recon


                #self.recon = tf.check_numerics(self.recon, 'recon error', name=None)

            with tf.name_scope("Error"):
                self.error = (1 - self.inputMask) * (self.scaled_inputImage - self.recon)

            with tf.name_scope("Loss"):
                if(self.fc):
                    self.reconError = tf.reduce_mean(tf.reduce_sum(tf.square(self.error), axis=[1]))
                    self.l1Sparsity = tf.reduce_mean(tf.reduce_sum(tf.abs(self.V1_A), axis=[1]))
                else:
                    self.reconError = tf.reduce_mean(tf.reduce_sum(tf.square(self.error), axis=[1, 2, 3]))
                    self.l1Sparsity = tf.reduce_mean(tf.reduce_sum(tf.abs(self.V1_A), axis=[1, 2, 3]))
                #self.reconError = tf.reduce_mean(tf.square(self.error))
                #self.l1Sparsity = tf.reduce_mean(tf.abs(self.V1_A))
                #Define loss
                self.loss = self.reconError/2 + self.thresh * self.l1Sparsity

            with tf.name_scope("Opt"):
                #Calculate A from U
                self.optimizerA0 = self.V1_A.assign(tf.nn.relu(self.V1_U - self.thresh))
                self.v1Reset = self.V1_U.assign(self.V1_init)

                self.optimizerA1 = tf.train.AdamOptimizer(self.learningRateA)

                #Find gradient wrt A
                self.lossGrad = self.optimizerA1.compute_gradients(self.reconError, [self.V1_A])
                #self.checkGrad = tf.check_numerics(self.lossGrad[0][0], "grad error", name=None)
                self.dU = [(self.lossGrad[0][0] - self.V1_A + self.V1_U, self.V1_U)];

                #TODO add momentum or ADAM here
                self.optimizerA = self.optimizerA1.apply_gradients(self.dU)

                #self.optimizerW = tf.train.AdadeltaOptimizer(self.learningRateW, epsilon=1e-6).minimize(self.loss,
                self.optimizerW = tf.train.AdamOptimizer(self.learningRateW, epsilon=1e-6).minimize(self.loss,
                        var_list=[
                            self.V1_W
                        ])

            with tf.name_scope("stats"):
                self.nnz = tf.reduce_mean(tf.cast(tf.not_equal(self.V1_A, 0), tf.float32))

                self.imageStd = tf.sqrt(tf.reduce_mean(tf.square(self.scaled_inputImage - tf.reduce_mean(self.scaled_inputImage))))
                self.errorStd = tf.sqrt(tf.reduce_mean(tf.square(self.error-tf.reduce_mean(self.error))))/self.imageStd
                self.l1_mean = tf.reduce_mean(tf.abs(self.V1_A))
                if(self.fc):
                    flat_weightImages = tf.transpose(self.V1_W, [1, 0]) #[numV, img]
                    self.weightImages = tf.reshape(flat_weightImages, [self.numV, self.imageShape[1], self.imageShape[2], self.imageShape[3]])
                else:
                    self.weightImages = tf.squeeze(tf.transpose(self.V1_W, [3, 0, 1, 2]))

                #For log of activities
                self.log_V1_A = tf.log(tf.abs(self.V1_A)+1e-13)

        #Summaries
        self.s_loss    = tf.summary.scalar('loss', self.loss)
        self.s_recon   = tf.summary.scalar('recon error', self.reconError)
        self.s_errorStd= tf.summary.scalar('errorStd', self.errorStd)
        self.s_l1      = tf.summary.scalar('l1_sparsity', self.l1Sparsity)
        self.s_l1_mean = tf.summary.scalar('l1_mean', self.l1_mean)
        self.s_s_nnz   = tf.summary.scalar('nnz', self.nnz)

        self.h_input    = tf.summary.histogram('input', self.inputImage)
        self.h_input    = tf.summary.histogram('scale_input', self.scaled_inputImage)
        self.h_recon    = tf.summary.histogram('recon', self.recon)
        self.h_v1_w     = tf.summary.histogram('V1_W', self.V1_W)
        self.h_v1_u     = tf.summary.histogram('V1_U', self.V1_U)
        self.h_v1_a     = tf.summary.histogram('V1_A', self.V1_A)
        self.h_log_v1_a = tf.summary.histogram('Log_V1_A', self.log_V1_A)

        #self.h_normVals = tf.histogram_summary('normVals', self.normVals, name="normVals")

    def encodeImage(self, feedDict):
        progress_time = time.time()
        #Reset u
        self.sess.run(self.v1Reset)
        for i in range(self.displayPeriod):
            #Run optimizer
            #This calculates A
            self.sess.run(self.optimizerA0, feed_dict=feedDict)
            #This updates U based on loss function wrt A
            self.sess.run(self.optimizerA, feed_dict=feedDict)
            self.timestep+=1
            if(self.timestep%self.writeStep == 0):
                summary = self.sess.run(self.mergedSummary, feed_dict=feedDict)
                self.train_writer.add_summary(summary, self.timestep)
            if(self.timestep%self.progress == 0):
                tmp_time = time.time()
                print("Timestep ", self.timestep, ":", float(self.progress)/(tmp_time - progress_time), " iterations per second")
                progress_time = tmp_time
            if(self.timestep%self.plotReconPeriod == 0):
                self.plotRecon()
            if(self.timestep%self.plotWeightPeriod == 0):
                self.plotWeight()
        return self.sess.run(self.V1_A)

    #Trains model for numSteps
    def trainA(self, save):
        #Define session
        feedDict = {self.inputImage: self.currImg, self.inputMask: self.currMask}
        self.encodeImage(feedDict)

        if(save):
            save_path = self.saver.save(self.sess, self.saveFile, global_step=self.timestep, write_meta_graph=False)
            print("Model saved in file: %s" % save_path)

    def normWeights(self):
        #Normalize weights
        self.sess.run(self.normalize_W)

    def plotRecon(self):
        #Visualization
        #if (self.plotTimestep % self.plotPeriod == 0):

        #Make directory for timestep
        outPlotDir = self.plotDir+"/"+str(self.timestep)+"/"
        if not os.path.exists(outPlotDir):
           os.makedirs(outPlotDir)

        np_inputImage = self.currImg
        feedDict = {self.inputImage: self.currImg, self.inputMask:self.currMask}
        [np_recon, np_unscaled_recon] = np.squeeze(self.sess.run([self.recon, self.unscaled_recon], feed_dict=feedDict))

        #Draw recons
        plotStr = outPlotDir + "recon_"
        if(np_recon.ndim == 3):
            [rescaled_inputImage, orig_inputImage, mask] = np.squeeze(self.sess.run([self.scaled_inputImage, self.inputImage, self.inputMask], feed_dict=feedDict))
            numRecon = np.minimum(self.batchSize, 4)
            plotRecon1d(np_recon, rescaled_inputImage, plotStr, r=range(numRecon), unscaled_img_matrix=orig_inputImage, unscaled_recon_matrix=np_unscaled_recon, mask_matrix=mask, groups=self.plot_groups, group_title=self.plot_group_title)
        else:
            plotRecon(np_recon, np_inputImage, plotStr, r=range(4))

    def plotWeight(self):
        #Make directory for timestep
        outPlotDir = self.plotDir+"/"+str(self.timestep)+"/"
        if not os.path.exists(outPlotDir):
           os.makedirs(outPlotDir)

        np_V1_W = self.sess.run(self.weightImages)
        np_V1_A = self.sess.run(self.V1_A)

        #plot_weights(rescaled_V1_W, self.plotDir+"dict_"+str(self.timestep), activity=np_V1_A)

        plotStr = outPlotDir + "dict_"
        if(np_V1_W.ndim == 3):
            plot_1d_weights(np_V1_W, plotStr, activity=np_V1_A, sepFeatures=True)
        else:
            plot_weights(np_V1_W, plotStr)

    def trainW(self):
        feedDict = {self.inputImage: self.currImg, self.inputMask: self.currMask}
        #Update weights
        self.sess.run(self.optimizerW, feed_dict=feedDict)
        #New image
        (self.currImg, self.currMask) = self.dataObj.getData(self.batchSize)


    #Finds sparse encoding of inData
    #inData must be in the shape of the image
    #[batch, nY, nX, nF]
    def evalData(self, inData):
        (nb, ny, nx, nf) = inData.shape
        #Check size
        assert(nb == self.batchSize)
        assert(ny == self.imageShape[1])
        assert(nx == self.imageShape[2])
        assert(nf == self.imageShape[3])

        feedDict = {self.inputImage: inData}
        self.encodeImage(feedDict)
        #Get thresholded v1 as an output
        outVals = self.V1_A.eval(session=self.sess)
        return outVals

    #def evalSet(self, evalDataObj, outFilename):
    #    numImages = evalDataObj.numImages
    #    #skip must be 1 for now
    #    assert(evalDataObj.skip == 1)
    #    numIterations = int(np.ceil(float(numImages)/self.batchSize))

    #    pvFile = pvpOpen(outFilename, 'w')
    #    for it in range(numIterations):
    #        print(str((float(it)*100)/numIterations) + "% done (" + str(it) + " out of " + str(numIterations) + ")")
    #        #Evaluate
    #        npV1_A = self.evalData(self.currImg)
    #        v1Sparse = convertToSparse4d(npV1_A)
    #        time = range(it*self.batchSize, (it+1)*self.batchSize)
    #        data = {"values":v1Sparse, "time":time}
    #        pvFile.write(data, shape=(self.VShape[1], self.VShape[2], self.VShape[3]))
    #        self.currImg = self.dataObj.getData(self.batchSize)
    #    pvFile.close()

    #def writePvpWeights(self, outputPrefix, rect=False):
    #    npw = self.sess.run(self.V1_W)
    #    [nyp, nxp, nfp, numK] = npw.shape
    #    filename = outputPrefix + ".pvp"
    #    #We need to get weights into pvp shape
    #    #6D dense numpy array of size [numFrames, numArbors, numKernels, ny, nx, nf]
    #    if(rect):
    #        outWeights = np.zeros((1, 1, numK*2, nyp, nxp, nfp))
    #    else:
    #        outWeights = np.zeros((1, 1, numK, nyp, nxp, nfp))
    #    weightBuf = np.transpose(npw, [3, 0, 1, 2])
    #    outWeights[0, 0, 0:numK, :, :, :] = weightBuf
    #    if(rect):
    #        outWeights[0, 0, numK:2*numK, :, :, :] = weightBuf * -1
    #    pvp = {}
    #    pvp['values'] = outWeights
    #    pvp['time'] = np.array([0])
    #    pv.writepvpfile(filename, pvp)

