import pdb
import numpy as np
import tensorflow as tf
from lca_adam import LCA_ADAM
from plots.plotWeights import plot_weights
from plots.plotRecon import plotRecon
from plots.plotFeaturemaps import plotFeaturemaps
from .utils import *
#Using pvp files for saving
from pvtools import *

class LCA_ADAM_time(LCA_ADAM):
    def makeDirs(self):
        super(LCA_ADAM_time, self).makeDirs()
        makeDir(self.reconDir)
        makeDir(self.weightDir)
        makeDir(self.featureMapDir)


    #Sets dictionary of params to member variables
    def loadParams(self, params):
        super(LCA_ADAM_time, self).loadParams(params)
        self.VStrideT = params['VStrideT']
        self.patchSizeT = params['patchSizeT']
        self.stereo = params['stereo']
        self.plotInd = params['plotInd']
        self.plotFM = params['plotFM']

        self.reconDir = self.plotDir + "/recon/"
        self.weightDir = self.plotDir + "/weight/"
        self.featureMapDir = self.plotDir + "/featuremap/"

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
        super(LCA_ADAM_time, self).__init__(params, dataObj)

    #Builds the model. inMatFilename should be the vgg file
    def buildModel(self, inputShape):
        #inputShape goes (time, y, x, f)
        #However, time enompases stereo as well
        assert(inputShape[0]/2 % self.VStrideT == 0)
        assert(inputShape[1] % self.VStrideY == 0)
        assert(inputShape[2] % self.VStrideX == 0)
        V_T = int(inputShape[0]/(2*self.VStrideT))
        V_Y = int(inputShape[1]/self.VStrideY)
        V_X = int(inputShape[2]/self.VStrideX)

        if(self.stereo):
            numTime = inputShape[0]/2
            numFeatures = inputShape[3]*2
        else:
            numTime = inputShape[0]
            numFeatures = inputShape[3]

        self.imageShape = (self.batchSize, numTime, inputShape[1], inputShape[2], numFeatures)
        self.WShape = (self.patchSizeT, self.patchSizeY, self.patchSizeX, numFeatures, self.numV)

        if(numTime == 1):
            self.VShape = (self.batchSize, 1, V_Y, V_X, self.numV)
        else:
            self.VShape = (self.batchSize, 2, V_Y, V_X, self.numV)

        #Running on GPU
        with tf.device(self.device):
            with tf.name_scope("inputOps"):
                #Get convolution variables as placeholders
                self.inputImage = node_variable((self.batchSize,) + inputShape, "inputImage")
                if(self.stereo):
                    #We split the time dimension to stereo and concatenate with feature dim
                    self.reshapeImage = tf.reshape(self.inputImage,
                            [self.batchSize, inputShape[0]/2, 2, inputShape[1], inputShape[2], inputShape[3]])
                    self.permuteImage = tf.transpose(self.reshapeImage, [0, 1, 3, 4, 5, 2])
                    self.outImage = tf.reshape(self.permuteImage,
                            [self.batchSize, inputShape[0]/2, inputShape[1], inputShape[2], inputShape[3]*2])
                else:
                    self.outImage = self.inputImage

                self.padInput = tf.pad(self.outImage, [[0, 0], [0, 0], [7, 7], [15, 15], [0, 0]])
                #Scale inputImage
                self.scaled_inputImage = self.padInput/np.sqrt(self.patchSizeX*self.patchSizeY*numFeatures)

            with tf.name_scope("Dictionary"):
                self.V1_W = sparse_weight_variable(self.WShape, "V1_W")

            with tf.name_scope("weightNorm"):
                self.normVals = tf.sqrt(tf.reduce_sum(tf.square(self.V1_W), reduction_indices=[0, 1, 2, 3], keep_dims=True))
                self.normalize_W = self.V1_W.assign(self.V1_W/(self.normVals + 1e-8))

            with tf.name_scope("LCA_ADAM"):
                self.V1_U = uniform_weight_variable(self.VShape, "V1_U", 0.0, 1.25*self.thresh)
                self.V1_A = weight_variable(self.VShape, "V1_A", 1e-3)

            with tf.name_scope("Recon"):
                assert(self.VStrideY >= 1)
                assert(self.VStrideX >= 1)
                outputShape = [self.imageShape[0], self.imageShape[1], self.imageShape[2]+14, self.imageShape[3]+30, self.imageShape[4]]
                #We build index tensor in numpy to gather
                self.recon = tf.nn.conv3d_transpose(self.V1_A, self.V1_W, outputShape, [1, self.VStrideT, self.VStrideY, self.VStrideX, 1], 'VALID', 'recon')

            with tf.name_scope("Error"):
                self.error = self.scaled_inputImage - self.recon

            with tf.name_scope("Loss"):
                self.reconError = tf.reduce_mean(tf.reduce_sum(tf.square(self.error), reduction_indices=[1, 2, 3, 4]))
                self.l1Sparsity = tf.reduce_mean(tf.reduce_sum(tf.abs(self.V1_A), reduction_indices=[1, 2, 3, 4]))
                #self.reconError = tf.reduce_mean(tf.square(self.error))
                #self.l1Sparsity = tf.reduce_mean(tf.abs(self.V1_A))
                #Define loss
                self.loss = self.reconError/2 + self.thresh * self.l1Sparsity

            with tf.name_scope("Opt"):
                #Calculate A from U
                self.optimizerA0 = self.V1_A.assign(tf.nn.relu(self.V1_U - self.thresh))

                self.optimizerA1 = tf.train.AdamOptimizer(self.learningRateA)

                #Find gradient wrt A
                self.lossGrad = self.optimizerA1.compute_gradients(self.reconError, [self.V1_A])
                #Apply such gradient to U
                self.dU = [(self.lossGrad[0][0] - self.V1_A + self.V1_U, self.V1_U)];

                self.optimizerA = self.optimizerA1.apply_gradients(self.dU)

                self.optimizerW = tf.train.AdadeltaOptimizer(self.learningRateW, epsilon=1e-6).minimize(self.loss,
                        var_list=[
                            self.V1_W
                        ])

            with tf.name_scope("stats"):
                self.nnz = tf.reduce_mean(tf.cast(tf.not_equal(self.V1_A, 0), tf.float32))

                self.errorStd = tf.sqrt(tf.reduce_mean(tf.square(self.error-tf.reduce_mean(self.error))))*np.sqrt(self.WShape[0]*self.WShape[1]*self.WShape[2]*self.WShape[3])
                self.l1_mean = tf.reduce_mean(tf.abs(self.V1_A))

                self.weightImages = tf.transpose(self.V1_W, [4, 0, 1, 2, 3])

                #For log of activities
                self.log_V1_A = tf.log(tf.abs(self.V1_A)+1e-15)

        #Summaries
        self.s_loss = tf.summary.scalar('loss', self.loss)
        self.s_recon = tf.summary.scalar('recon error', self.reconError)
        self.s_errorStd= tf.summary.scalar('errorStd', self.errorStd)
        self.s_l1= tf.summary.scalar('l1 sparsity', self.l1Sparsity)
        self.s_l1_mean = tf.summary.scalar('l1 mean', self.l1_mean)
        self.s_s_nnz = tf.summary.scalar('nnz', self.nnz )

        self.h_input = tf.summary.histogram('input', self.inputImage)
        self.h_recon = tf.summary.histogram('recon', self.recon)
        self.h_v1_w = tf.summary.histogram('V1_W', self.V1_W)

        self.h_v1_u = tf.summary.histogram('V1_U', self.V1_U)
        self.h_v1_a = tf.summary.histogram('V1_A', self.V1_A)
        self.h_log_v1_a = tf.summary.histogram('Log_V1_A', self.log_V1_A)

        self.h_normVals = tf.summary.histogram('normVals', self.normVals)

    def evalAndPlotFeaturemaps(self, feedDict, prefix):
        print "Plotting featuremaps"
        np_v1 = self.sess.run(self.V1_A, feed_dict=feedDict)
        np_inputImage = self.currImg
        #We only care about the left last frame
        #TODO make the indices general
        np_v1 = np_v1[:, 1, :, :, :]
        if(self.stereo):
            np_inputImage = np_inputImage[:, 4, :, :, :]
        else:
            np_inputImage = np_inputImage[:, 2, :, :, :]

        plotFeaturemaps(np_v1, np_inputImage, prefix, r=[2])
        #plotFeaturemaps(np_v1, np_inputImage, prefix, r=[1])

    def evalAndPlotWeights(self, feedDict, prefix):
        print "Plotting weights"
        np_weights = self.sess.run(self.V1_W, feed_dict=feedDict)
        np_v1 = self.sess.run(self.V1_A, feed_dict=feedDict)
        (ntime, ny, nx, nfns, nf) = np_weights.shape
        if(self.stereo):
            np_weights_reshape = np.reshape(np_weights, (ntime, ny, nx, nfns/2, 2, nf))
            for s in range(2):
                if(s == 0):
                    suffix = "_left"
                elif(s == 1):
                    suffix = "_right"
                for t in range(ntime):
                    plotWeights = np_weights_reshape[t, :, :, :, s, :]
                    plot_weights(plotWeights, prefix, suffix + "_time" + str(t), [3, 0, 1, 2], np_v1, plotInd = self.plotInd)
        else:
            filename = prefix
            for t in range(ntime):
                plotWeights = np_weights[t, :, :, :, :]
                plot_weights(plotWeights, prefix, "_time" + str(t), [3, 0, 1, 2], np_v1, plotInd = self.plotInd)

    def evalAndPlotRecons(self, feedDict, prefix):
        print "Plotting recons"
        np_recon = self.sess.run(self.recon, feed_dict=feedDict)
        np_inputImage = self.sess.run(self.padInput, feed_dict=feedDict)
        (batch, ntime, ny, nx, nf) = np_recon.shape
        #Shape recon and image into (batch, time, ny, nx, nfcolor, stereo)

        if(self.stereo):
            np_recon = np.reshape(np_recon, [batch, ntime, ny, nx, nf/2, 2])
            np_inputImage = np.reshape(np_inputImage, [batch, ntime, ny, nx, nf/2, 2])
            for s in range(2):
                filename = prefix
                if(s == 0):
                    filename += "_left"
                elif(s == 1):
                    filename += "_right"
                for t in range(ntime):
                    recon = np_recon[:, t, :, :, :, s]
                    image = np_inputImage[:, t, :, :, :, s]

                    plotRecon(recon, image, filename + "_recon_time" + str(t) + "_batch")
        else:
            filename = prefix
            for t in range(ntime):
                recon = np_recon[:, t, :, :, :]
                image = np_inputImage[:, t, :, :, :]
                plotRecon(recon, image, filename + "_recon_time" + str(t) + "_batch")

    def trainW(self):
        feedDict = {self.inputImage: self.currImg}

        #Visualization
        if (self.plotTimestep % self.plotPeriod == 0):
            #np_V1_W = self.sess.run(self.weightImages)
            #pdb.set_trace()
            #(numV, time, ny, nx, nf) = V1_W.shape
            #assert(nf == 6)
            #np_reshape = np.reshape(V1_W, [numV, time, ny, nx, 3, 2])

            #plot_weights_time(np_V1_W[:, :, :, :, :, 0], self.plotDir+"leftDict_"+str(self.timestep)+".png")
            #plot_weights_time(np_V1_W[:, :, :, :, :, 1], self.plotDir+"rightDict_"+str(self.timestep)+".png")
            #Draw recons
            #np_inputImage = self.currImg
            #np_recon = self.sess.run(self.recon, feed_dict=feedDict)

            #plotRecon(np_recon, np_inputImage, self.plotDir+"recon_"+str(self.timestep), r=range(4))
            suffix = "/train_" + str(self.timestep)
            self.evalAndPlotRecons(feedDict, self.reconDir + suffix)
            if(self.plotFM):
                self.evalAndPlotFeaturemaps(feedDict, self.featureMapDir + suffix)
            self.evalAndPlotWeights(feedDict, self.weightDir + suffix)


        #Update weights
        self.sess.run(self.optimizerW, feed_dict=feedDict)
        #New image
        self.currImg = self.dataObj.getData(self.batchSize)[0]
        self.plotTimestep += 1

    def evalSet(self, evalDataObj, outPrefix):
        numImages = evalDataObj.numImages
        #skip must be 1 for now
        assert(evalDataObj.skip == 1)
        numIterations = int(np.ceil(float(numImages)/self.batchSize))

        (batchSize, nt, ny, nx, nf) = self.VShape

        if(self.stereo):
            numTime = nt*2
        else:
            numTime = nt

        pvpFileList = []
        for t in range(numTime):
            suffix = ""
            if self.stereo:
                #Time idx
                time = int(np.floor(t/2))
                suffix += "_time_"+str(time)
                #left eye
                if t%2 == 0:
                    suffix += "_left.pvp"
                else:
                    suffix += "_right.pvp"
            else:
                suffix += "_time_"+str(t)+".pvp"
            pvpFileList.append(pvpOpen(outPrefix+suffix, 'w'))

        for it in range(numIterations):
            print str((float(it)*100)/numIterations) + "% done (" + str(it) + " out of " + str(numIterations) + ")"
            #Evaluate
            npV1_A = self.evalData(self.currImg)
            if(self.stereo):
                numN = np.floor(nf/2)
                reshape_v1 = np.reshape(npV1_A, (batchSize, nt, ny, nx, numN, 2))
                permute_v1 = np.transpose(reshape_v1, (0, 1, 5, 2, 3, 4))
                out_v1 = np.reshape(permute_v1, (batchSize, numTime, ny, nx, numN))
            else:
                numN = nf
                out_v1 = npV1_A

            for t in range(numTime):
                singleV1 = out_v1[:, t, :, :, :]
                v1Sparse = convertToSparse4d(singleV1)
                time = range(it*self.batchSize, (it+1)*self.batchSize)
                data = {"values":v1Sparse, "time":time}
                pvpFileList[t].write(data, shape=(ny, nx, numN))

            data = self.dataObj.getData(self.batchSize)
            self.currImg = data[0]

        for t in range(numTime):
            pvpFileList[t].close()

    def writePvpWeights(self, outputPrefix, rect=False):
        npw = self.sess.run(self.V1_W)
        [nyp, nxp, nfp, numK] = npw.shape
        filename = outputPrefix + ".pvp"
        #We need to get weights into pvp shape
        #6D dense numpy array of size [numFrames, numArbors, numKernels, ny, nx, nf]
        if(rect):
            outWeights = np.zeros((1, 1, numK*2, nyp, nxp, nfp))
        else:
            outWeights = np.zeros((1, 1, numK, nyp, nxp, nfp))
        weightBuf = np.transpose(npw, [3, 0, 1, 2])
        outWeights[0, 0, 0:numK, :, :, :] = weightBuf
        if(rect):
            outWeights[0, 0, numK:2*numK, :, :, :] = weightBuf * -1
        pvp = {}
        pvp['values'] = outWeights
        pvp['time'] = np.array([0])
        writepvpfile(filename, pvp)

