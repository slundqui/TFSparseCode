import pdb
import numpy as np
import tensorflow as tf
from base import base
from plots.plotWeights import plot_weights
from plots.plotRecon import plotRecon
from .utils import sparse_weight_variable, weight_variable, node_variable, conv2d, conv2d_oneToMany, convertToSparse4d, save_sparse_csr

class ISTA(base):
    #Sets dictionary of params to member variables
    def loadParams(self, params):
        super(ISTA, self).loadParams(params)
        self.learningRateA = params['learningRateA']
        self.learningRateW = params['learningRateW']
        self.thresh = params['thresh']
        self.numV = params['numV']
        self.VStrideY = params['VStrideY']
        self.VStrideX = params['VStrideX']
        self.patchSizeY = params['patchSizeY']
        self.patchSizeX = params['patchSizeX']
        self.epsilon = params['epsilon']

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

    #def getLoadVars(self):
    #    return tf.all_variables()

    #Constructor takes inputShape, which is a 3 tuple (ny, nx, nf) based on the size of the image being fed in
    def __init__(self, params, dataObj):
        super(ISTA, self).__init__(params, dataObj)
        self.currImg = self.dataObj.getData(self.batchSize)

    #Builds the model. inMatFilename should be the vgg file
    def buildModel(self, inputShape):
        assert(inputShape[0] % self.VStrideY == 0)
        assert(inputShape[1] % self.VStrideX == 0)
        V_Y = int(inputShape[0]/self.VStrideY)
        V_X = int(inputShape[1]/self.VStrideX)
        self.imageShape = (self.batchSize, inputShape[0], inputShape[1], inputShape[2])
        self.WShape = (self.patchSizeY, self.patchSizeX, 3, self.numV)
        self.VShape = (self.batchSize, V_Y, V_X, self.numV)

        #Running on GPU
        with tf.device(self.device):
            with tf.name_scope("inputOps"):
                #Get convolution variables as placeholders
                self.inputImage = node_variable(self.imageShape, "inputImage")
                #Scale inputImage
                self.scaled_inputImage = self.inputImage/np.sqrt(self.patchSizeX*self.patchSizeY*inputShape[2])

            with tf.name_scope("Dictionary"):
                self.V1_W = weight_variable(self.WShape, "V1_W", 1e-3)

            with tf.name_scope("weightNorm"):
                self.normVals = tf.sqrt(tf.reduce_sum(tf.square(self.V1_W), reduction_indices=[0, 1, 2], keep_dims=True))
                self.normalize_W = self.V1_W.assign(self.V1_W/(self.normVals + 1e-8))

            with tf.name_scope("ISTA"):
                #Soft threshold
                self.V1_A = weight_variable(self.VShape, "V1_A", 1e-3)
                #Reinitializer
                self.randV1 = tf.truncated_normal(self.VShape, mean=0, stddev=1e-3)
                self.initV1 = self.V1_A.assign(self.randV1)


            with tf.name_scope("Recon"):
                assert(self.VStrideY >= 1)
                assert(self.VStrideX >= 1)
                #We build index tensor in numpy to gather
                self.recon = conv2d_oneToMany(self.V1_A, self.V1_W, self.imageShape, "recon", self.VStrideY, self.VStrideX)

            with tf.name_scope("Error"):
                self.error = self.scaled_inputImage - self.recon

            with tf.name_scope("Loss"):
                self.reconError = tf.reduce_mean(tf.reduce_sum(tf.square(self.error), reduction_indices=[1, 2, 3]))
                self.l1Sparsity = tf.reduce_mean(tf.reduce_sum(tf.abs(self.V1_A), reduction_indices=[1, 2, 3]))
                #Define loss
                self.loss = self.reconError/2 + self.thresh * self.l1Sparsity

            with tf.name_scope("Opt"):
                ##Define optimizer
                ##self.optimizerA = tf.train.GradientDescentOptimizer(self.learningRateA).minimize(self.loss,
                #self.optimizerA = tf.train.AdamOptimizer(self.learningRateA).minimize(self.loss,
                #        var_list=[
                #            self.V1_A
                #        ])
                self.reconGrad = self.learningRateA * tf.gradients(self.reconError, [self.V1_A])[0]
                #We add epslon to avoid taking sign of 0 if v1_a is 0
                self.newA = tf.nn.relu(tf.abs(self.V1_A - self.reconGrad) - self.thresh*self.learningRateA) * tf.sign(self.V1_A)
                self.optimizerA = self.V1_A.assign(self.newA)

                self.optimizerW = tf.train.AdadeltaOptimizer(self.learningRateW).minimize(self.loss,
                        var_list=[
                            self.V1_W
                        ])

            with tf.name_scope("stats"):
                self.nnz = tf.reduce_mean(tf.cast(tf.not_equal(self.V1_A, 0), tf.float32))

                self.errorStd = tf.sqrt(tf.reduce_mean(tf.square(self.error-tf.reduce_mean(self.error))))*np.sqrt(self.patchSizeY*self.patchSizeX*inputShape[2])
                self.l1_mean = tf.reduce_mean(tf.abs(self.V1_A))

                self.weightImages = tf.transpose(self.V1_W, [3, 0, 1, 2])

                #For log of activities
                self.log_V1_A = tf.log(tf.abs(self.V1_A)+1e-15)

        #Summaries
        self.s_loss = tf.scalar_summary('loss', self.loss, name="lossSum")
        self.s_recon = tf.scalar_summary('recon error', self.reconError, name="reconError")
        self.s_errorStd= tf.scalar_summary('errorStd', self.errorStd, name="errorStd")
        self.s_l1= tf.scalar_summary('l1 sparsity', self.l1Sparsity, name="l1Sparsity")
        self.s_l1_mean = tf.scalar_summary('l1 mean', self.l1_mean, name="l1Mean")
        self.s_s_nnz = tf.scalar_summary('nnz', self.nnz, name="nnz")

        self.h_input = tf.histogram_summary('input', self.inputImage, name="input")
        self.h_recon = tf.histogram_summary('recon', self.recon, name="recon")
        self.h_v1_w = tf.histogram_summary('V1_W', self.V1_W, name="V1_W")

        self.h_v1_a = tf.histogram_summary('V1_A', self.V1_A, name="V1_A")
        self.h_log_v1_a = tf.histogram_summary('Log_V1_A', self.log_V1_A, name="Log_V1_A")

        self.h_normVals = tf.histogram_summary('normVals', self.normVals, name="normVals")

    #Trains model for numSteps
    def trainA(self, save):
        #Define session
        feedDict = {self.inputImage: self.currImg}

        #Reinitialize v1
        self.sess.run(self.initV1)

        for i in range(self.displayPeriod):
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
        feedDict = {self.inputImage: self.currImg}

        #Visualization
        if (self.plotTimestep % self.plotPeriod == 0):
            np_V1_W = self.sess.run(self.weightImages)
            plot_weights(np_V1_W, self.plotDir+"dict_"+str(self.timestep)+".png")
            #Draw recons
            np_inputImage = self.currImg
            np_recon = self.sess.run(self.recon, feed_dict=feedDict)
            plotRecon(np_recon, np_inputImage, self.plotDir+"recon_"+str(self.timestep), r=range(4))

        #Update weights
        self.sess.run(self.optimizerW, feed_dict=feedDict)
        #New image
        self.currImg = self.dataObj.getData(self.batchSize)
        self.plotTimestep += 1


    #Finds sparse encoding of inData
    #inData must be in the shape of the image
    #[batch, nY, nX, nF]
    def evalData(self, inData, displayPeriod=None):
        (nb, ny, nx, nf) = inData.shape
        #Check size
        assert(nb == self.batchSize)
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
                print "Timestep ", str(i) , " out of ", str(self.displayPeriod)
        #Get thresholded v1 as an output
        outVals = self.V1_A.eval(session=self.sess)
        return outVals

    def evalSet(self, evalDataObj, outPrefix, displayPeriod=None):
        numImages = evalDataObj.numImages
        #skip must be 1 for now
        assert(evalDataObj.skip == 1)
        numIterations = int(np.ceil(float(numImages)/self.batchSize))

        for it in range(numIterations):
            print str((float(it)*100)/numIterations) + "% done (" + str(it) + " out of " + str(numIterations) + ")"
            #Evaluate
            npV1_A = self.evalData(self.currImg, displayPeriod=displayPeriod)
            for b in range(self.batchSize):
                frameIdx = it*self.batchSize + b
                if(frameIdx < numImages):
                    v1Sparse = convertToSparse4d(np.expand_dims(npV1_A[b, :, :, :], 0))
                    save_sparse_csr(outPrefix+str(frameIdx), v1Sparse)
            self.currImg = self.dataObj.getData(self.batchSize)

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
    #    writepvpfile(filename, pvp)

