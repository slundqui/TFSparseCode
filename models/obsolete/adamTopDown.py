import pdb
import numpy as np
import tensorflow as tf
from base import base
from plots.plotWeights import plot_weights
from plots.plotRecon import plotRecon
from .utils import sparse_weight_variable, weight_variable, node_variable, conv2d, conv2d_oneToMany, convertToSparse4d, save_sparse_csr
#Using pvp files for saving
from pvtools import *

class AdamTopDown(base):
    #Sets dictionary of params to member variables
    def loadParams(self, params):
        super(AdamTopDown, self).loadParams(params)
        self.learningRateA = params['learningRateA']
        self.learningRateW = params['learningRateW']
        self.thresh = params['thresh']
        self.numV = params['numV']
        self.VStrideY = params['VStrideY']
        self.VStrideX = params['VStrideX']
        self.patchSizeY = params['patchSizeY']
        self.patchSizeX = params['patchSizeX']
        self.zeroThresh = params['zeroThresh']
        self.numLayers = params['numLayers']

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
        super(AdamTopDown, self).__init__(params, dataObj)
        self.currImg = self.dataObj.getData(self.batchSize)

    #Builds the model. inMatFilename should be the vgg file
    def buildModel(self, inputShape):

        #Running on GPU
        with tf.device(self.device):
            with tf.name_scope("inputOps"):
                self.imageShape = (self.batchSize, inputShape[0], inputShape[1], inputShape[2])
                #Get convolution variables as placeholders
                self.inputImage = node_variable(self.imageShape, "inputImage")

            self.V1_W = []
            self.normalize_W = []
            self.V1_A = []
            self.recon = []
            self.error = []
            self.reconError = []
            self.sparseError = []
            self.scaledInput = []

            self.t_V1_A = []
            self.t_recon = []
            self.t_error=[]
            self.t_reconError = []
            self.t_sparseError = []

            self.underThresh = []
            self.errorStd = []
            self.l1_mean = []
            self.t_errorStd = []
            self.t_l1_mean = []
            self.log_V1_A = []

            self.WShape = []
            self.VShape = []
            self.inShape = []

            for l in range(self.numLayers):
                if l == 0:
                    numInF = inputShape[2]
                else:
                    numInF = self.numV[l-1]

                V_Y = float(inputShape[0])
                V_X = float(inputShape[1])

                for i in range(l+1):
                    V_Y_Prev = V_Y
                    V_X_Prev = V_X
                    assert(int(V_Y) % self.VStrideY[i] == 0)
                    assert(int(V_X) % self.VStrideX[i] == 0)
                    V_Y = V_Y/self.VStrideY[i]
                    V_X = V_X/self.VStrideX[i]

                V_Y = int(V_Y)
                V_Y_Prev = int(V_Y_Prev)
                V_X = int(V_X)
                V_X_Prev = int(V_X_Prev)

                self.WShape.append((self.patchSizeY[l], self.patchSizeX[l], numInF, self.numV[l]))
                self.VShape.append((self.batchSize, V_Y, V_X, self.numV[l]))
                self.inShape.append((self.batchSize, V_Y_Prev, V_X_Prev, numInF))

                with tf.name_scope("Dictionary"):
                    self.V1_W.append(sparse_weight_variable(self.WShape[l], "V1_W"+str(l)))

                with tf.name_scope("weightNorm"):
                    self.normVals = tf.sqrt(tf.reduce_sum(tf.square(self.V1_W[l]), reduction_indices=[0, 1, 2], keep_dims=True))
                    self.normalize_W.append(self.V1_W[l].assign(self.V1_W[l]/(self.normVals+1e-8)))

                with tf.name_scope("Encoding"):
                    #Soft threshold
                    self.V1_A.append(weight_variable(self.VShape[l], "V1_A"+str(l), 1e-3))
                    zeroConst = tf.zeros(self.VShape[l])
                    boolUnderThresh = tf.greater(self.zeroThresh[l], tf.abs(self.V1_A[l]))
                    self.t_V1_A.append(tf.select(boolUnderThresh, zeroConst, self.V1_A[l]))

                with tf.name_scope("Recon"):
                    assert(self.VStrideY[l] >= 1)
                    assert(self.VStrideX[l] >= 1)
                    #We build index tensor in numpy to gather
                    self.recon.append(conv2d_oneToMany(self.V1_A[l], self.V1_W[l], self.inShape[l], "recon", self.VStrideY[l], self.VStrideX[l]))
                    self.t_recon.append(conv2d_oneToMany(self.t_V1_A[l], self.V1_W[l], self.inShape[l], "t_recon", self.VStrideY[l], self.VStrideX[l]))

                with tf.name_scope("Error"):
                    #Scale inputImage
                    if(l == 0):
                        #self.scaledInput.append(self.inputImage/np.sqrt(self.patchSizeX[0]*self.patchSizeY[0]*inputShape[2]))
                        self.scaledInput.append(self.inputImage)
                    else:
                        #self.scaledInput.append(self.V1_A[l-1]/np.sqrt(self.patchSizeX[l]*self.patchSizeY[l]*self.numV[l-1]))
                        self.scaledInput.append(self.V1_A[l-1])
                    self.error.append(self.scaledInput[l] - self.recon[l])
                    self.t_error.append(self.scaledInput[l] - self.t_recon[l])

                with tf.name_scope("Loss"):
                    #Sum across axis except for match
                    self.reconError.append(tf.reduce_mean(tf.reduce_sum(tf.square(self.error[l]), reduction_indices=[1, 2, 3])))
                    self.sparseError.append(tf.reduce_mean(tf.reduce_sum(tf.abs(self.V1_A[l]), reduction_indices=[1, 2, 3])))

                    self.t_reconError.append(tf.reduce_mean(tf.reduce_sum(tf.square(self.t_error[l]), reduction_indices=[1, 2, 3])))
                    self.t_sparseError.append(tf.reduce_mean(tf.reduce_sum(tf.abs(self.t_V1_A[l]), reduction_indices=[1, 2, 3])))

                with tf.name_scope("stats"):
                    self.underThresh.append(tf.reduce_mean(tf.cast(tf.abs(self.V1_A[l]) > self.zeroThresh[l], tf.float32)))


                    eStd = tf.sqrt(tf.reduce_mean(tf.square(self.error[l] - tf.reduce_mean(self.error[l]))))
                    t_eStd = tf.sqrt(tf.reduce_mean(tf.square(self.t_error[l] - tf.reduce_mean(self.t_error[l]))))
                    inStd = tf.sqrt(tf.reduce_mean(tf.square(self.scaledInput[l] - tf.reduce_mean(self.scaledInput[l]))))

                    self.errorStd.append(eStd/inStd)
                    self.t_errorStd.append(t_eStd/inStd)

                    self.l1_mean.append(tf.reduce_mean(tf.abs(self.V1_A[l])))
                    self.t_l1_mean.append(tf.reduce_mean(tf.abs(self.t_V1_A[l])))

                    #For log of activities
                    self.log_V1_A.append(tf.log(tf.abs(self.V1_A[l])+1e-15))

            with tf.name_scope("Loss"):
                #Define loss
                self.loss = self.reconError[0]/2 + self.thresh[0] * self.sparseError[0]
                self.t_loss = self.t_reconError[0]/2 + self.thresh[0] * self.t_sparseError[0]
                for l in range(1, self.numLayers):
                    self.loss += self.reconError[l]/2 + self.thresh[l] * self.sparseError[l]
                    self.t_loss += self.t_reconError[l]/2 + self.thresh[l] * self.t_sparseError[l]

            with tf.name_scope("Opt"):
                #Define optimizer
                #self.optimizerA = tf.train.GradientDescentOptimizer(self.learningRateA).minimize(self.loss,
                self.optimizerA = tf.train.AdamOptimizer(self.learningRateA).minimize(self.loss,
                #self.optimizerA = tf.train.AdadeltaOptimizer(self.learningRateA).minimize(self.loss,
                        var_list = self.V1_A
                        )
                #self.optimizerW = tf.train.AdamOptimizer(self.learningRateW).minimize(self.loss,
                #Minimizing weights with respect to the cutoff weights
                #self.optimizerW = tf.train.AdamOptimizer(self.learningRateW).minimize(self.t_loss,
                self.optimizerW = tf.train.AdadeltaOptimizer(self.learningRateW, epsilon=1e-6).minimize(self.loss,
                        var_list= self.V1_W
                        )

        with tf.name_scope("ReconVis"):
            self.visRecon = []
            self.t_visRecon = []
            for l in range(self.numLayers):
                outRecon = self.recon[l]
                t_outRecon = self.t_recon[l]
                for ll in range(l)[::-1]:
                    #We prob recons down layers
                    outRecon = conv2d_oneToMany(outRecon, self.V1_W[ll], self.inShape[ll], "recon_"+str(l)+"_"+str(ll), self.VStrideY[ll], self.VStrideX[ll])
                    t_outRecon = conv2d_oneToMany(t_outRecon, self.V1_W[ll], self.inShape[ll], "recon_"+str(l)+"_"+str(ll), self.VStrideY[ll], self.VStrideX[ll])
                self.visRecon.append(outRecon)
                self.t_visRecon.append(t_outRecon)

        with tf.name_scope("WeightVis"):
            self.visWeight = []

            for l in range(self.numLayers):
                outWeight = tf.transpose(self.V1_W[l], [3, 0, 1, 2])
                numN = self.WShape[l][3]
                numY = self.WShape[l][0]
                numX = self.WShape[l][1]
                numF = self.WShape[l][2]

                for ll in range(l)[::-1]:
                    numY = self.WShape[ll][0] + (numY-1) * self.VStrideY[ll]
                    numX = self.WShape[ll][1] + (numX-1) * self.VStrideX[ll]
                    numF = self.WShape[ll][2]
                    inShape = (numN, numY, numX, numF)
                    outWeight = conv2d_oneToMany(outWeight, self.V1_W[ll], inShape, "weight_"+str(l)+"_"+str(ll), self.VStrideY[ll], self.VStrideX[ll], padding="VALID")

                self.visWeight.append(outWeight)

        #Summaries
        self.s_loss = tf.scalar_summary('loss', self.loss, name="lossSum")
        self.s_t_loss = tf.scalar_summary('t loss' + str(l), self.t_loss, name="t_lossSum")
        self.h_input = tf.histogram_summary('inputImage', self.inputImage, name="input")

        for l in range(self.numLayers):
            self.s_recon = tf.scalar_summary('recon error' + str(l), self.reconError[l], name="reconError")
            self.s_errorStd= tf.scalar_summary('errorStd' + str(l), self.errorStd[l], name="errorStd")
            self.s_l1= tf.scalar_summary('l1 sparsity' + str(l), self.sparseError[l], name="sparseError")
            self.s_l1_mean = tf.scalar_summary('l1 mean' + str(l), self.l1_mean[l], name="l1Mean")
            self.s_s_nnz = tf.scalar_summary('nnz' + str(l), self.underThresh[l], name="nnz")

            self.s_t_recon = tf.scalar_summary('t recon error' + str(l), self.t_reconError[l], name="t_reconError")
            self.s_t_errorStd= tf.scalar_summary('t errorStd' + str(l), self.t_errorStd[l], name="t_errorStd")
            self.s_t_l1= tf.scalar_summary('t l1 sparsity' + str(l), self.t_sparseError[l], name="t_sparseError")
            self.s_t_l1_mean = tf.scalar_summary('t l1 mean' + str(l), self.t_l1_mean[l], name="t_l1Mean")

            self.h_input = tf.histogram_summary('scaledInput'+str(l), self.scaledInput[l], name="input")
            self.h_recon = tf.histogram_summary('recon' + str(l), self.recon[l], name="recon")
            self.h_v1_w = tf.histogram_summary('V1_W' + str(l), self.V1_W[l], name="V1_W")
            self.h_v1_a = tf.histogram_summary('V1_A' + str(l), self.V1_A[l], name="V1_A")
            self.h_log_v1_a = tf.histogram_summary('Log_V1_A' + str(l), self.log_V1_A[l], name="Log_V1_A")

    #We only load weights, not v1
    def getLoadVars(self):
        #v = tf.all_variables()
        #out = [var for var in v if ("V1_A" not in var.name)]
        #n = [var.name for var in out]
        #return out
        return tf.all_variables()

    def encodeImage(self, feedDict):
        for i in range(self.displayPeriod):
            #Run optimizer
            self.sess.run(self.optimizerA, feed_dict=feedDict)
            self.timestep+=1
            if((i+1)%self.writeStep == 0):
                summary = self.sess.run(self.mergedSummary, feed_dict=feedDict)
                self.train_writer.add_summary(summary, self.timestep)
            if((i+1)%self.progress == 0):
                print("Timestep ", self.timestep)

    #Trains model for numSteps
    def trainA(self, save):
        #Define session
        feedDict = {self.inputImage: self.currImg}
        self.encodeImage(feedDict)

        if(save):
            save_path = self.saver.save(self.sess, self.saveFile, global_step=self.timestep, write_meta_graph=False)
            print("Model saved in file: %s" % save_path)

    def normWeights(self):
        #Normalize weights
        for l in range(self.numLayers):
            self.sess.run(self.normalize_W[l])

    def trainW(self):
        feedDict = {self.inputImage: self.currImg}

        #Visualization
        if (self.plotTimestep % self.plotPeriod == 0):
            for l in range(self.numLayers):
                np_V1_W = self.sess.run(self.visWeight[l])
                plot_weights(np_V1_W, self.plotDir+"dict_S" + str(l) + "_" +str(self.timestep)+".png")
                #Draw recons
                np_inputImage = self.currImg

                np_recon = self.sess.run(self.visRecon[l], feed_dict=feedDict)
                np_t_recon = self.sess.run(self.t_visRecon[l], feed_dict=feedDict)
                plotRecon(np_recon, np_inputImage, self.plotDir+"recon_S"+str(l)+"_"+str(self.timestep)+".png", r=range(4))
                plotRecon(np_t_recon, np_inputImage, self.plotDir+"t_recon_S"+str(l)+"_"+str(self.timestep)+".png", r=range(4))

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
        self.encodeImage(feedDict)
        #Get thresholded v1 as an output
        outVals = []
        for l in range(self.numLayers):
            outVals.append(self.t_V1_A.eval(session=self.sess))
        return outVals

    def evalSet(self, evalDataObj, outFilename, displayPeriod=None):
        numImages = evalDataObj.numImages
        #skip must be 1 for now
        assert(evalDataObj.skip == 1)
        numIterations = int(np.ceil(float(numImages)/self.batchSize))

        pvFile = pvpOpen(outFilename, 'w')
        for it in range(numIterations):
            print(str((float(it)*100)/numIterations) + "% done (" + str(it) + " out of " + str(numIterations) + ")")
            #Evaluate
            npV1_A = self.evalData(self.currImg, displayPeriod=displayPeriod)
            pdb.set_trace()
            #TODO

            v1Sparse = convertToSparse4d(npV1_A)
            time = range(it*self.batchSize, (it+1)*self.batchSize)
            data = {"values":v1Sparse, "time":time}
            pvFile.write(data, shape=(self.VShape[1], self.VShape[2], self.VShape[3]))
            self.currImg = self.dataObj.getData(self.batchSize)
        pvFile.close()

    #TODO
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

