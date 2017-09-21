import pdb
import numpy as np
import tensorflow as tf
from .utils import *
import matplotlib.pyplot as plt
from plots.plotWeights import plot_weights
from base import base

class Supervised(base):

    #Sets dictionary of params to member variables
    def loadParams(self, params):
        super(Supervised, self).loadParams(params)
        self.learningRate = params['learningRate']
        self.numClasses = params['numClasses']
        self.VStrideY = params['VStrideY']
        self.VStrideX = params['VStrideX']
        self.patchSizeY = params['patchSizeY']
        self.patchSizeX = params['patchSizeX']
        self.numV = params['numV']
        self.maxPool = params['maxPool']
        self.epsilon = params['epsilon']
        self.regularizer = params['regularizer']
        self.regWeight = params['regWeight']
        self.preTrain = params['preTrain']


    def runModel(self, trainDataObj, testDataObj=None, numTest = None):
        #Load summary
        self.writeSummary()
        for i in range(self.numIterations):
           #Plot flag
           if(i%self.plotPeriod == 0):
               plot = True
           else:
               plot=False
           if(testDataObj):
               if(numTest is None):
                   numTest = self.batchSize
               #Evaluate test frame, providing gt so that it writes to summary
               (evalData, gtData) = testDataObj.getData(numTest)
               self.evalModel(evalData, gtData, plot=plot)
               print("Done test eval")
           #Train
           if(i%self.savePeriod == 0):
               self.trainModel(trainDataObj, save=True, plot=plot)
           else:
               self.trainModel(trainDataObj, save=False, plot=plot)


    #Builds the model. inMatFilename should be the vgg file
    def buildModel(self, inputShape):
        with tf.device(self.device):
            self.imageShape = (self.batchSize, inputShape[0], inputShape[1], inputShape[2])
            self.encodeWeightShape = (self.patchSizeY, self.patchSizeX, inputShape[2], self.numV)
            self.weightShape = (4*self.numV, self.numClasses)
            with tf.name_scope("inputOps"):
                #Get convolution variables as placeholders
                self.input = node_variable([None, inputShape[0], inputShape[1], inputShape[2]], "inputImage")
                self.gt = node_variable([None, self.numClasses], "gt")
                #Model variables for convolutions

            with tf.name_scope("Encode"):
                self.W_encode = weight_variable_xavier(self.encodeWeightShape, "encode_w", True)
                self.B_encode = bias_variable([self.numV], "encode_b")
                self.h_encode = tf.nn.relu(conv2d(self.input, self.W_encode, "h_encode", [1, 2, 2, 1]) + self.B_encode)

            with tf.name_scope("SLP"):
                #Max pooled values
                self.W_slp = weight_variable_xavier(self.weightShape, "slp_w", False)
                self.B_slp = bias_variable([self.numClasses], "slp_b")
                if(self.maxPool):
                    self.pooled = tf.nn.max_pool(self.h_encode, ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1], padding='SAME', name="pooled")
                else:
                    self.pooled = tf.nn.avg_pool(self.h_encode, ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1], padding='SAME', name="pooled")
                self.flat_pooled = tf.reshape(self.pooled, [-1, 4*self.numV])
                #self.pooled = tf.reduce_max(self.relu_input, reduction_indices=[1, 2])
                self.est = tf.nn.softmax(tf.matmul(self.flat_pooled, self.W_slp) + self.B_slp)

            with tf.name_scope("Loss"):
                #Define loss
                self.loss = tf.reduce_mean(-tf.reduce_sum(self.gt * tf.log(self.est+self.epsilon), reduction_indices=[1]))

                if(self.regularizer == "none"):
                    self.reg_loss = self.loss
                elif(self.regularizer == "weightsl1"):
                    self.reg_loss = self.loss + self.regWeight * tf.reduce_mean(tf.abs(self.W_encode))
                elif(self.regularizer == "weightsl2"):
                    self.reg_loss = self.loss + self.regWeight * tf.reduce_mean(tf.square(self.W_encode))
                elif(self.regularizer == "activitesl1"):
                    self.reg_loss = self.loss + self.regWeight * tf.reduce_mean(tf.abs(self.h_encode))
                elif(self.regularizer == "activitesl2"):
                    self.reg_loss = self.loss + self.regWeight * tf.reduce_mean(tf.square(self.h_encode))
                else:
                    assert(0)

            with tf.name_scope("Opt"):
                #Define optimizer
                self.optimizer = tf.train.AdamOptimizer(self.learningRate).minimize(self.reg_loss)
                self.optimizerPre = tf.train.AdamOptimizer(self.learningRate).minimize(self.reg_loss,
                        var_list=[
                            self.W_slp,
                            self.B_slp
                        ]
                        )

            with tf.name_scope("Metric"):
                self.correct = tf.equal(tf.argmax(self.gt, 1), tf.argmax(self.est, 1))
                self.accuracy = tf.reduce_mean(tf.cast(self.correct, tf.float32))

        #Summaries
        tf.scalar_summary('loss', self.loss, name="loss")
        tf.scalar_summary('reg_loss', self.reg_loss, name="reg_loss")
        tf.scalar_summary('accuracy', self.accuracy, name="accuracy")

        tf.histogram_summary('input', self.input, name="image")
        tf.histogram_summary('h_encode', self.input, name="image")
        tf.histogram_summary('pooled', self.pooled, name="pooled")
        tf.histogram_summary('gt', self.gt, name="gt")
        #Conv layer histograms
        tf.histogram_summary('est', self.est, name="est")
        #Weight and bias hists
        tf.histogram_summary('w_encode', self.W_slp, name="w_slp")
        tf.histogram_summary('b_encode', self.B_slp, name="b_slp")
        tf.histogram_summary('w_slp', self.W_slp, name="w_slp")
        tf.histogram_summary('b_slp', self.B_slp, name="b_slp")

    #Trains model for numSteps
    #If pre is False, will train entire network
    #If pre is True, will train only fully connected network
    def trainModel(self, dataObj, save, plot):
        #Define session
        for i in range(self.displayPeriod):
            #Get data from dataObj
            data = dataObj.getData(self.batchSize)
            feedDict = {self.input: data[0], self.gt: data[1]}
            #Run optimizer
            if(self.preTrain):
                self.sess.run(self.optimizerPre, feed_dict=feedDict)
            else:
                self.sess.run(self.optimizer, feed_dict=feedDict)
            if(i%self.writeStep == 0):
                summary = self.sess.run(self.mergedSummary, feed_dict=feedDict)
                self.train_writer.add_summary(summary, self.timestep)
            if(i%self.progress == 0):
                print("Timestep ", self.timestep)
            self.timestep+=1
        if(save):
            save_path = self.saver.save(self.sess, self.saveFile, global_step=self.timestep, write_meta_graph=False)
            print("Model saved in file: %s" % save_path)
        if(plot):
            filename = self.plotDir + "weights_" + str(self.timestep) + ".png"
            np_w = self.sess.run(self.W_encode, feed_dict=feedDict)
            plot_weights(np_w, filename, order=[3, 0, 1, 2])

    #Evaluates all of inData at once
    #If an inGt is provided, will calculate summary as test set
    def evalModel(self, inData, inGt = None, plot=True):
        (numData, ny, nx, nf) = inData.shape
        if(inGt is not None):
            (numGt, drop) = inGt.shape
            assert(numData == numGt)
            feedDict = {self.input: inData, self.gt: inGt}
        else:
            feedDict = {self.input: inData}

        outVals = self.est.eval(feed_dict=feedDict, session=self.sess)
        if(inGt is not None):
            summary = self.sess.run(self.mergedSummary, feed_dict=feedDict)
            self.test_writer.add_summary(summary, self.timestep)
        #if(plot and inGt != None):
        #    filename = self.plotDir + "test_" + str(self.timestep) + ".png"
        #    self.evalAndPlotCam(feedDict, filename)

        return outVals

