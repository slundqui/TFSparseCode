import tensorflow as tf
from models.base import base
import models.utils as utils
import pdb
import numpy as np
from models.lcaSC import lcaSC
from plots import plotRecon, plotWeights

class mlp(base):
    def buildModel(self):
        with tf.device(self.params.device):
            with tf.name_scope("placeholders"):
                curr_input_shape = [self.params.batch_size, ] + self.params.input_shape
                self.input = tf.placeholder(tf.float32,
                        shape=curr_input_shape,
                        name = "input")

                self.gt = tf.placeholder(tf.int64,
                    shape=[self.params.batch_size], name="gt")

                self.ndims_input = len(curr_input_shape)
                #TODO add in for images instead of only 1d
                if(self.ndims_input == 3):
                    (example_size, num_features) = self.params.input_shape
                else:
                    print("Not implemented")
                    assert(0)

                if(self.params.norm_ind_features):
                    norm_reduction_idx = [1,]
                else:
                    norm_reduction_idx = [1,2]
                (data_mean, data_var) = tf.nn.moments(self.input, axes=norm_reduction_idx, keep_dims=True)
                if(self.params.norm_input):
                    calc_norm = ((self.input - data_mean)/tf.sqrt(data_var)) * self.params.target_norm_std
                    #Expand data_mean and data_var to have same shape as input
                    if(self.params.norm_ind_features):
                        data_mean = tf.tile(data_mean, [1, example_size, 1])
                        data_var = tf.tile(data_var, [1, example_size, 1])
                    else:
                        data_mean = tf.tile(data_mean, [1, example_size, num_features])
                        data_var = tf.tile(data_var, [1, example_size, num_features])

                    self.norm_input = tf.where(tf.equal(data_var, 0), data_mean, calc_norm)
                else:
                    self.norm_input = self.input * self.params.target_norm_std

                self.varDict["input"] = self.input
                self.varDict["gt"] = self.gt
                self.varDict["norm_input"] = self.norm_input
                #self.varDict["mask"] = self.mask

            with tf.variable_scope("mlp"):
                self.latent_layer= tf.layers.Conv1D(self.params.dict_size,
                    self.params.dict_patch_size, self.params.stride,
                    padding="same", activation="relu")
                self.act = self.latent_layer(self.norm_input)

                self.latent_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                    scope="mlp/conv1d/kernel")[0]
                self.latent_bias = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                    scope="mlp/conv1d/bias")[0]

                #Global average pool across conv dimension
                pooled_act = tf.reduce_mean(self.act, axis=1)

                logits = tf.layers.dense(pooled_act, 2, activation=None)

                self.classifier_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                    scope="mlp/dense/kernel")[0]
                self.classifier_bias = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                    scope="mlp/dense/bias")[0]

                self.est = tf.nn.softmax(logits, axis=-1)
                self.est_class = tf.argmax(self.est, axis=-1)
                self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.est_class, self.gt), tf.float32))
                self.class_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=self.gt, logits=logits)

                self.class_opt = tf.train.AdamOptimizer(self.params.class_lr).minimize(self.class_loss,
                    var_list=[self.latent_weights, self.latent_bias, self.classifier_weights, self.classifier_bias])

            self.varDict["latent_w"] = self.latent_weights
            self.varDict["latent_b"] = self.latent_bias
            self.varDict["classifier_w"] = self.classifier_weights
            self.varDict["classifier_b"] = self.classifier_bias
            self.varDict["layer_pooled_output"] = pooled_act
            self.scalarDict["classifier_loss"] = tf.reduce_sum(self.class_loss)
            self.scalarDict["accuracy"] = self.accuracy

            self.varDict   ["layer_input"]       = self.input
            self.varDict   ["layer_output"]      = self.act

    def getTrainFeedDict(self, dataObj):
        dataDict = dataObj.getData(self.params.batch_size, dataset="train")
        outdict = {}
        outdict[self.input] = dataDict['data']
        outdict[self.gt] = dataDict['gt']
        return outdict

    def getTestFeedDict(self, dataObj):
        dataDict = dataObj.getData(self.params.batch_size, dataset="test")
        outdict = {}
        outdict[self.input] = dataDict['data']
        outdict[self.gt] = dataDict['gt']
        return outdict

    def getEvalFeedDict(self, data):
        outdict={}
        outdict[self.input] = data
        return outdict

    def evalModel(self, feed_dict):
      #TODO
      assert(False)

    def plotWeights(self, fn_prefix):
        np_dict = self.sess.run(self.latent_weights)

        curr_dict = np_dict

        #Plot weights
        plotWeights.plotWeights1D(curr_dict, fn_prefix+"layer_weights",
                order=[2,0,1],
                group_policy="group",
                num_plot = self.params.num_plot_weights,
                groups=self.params.plot_groups, group_title=self.params.plot_group_title,
                legend=self.params.legend)

    def plot(self, step, feed_dict, fn_prefix, is_train):
        print("Plotting weights")
        self.plotWeights(fn_prefix)

    def trainStepInit(self, train_feed_dict):
        pass

    def testStepInit(self, test_feed_dict):
        pass

    def trainStep(self, step, train_feed_dict):
        [drop, accuracy] = self.sess.run([self.class_opt, self.accuracy], train_feed_dict)
        if(self.params.progress > 0 and step % self.params.progress == 0):
          print("Train Accuracy: ", accuracy)

    #For printing test accuracy
    def evalModelSummary(self, test_feed_dict):
        super(mlp, self).evalModelSummary(test_feed_dict)
        accuracy = self.sess.run(self.accuracy, test_feed_dict)
        print("Test Accuracy: ", accuracy)


