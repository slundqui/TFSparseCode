import tensorflow as tf
from models.base import base
import models.utils as utils
import pdb
import numpy as np
from models.lcaDeepSC import lcaDeepSC
from plot.plot import plotRecon, plotWeights

class sparseCode(base):
    def buildModel(self):
        with tf.device(self.params.device):
            with tf.name_scope("placeholders"):
                #TODO Split input into input and ground truth, since last input is never being used
                self.input = tf.placeholder(tf.float32,
                        shape=[self.params.batch_size, ] + self.params.image_shape,
                        name = "input")

                (data_mean, data_var) = tf.nn.moments(self.sub_input, axes=1, keep_dims=True)
                if(self.params.norm_input):
                    calc_norm = ((self.sub_input - data_mean)/tf.sqrt(data_var)) * self.params.target_norm_std
                    #Expand data_mean and data_var to have same shape as input
                    data_mean = tf.tile(data_mean, [1, self.params.example_size, 1])
                    data_var = tf.tile(data_var, [1, self.params.example_size, 1])

                    self.norm_input = tf.where(tf.equal(data_var, 0), data_mean, calc_norm)
                else:
                    self.norm_input = self.sub_input * self.params.target_norm_std

                self.varDict["input"] = self.sub_input
                self.varDict["norm_input"] = self.norm_input
                #self.varDict["mask"] = self.mask

            with tf.name_scope("sc"):
                ##For semi-supervised learning, i.e., clamping top set of neurons
                ##TODO better just to add supervised loss?
                #self.inject_act_bool = tf.placeholder_with_default([False for i in range(self.params.batch_size)], shape=[self.params.batch_size], name="inject_act_bool")
                ##Calculate last layer shape
                #if("fc" in self.params.layer_type[self.params.num_layers-1]):
                #    last_layer_shape = [self.params.batch_size, self.params.dict_size[self.params.num_layers-1]]
                #else:
                #    input_shape = self.sub_input.get_shape().as_list()
                #    num_x = input_shape[1]
                #    for stride in self.params.stride[:self.params.num_layers]:
                #        assert(stride is not None)
                #        num_x = num_x // stride
                #    last_layer_shape = [self.params.batch_size, num_x, self.params.dict_size[self.params.num_layers-1]]

                #self.inject_act = tf.placeholder_with_default(tf.zeros(last_layer_shape, dtype=tf.float32), shape=last_layer_shape, name="inject_act")

                self.scObj = lcaDeepSC(self.norm_input, self.params.num_layers, self.params.l1_weight,
                        self.params.dict_size, self.params.sc_lr, self.params.D_lr,
                        layer_type=self.params.layer_type,
                        patch_size = self.params.dict_patch_size,
                        stride=self.params.stride, err_weight=self.params.err_weight,
                        act_weight=self.params.act_weight,
                        #inject_act_bool = self.inject_act_bool, inject_act = self.inject_act,
                        normalize_act=self.params.normalize_act,
                        )

            with tf.name_scope("active_buf"):
                #Keep track of last 10 batches to calculate most active
                num_buf = 10
                self.update_act_count = []
                self.active_count = []
                for l in range(self.params.num_layers):
                    curr_act= self.scObj.model["activation"][l]
                    if(curr_act is None):
                        self.update_act_count.append(tf.no_op())
                        self.active_count.append(tf.no_op())
                    else:
                        #Reduce everything but last axis
                        reduce_axis = list(range(len(curr_act.get_shape().as_list()) - 1))
                        curr_act_count = tf.reduce_sum(tf.cast(tf.greater(curr_act, 0), tf.float32), axis=reduce_axis)

                        most_active_buf = tf.Variable(tf.zeros([num_buf, self.params.dict_size[l]]), trainable=False, name="activation_count_" + str(l))
                        idx = tf.mod(self.timestep, num_buf)
                        self.update_act_count.append(tf.scatter_update(most_active_buf, idx, curr_act_count))
                        self.active_count.append(tf.reduce_sum(most_active_buf, axis=0))

                self.update_act_count = tf.group(*self.update_act_count)

                for l in range(self.params.num_layers):
                    self.varDict   ["layer_"+str(l)+"_dict"]        = self.scObj.model["dictionary"][l]
                    self.varDict   ["layer_"+str(l)+"_input"]          = self.scObj.model["input"][l]
                    self.varDict   ["layer_"+str(l)+"_output"]          = self.scObj.model["output"][l]
                    self.scalarDict["layer_"+str(l)+"_nnz"]         = self.scObj.model["nnz"][l]

                for l in range(self.params.num_layers):
                    if("sc" not in self.params.layer_type[l]):
                        continue
                    self.varDict   ["layer_"+str(l)+"_sc_potential"]   = self.scObj.model["potential"][l]
                    self.varDict   ["layer_"+str(l)+"_sc_activation"]  = self.scObj.model["activation"][l]
                    self.varDict   ["layer_"+str(l)+"_recon"]          = self.scObj.model["recon"][l]
                    self.scalarDict["layer_"+str(l)+"_sc_recon_err"]   = self.scObj.model["recon_error"][l]
                    self.scalarDict["layer_"+str(l)+"_sc_l1_sparsity"] = self.scObj.model["l1_sparsity"][l]
                    self.scalarDict["layer_"+str(l)+"_sc_loss"]        = self.scObj.model["loss"][l]

                self.scalarDict["total_recon_error"] = self.scObj.model["total_recon_error"]


    def getTrainFeedDict(self, dataObj):
        dataDict = dataObj.getData(self.params.batch_size, self.params.example_size, dataset="train")
        outdict = {}
        outdict[self.input] = dataDict['data']
        return outdict

    def getTestFeedDict(self, dataObj):
        dataDict = dataObj.getData(self.params.batch_size, self.params.example_size, dataset="test")
        outdict = {}
        outdict[self.input] = dataDict['data']
        return outdict

    def getEvalFeedDict(self, data):
        outdict={}
        outdict[self.input] = data
        return outdict

    def evalModel(self, feed_dict):
        self.scObj.calcActivations(self.sess, feed_dict, max_iterations=self.params.sc_iter, verbose=self.params.sc_verbose)
        return self.sess.run(self.scObj.model["activation"][-1])

    def plotRecon(self, feed_dict, fn_prefix, is_train):
        np_input = self.sess.run(self.norm_input, feed_dict=feed_dict)
        np_recon = self.sess.run(self.scObj.model["recon"][0])
        #TODO
        assert(0)

    def plotWeights(self, fn_prefix):
        np_dict = self.sess.run(self.scObj.model["layer_weights"])
        np_act_count = self.sess.run(self.active_count)
        for l in range(self.params.num_layers):
            if(np_dict[l] is None):
                continue
            curr_dict = np_dict[l]
            curr_act_count = np_act_count[l]
            #Plot weights
            plotWeights(curr_dict, fn_prefix+"layer_"+str(l) + "_weights", activity_count=curr_act_count, legend = self.params.labels)

    def plot(self, step, feed_dict, fn_prefix, is_train):
        self.plotRecon(feed_dict, fn_prefix, is_train)
        self.plotWeights(fn_prefix)

    def trainStepInit(self, train_feed_dict):
        #Compute sc
        self.scObj.calcActivations(self.sess, train_feed_dict, max_iterations=self.params.sc_iter, verbose=self.params.sc_verbose)
        #Update active count buffer
        self.sess.run(self.update_act_count)

    def testStepInit(self, test_feed_dict):
        #Compute sc
        self.scObj.calcActivations(self.sess, test_feed_dict, max_iterations=self.params.sc_iter, verbose=self.params.sc_verbose)

    def trainStep(self, step, train_feed_dict):
        self.scObj.updateDict(self.sess, train_feed_dict)


