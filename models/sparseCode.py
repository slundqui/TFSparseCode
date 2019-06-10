import tensorflow as tf
from models.base import base
import models.utils as utils
import pdb
import numpy as np
from models.lcaSC import lcaSC
from plots import plotRecon, plotWeights

class sparseCode(base):
    def buildModel(self):
        with tf.device(self.params.device):
            self.train_classifier = False
            with tf.name_scope("placeholders"):
                #TODO Split input into input and ground truth, since last input is never being used
                curr_input_shape = [self.params.batch_size, ] + self.params.input_shape
                self.input = tf.placeholder(tf.float32,
                        shape=curr_input_shape,
                        name = "input")



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
                self.varDict["norm_input"] = self.norm_input
                #self.varDict["mask"] = self.mask

            with tf.name_scope("sc"):
                self.scObj = lcaSC(self.norm_input, self.params.l1_weight,
                        self.params.dict_size, self.params.sc_lr, self.params.D_lr,
                        layer_type=self.params.layer_type,
                        patch_size = self.params.dict_patch_size,
                        stride=self.params.stride,
                        #optimizer = tf.train.GradientDescentOptimizer
                        gan=self.params.use_gan,
                        gan_weight = self.params.gan_weight
                        )

            with tf.name_scope("active_buf"):
                #Keep track of last 10 batches to calculate most active
                num_buf = 10
                self.update_act_count = []
                self.active_count = []

                curr_act= self.scObj.model["activation"]
                if(curr_act is None):
                    self.update_act_count = tf.no_op()
                    self.active_count = tf.no_op()
                else:
                    #Reduce everything but last axis
                    reduce_axis = list(range(len(curr_act.get_shape().as_list()) - 1))
                    curr_act_count = tf.reduce_sum(tf.cast(tf.greater(curr_act, 0), tf.float32), axis=reduce_axis)

                    most_active_buf = tf.Variable(tf.zeros([num_buf, self.params.dict_size]), trainable=False, name="activation_count")
                    idx = tf.mod(self.timestep, num_buf)
                    self.update_act_count = tf.scatter_update(most_active_buf, idx, curr_act_count)
                    self.active_count = tf.reduce_sum(most_active_buf, axis=0)

                #Final recon set here
                self.input_recon = self.scObj.model["recon"]
                if(self.params.norm_input):
                    self.unscaled_recon = (self.input_recon/self.params.target_norm_std) * tf.sqrt(data_var) + data_mean
                else:
                    self.unscaled_recon = self.input_recon/self.params.target_norm_std

            with tf.variable_scope("classifier"):
                if(self.params.use_classifier):
                    self.gt = tf.placeholder(tf.int64,
                        shape=[self.params.batch_size,],
                        name = "gt")
                    act = self.scObj.model["activation"]
                    #Global average pool across conv dimension
                    pooled_act = tf.reduce_mean(act, axis=1)

                    logits = tf.layers.dense(pooled_act, 2,activation=None)

                    self.classifier_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                        scope="classifier/dense/kernel")[0]

                    self.classifier_bias = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                        scope="classifier/dense/bias")[0]

                    self.est = tf.nn.softmax(logits, axis=-1)
                    self.est_class = tf.argmax(self.est, axis=-1)
                    self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.est_class, self.gt), tf.float32))

                    self.class_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=self.gt, logits=logits)

                    self.class_opt = tf.train.AdamOptimizer(self.params.class_lr).minimize(self.class_loss,
                        var_list=[self.classifier_weights, self.classifier_bias])

                    self.varDict["classifier_w"] = self.classifier_weights
                    self.varDict["classifier_b"] = self.classifier_bias
                    self.varDict["classifier_logits"] = logits
                    self.varDict["layer_pooled_output"] = pooled_act
                    self.scalarDict["classifier_loss"] = tf.reduce_sum(self.class_loss)
                    self.scalarDict["accuracy"] = self.accuracy


            self.varDict   ["layer_dict"]        = self.scObj.model["dictionary"]
            self.varDict   ["layer_input"]       = self.scObj.model["input"]
            self.varDict   ["layer_output"]      = self.scObj.model["activation"]
            self.scalarDict["layer_nnz"]        = self.scObj.model["nnz"]


            self.varDict   ["layer_potential"]   = self.scObj.model["potential"]
            self.varDict   ["layer_recon"]          = self.scObj.model["recon"]
            self.scalarDict["layer_sc_recon_err"]   = self.scObj.model["recon_error"]
            self.scalarDict["layer_sc_l1_sparsity"] = self.scObj.model["l1_sparsity"]
            self.scalarDict["layer_sc_loss"]        = self.scObj.model["loss"]

    def getLoadVars(self):
        load_vars = ["global_step:0", "sc/lca_layer/dictionary:0"]

        vv = [v for v in tf.global_variables() if v.name in load_vars]

        if(self.params.load_gan):
            vvv = [v for v in tf.global_variables() if "GAN" in v.name]
            vv = vvv + vv
        if(self.params.load_classifier):
           vvv = [v for v in tf.global_variables() if "classifier" in v.name]
           vv = vvv + vv

        return vv

    def getTrainFeedDict(self, dataObj):
        dataDict = dataObj.getData(self.params.batch_size, dataset="train")
        outdict = {}
        outdict[self.input] = dataDict['data']
        if(self.params.use_classifier):
            outdict[self.gt] = dataDict['gt']
        return outdict

    def getTestFeedDict(self, dataObj):
        dataDict = dataObj.getData(self.params.batch_size, dataset="test")
        outdict = {}
        outdict[self.input] = dataDict['data']
        if(self.params.use_classifier):
            outdict[self.gt] = dataDict['gt']
        return outdict

    def getEvalFeedDict(self, data):
        outdict={}
        outdict[self.input] = data
        return outdict

    def evalModel(self, feed_dict):
        self.scObj.calcActivations(self.sess, feed_dict, max_iterations=self.params.sc_iter, verbose=self.params.sc_verbose)
        return self.sess.run(self.scObj.model["activation"])

    def calcRecon(self, act):
        feed_dict = {self.scObj.model["inject_act_placeholder"]: act}
        return self.sess.run(self.scObj.model["recon_from_act"], feed_dict)

    def plotRecon(self, feed_dict, fn_prefix, is_train):
        np_input = self.sess.run(self.norm_input, feed_dict=feed_dict)
        np_recon = self.sess.run(self.input_recon, feed_dict=feed_dict)

        np_unscaled_input = feed_dict[self.input]
        np_unscaled_recon = self.sess.run(self.unscaled_recon, feed_dict=feed_dict)

        if(self.ndims_input == 3):
            plotRecon.plotRecon1D(np_recon, np_input, fn_prefix+"recon",
                    num_plot = self.params.num_plot_recon,
                    unscaled_img_matrix = np_unscaled_input, unscaled_recon_matrix=np_unscaled_recon,
                    groups=self.params.plot_groups, group_title=self.params.plot_group_title,
                    legend=self.params.legend)
        else:
            assert(0)

    def plotWeights(self, fn_prefix):
        np_dict = self.sess.run(self.scObj.model["dictionary"])
        np_act_count = self.sess.run(self.active_count)

        curr_dict = np_dict
        curr_act_count = np_act_count

        #Plot weights
        plotWeights.plotWeights1D(curr_dict, fn_prefix+"layer_weights",
                order=[2,0,1],
                activity_count=curr_act_count, group_policy="group",
                num_plot = self.params.num_plot_weights,
                groups=self.params.plot_groups, group_title=self.params.plot_group_title,
                legend=self.params.legend)

    def plot(self, step, feed_dict, fn_prefix, is_train):
        print("Plotting recon")
        self.plotRecon(feed_dict, fn_prefix, is_train)
        print("Plotting weights")
        self.plotWeights(fn_prefix)

    def trainStepInit(self, train_feed_dict):
        #Compute sc
        self.scObj.calcActivations(self.sess, train_feed_dict, max_iterations=self.params.sc_iter, verbose=self.params.sc_verbose)
        #Update active count buffer
        self.sess.run(self.update_act_count)

    def testStepInit(self, test_feed_dict):
        #Compute sc
        self.scObj.calcActivations(self.sess, test_feed_dict, max_iterations=self.params.sc_iter, verbose=self.params.sc_verbose)

    def trainClassifier(self, trainDataObj):
       assert(self.params.use_classifier)
       self.train_classifier = True
       self.trainModel(trainDataObj)

    def trainStep(self, step, train_feed_dict):
        if(self.train_classifier):
            [accuracy, drop] = self.sess.run([self.accuracy, self.class_opt], train_feed_dict)
            #TODO fancier printing
            if(self.params.progress > 0 and step % self.params.progress == 0):
                print("Train Accuracy: ", accuracy)
        else:
            self.scObj.updateDict(self.sess, train_feed_dict)

    #For printing test accuracy
    def evalModelSummary(self, test_feed_dict):
        super(sparseCode, self).evalModelSummary(test_feed_dict)
        accuracy = self.sess.run(self.accuracy, test_feed_dict)
        print("Test Accuracy: ", accuracy)


