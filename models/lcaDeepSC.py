import tensorflow as tf
import models.utils as utils
import numpy as np
import pdb

class lcaDeepSC(object):
    def __init__(self, inputNode, num_layers, l1_weight, dict_size, sc_lr, dict_lr, layer_type=None, patch_size=None, stride=None, mask=None, err_weight=None, act_weight=None, top_down_weight=None, normalize_act=None, inject_act_bool=None, inject_act=None):
        curr_input = inputNode
        #Model variables and outputs
        self.model = {}
        self.model["dictionary"] = []
        self.model["potential"] = []
        self.model["activation"] = []
        self.model["recon"] = []
        self.model["input"] = []
        self.model["output"] = []
        self.model["error"] = []
        self.model["recon_error"] = []
        self.model["l1_sparsity"] = []
        self.model["nnz"] = []
        self.model["loss"] = []
        self.model["act_norm"] = []
        self.model["act_mean"] = []
        self.model["act_std"] = []
        self.model["act_max"] = []
        self.model["pot_norm"] = []
        self.model["pot_mean"] = []
        self.model["pot_std"] = []
        self.model["input_norm"] = []
        self.model["output_norm"] = []

        assert(layer_type is not None)


        #Model operations
        self.calc_activation = []
        self.reset_potential = []

        switch_fc = False

        if(err_weight is None):
            err_weight = [1 for i in range(num_layers)]
        if(act_weight is None):
            act_weight = [1 for i in range(num_layers)]
        if(top_down_weight is None):
            top_down_weight = [1 for i in range(num_layers)]

        for l in range(num_layers):
            with tf.name_scope("lca_layer_"+str(l)):
                curr_layer_type = layer_type[l]
                curr_dict_size = dict_size[l]
                curr_stride = stride[l]
                curr_patch_size = patch_size[l]
                curr_l1_weight = l1_weight[l]
                curr_normalize = normalize_act[l]

                input_shape = curr_input.get_shape().as_list()

                if(len(input_shape) == 3):
                    [batch, input_size, input_features] = input_shape
                else:
                    [batch, input_features] = input_shape

                if("sc_fc" == curr_layer_type):
                    switch_fc = True
                    curr_input = tf.reshape(curr_input, [batch, -1])
                    input_features = curr_input.get_shape().as_list()[1]
                    D_shape = [input_features, curr_dict_size]
                    act_shape = [batch, curr_dict_size]
                    reduce_axis = [1]
                else:
                    assert(not switch_fc)
                    D_shape = [curr_patch_size, input_features, curr_dict_size]
                    assert(input_size % curr_stride == 0)
                    act_shape = [batch, input_size//curr_stride, curr_dict_size]
                    reduce_axis = [1, 2]

                curr_dict = utils.l2_weight_variable(D_shape, "dictionary"+str(l))

                curr_potential = utils.weight_variable(act_shape, "potential"+str(l), std=1e-3)
                curr_activation = utils.weight_variable(act_shape, "activation"+str(l), std=1e-3)

                if("sc_fc" == curr_layer_type):
                    curr_recon = tf.matmul(curr_activation, curr_dict, transpose_b=True)
                elif("sc_conv" == curr_layer_type):
                    curr_recon = tf.contrib.nn.conv1d_transpose(curr_activation, curr_dict, [batch, input_size, input_features], curr_stride, padding='SAME')
                else:
                    assert(0)

                curr_error = curr_input - curr_recon
                curr_recon_error = err_weight[l] * 0.5 * tf.reduce_mean(tf.reduce_sum(curr_error**2, axis=reduce_axis))
                curr_l1_sparsity = err_weight[l] * tf.reduce_mean(tf.reduce_sum(tf.abs(curr_activation), axis=reduce_axis))
                #curr_recon_error = err_weight[l] * 0.5 * tf.reduce_mean(curr_error**2)
                #curr_l1_sparsity = err_weight[l] * tf.reduce_mean(tf.abs(curr_activation))
                curr_loss = curr_recon_error + 0.5 * curr_l1_weight * curr_l1_sparsity

                self.model["error"].append(curr_error)
                self.model["recon_error"].append(curr_recon_error)
                self.model["potential"].append(curr_potential)
                self.model["activation"].append(curr_activation)
                self.model["recon"].append(curr_recon)
                self.model["l1_sparsity"].append(curr_l1_sparsity)
                self.model["loss"].append(curr_loss)

                #Ops
                #Use inject act if last layer for semi-supervised learning
                calc_act = tf.nn.relu(curr_potential - curr_l1_weight)
                if(l == num_layers - 1 and inject_act_bool is not None):
                    set_act = tf.where(inject_act_bool, inject_act, calc_act)
                    self.calc_activation.append(curr_activation.assign(set_act))
                else:
                    self.calc_activation.append(curr_activation.assign(calc_act))

                if(curr_l1_weight == 0):
                    low_init_val = -.1
                    high_init_val = .1
                else:
                    low_init_val = 0
                    high_init_val  = 1.01*curr_l1_weight
                potential_init = tf.random_uniform(act_shape, low_init_val, high_init_val, dtype=tf.float32)
                self.reset_potential.append(curr_potential.assign(potential_init))

                num_total_act = 1
                for s in act_shape:
                    num_total_act *= s

                curr_nnz = tf.count_nonzero(curr_activation) / num_total_act

                #Save all variables
                self.model["dictionary"].append(curr_dict)
                self.model["output"].append(curr_activation)
                self.model["input"].append(curr_input)
                self.model["nnz"].append(curr_nnz)

                #Calculate means/std of activations
                #Do this across batches
                #Normalize each feature/dictionary element individually
                if(len(act_shape) == 3):
                    moment_reduce_axis = [0, 1]
                    tile_input = [act_shape[0], act_shape[1], 1]
                elif(len(act_shape) == 2):
                    moment_reduce_axis = 0
                    tile_input = [act_shape[0], 1]
                else:
                    assert(0)

                act_norm = tf.norm(curr_activation, axis=moment_reduce_axis, keepdims=True)
                act_mean, act_var = tf.nn.moments(curr_activation, axes=moment_reduce_axis, keep_dims=True)
                act_std = tf.sqrt(act_var)
                act_max = tf.reduce_max(curr_activation)

                pot_norm = tf.norm(curr_potential, axis=moment_reduce_axis, keepdims=True)
                pot_mean, pot_var = tf.nn.moments(curr_potential, axes=moment_reduce_axis, keep_dims=True)
                pot_std = tf.sqrt(pot_var)

                self.model["act_norm"].append(act_norm)
                self.model["act_mean"].append(act_mean)
                self.model["act_std"].append(act_std)
                self.model["act_max"].append(act_max)
                self.model["pot_norm"].append(pot_norm)
                self.model["pot_mean"].append(pot_mean)
                self.model["pot_std"].append(pot_std)

                input_norm = tf.norm(curr_input, axis=moment_reduce_axis)
                self.model["input_norm"].append(input_norm)

                if(curr_normalize):
                    #curr_input = ((curr_activation - act_mean)/(act_std+1e-8)) * act_weight[l]
                    curr_input = ((curr_potential - pot_mean)/(pot_std+1e-8)) * act_weight[l]
                else:
                    #curr_input = curr_activation * act_weight[l]
                    curr_input = curr_potential * act_weight[l]

                output_norm = tf.norm(curr_input, axis=moment_reduce_axis)
                self.model["output_norm"].append(output_norm)

                #Stop gradient, as we explcitly compute top down feedback
                curr_input = tf.stop_gradient(curr_input)

        with tf.name_scope("optimizer"):
            #Group ops
            self.calc_activation = tf.group(*self.calc_activation)
            self.reset_potential = tf.group(*self.reset_potential)

            #Define optimizer
            #TODO different learning rates?
            opt = tf.train.AdamOptimizer(sc_lr)

            total_recon_error = tf.reduce_sum(self.model["recon_error"])
            self.model["total_recon_error"] = total_recon_error

            #Calculate recon gradient wrt activation
            recon_grad = opt.compute_gradients(total_recon_error, self.model["activation"])

            #Apply gradient (plus shrinkage) to potential
            #Needs to be a list of number of gradients, each element as a tuple of (gradient, wrt)

            d_potential = []
            for i, (grad, var) in enumerate(recon_grad):
                shrink_term = err_weight[i] * (1/batch) * (self.model["potential"][i] - self.model["activation"][i])
                #The top down term doesn't exist with the recon loss as written, since potential
                #isnt connected to the total recon loss
                if(i < (num_layers - 1)):
                    top_down_term = top_down_weight[i] * self.model["error"][i+1]
                else:
                    top_down_term = 0
                d_potential.append((grad + shrink_term - top_down_term, self.model["potential"][i]))

            self.train_step = opt.apply_gradients(d_potential)
            #Reset must be called after apply_gradients to define opt variables
            self.reset_opt = tf.group([v.initializer for v in opt.variables()])

            #Dictionary update variables
            opt_D = tf.train.AdamOptimizer(dict_lr)
            self.update_D = opt_D.minimize(total_recon_error, var_list=[self.model["dictionary"]])

            #Normalize D
            self.normalize_D = []
            for l in range(num_layers):
                curr_dict = self.model["dictionary"][l]
                dict_shape = curr_dict.get_shape().as_list()
                if(len(dict_shape) == 3):
                    curr_norm = tf.norm(curr_dict, axis=(0, 1))
                else:
                    curr_norm = tf.norm(curr_dict, axis=0)
                curr_norm = tf.maximum(tf.ones(dict_shape), curr_norm)
                self.normalize_D.append(curr_dict.assign(curr_dict/curr_norm))
            self.normalize_D = tf.group(*self.normalize_D)

        with tf.name_scope("weight_recon"):
            #Allows calculating reconstruction from each layer
            layer_weights = []
            for l in range(num_layers):
                recon_l_fc = ("sc_fc" == layer_type[l])
                recon_l_num_dict = dict_size[l]
                recon_act = tf.eye(recon_l_num_dict)
                if(not recon_l_fc):
                    recon_act = recon_act[:, tf.newaxis, :]

                switch_conv = not recon_l_fc

                curr_act = recon_act
                curr_pot = None
                for ll in reversed(range(l+1)):
                    curr_dict = self.model["dictionary"][ll]
                    curr_layer_type = layer_type[ll]
                    curr_stride = stride[ll]
                    curr_patch_size = patch_size[ll]
                    curr_l1_weight = l1_weight[ll]
                    curr_normalize = normalize_act[ll]

                    #Find activity given potential
                    #Don't normalize layer we're visualizing
                    if(ll != l):
                        #Normalize the potential and calculate next activity
                        if(curr_normalize):
                            curr_pot = (curr_pot/act_weight[ll]) * (self.model["pot_std"][ll] + 1e-8) + self.model["pot_mean"][ll]
                        else:
                            curr_pot = curr_pot/act_weight[ll]
                        curr_act = tf.nn.relu(curr_pot - curr_l1_weight)

                    #Reshape if needed (fc -> conv layer)
                    if("sc_conv" == curr_layer_type):
                        input_shape = curr_act.get_shape().as_list()
                        if(not switch_conv):
                            switch_conv = True
                            input_shape = self.model["output"][ll].get_shape().as_list()
                            input_shape[0] = recon_l_num_dict
                            curr_act = tf.reshape(curr_act, input_shape)

                    #Reconstruct given the activity
                    if("sc_fc" == curr_layer_type):
                        curr_pot = tf.matmul(curr_act, curr_dict, transpose_b=True)
                    else:
                        if(recon_l_fc):
                            if(ll == 0):
                                output_shape = inputNode.get_shape().as_list()
                            else:
                                output_shape = self.model["output"][ll-1].get_shape().as_list()
                            output_shape[0] = recon_l_num_dict
                        else:
                            num_x = input_shape[1]
                            num_out_x = curr_patch_size + ((num_x-1) * curr_stride)
                            if(ll == 0):
                                output_features = inputNode.get_shape().as_list()[-1]
                            else:
                                output_features = self.model["output"][ll-1].get_shape().as_list()[-1]
                            output_shape = [recon_l_num_dict, num_out_x, output_features]
                        if(recon_l_fc):
                            padding='SAME'
                        else:
                            padding='VALID'

                        curr_pot = tf.contrib.nn.conv1d_transpose(curr_act, curr_dict, output_shape, curr_stride, padding=padding)


                layer_weights.append(curr_pot)
            self.model["layer_weights"] = layer_weights

    def reset(self, sess):
        #Reset states
        sess.run(self.reset_opt)
        sess.run(self.reset_potential)
        sess.run(self.calc_activation)

    def step(self, sess, feed_dict, verbose, step, verbose_period=10):
        if(verbose):
            if((step+1) % verbose_period == 0):
                #act_norm= [np.mean(a) for a in act_norm]
                [recon_err, output_norm, act_max, nnz] = sess.run(
                        [self.model["recon_error"], self.model["output_norm"], self.model["act_max"], self.model["nnz"]], feed_dict=feed_dict)

                output_norm = [np.mean(a) for a in output_norm]
                #pot_std = [np.mean(a) for a in pot_std]

                outstr = ""
                outstr += "%3d"%(step+1) + ": \trecon_error ["
                for num in recon_err:
                    if(num is None):
                        outstr += "  None "
                    else:
                        outstr += "%7.2f"%num + ", "
                outstr += "] \tnnz ["
                for num in nnz:
                    outstr += "%4.4f"%num + ", "
                outstr += "] \toutput_norm["
                for num in output_norm:
                    outstr += "%7.5f"%num + ", "
                outstr += "] \tact_max["
                for num in act_max:
                    outstr += "%7.5f"%num + ", "
                outstr += "]"

                #print("%3f"%step, ": \trecon_error", recon_err, "\tl1_sparsity", l1_sparsity, "\tloss", loss, "\tnnz", nnz)
                print(outstr)
        sess.run(self.train_step, feed_dict=feed_dict)
        sess.run(self.calc_activation)

    def calcActivations(self, sess, feed_dict, max_iterations=400, verbose=False):
        self.reset(sess)
        #TODO stopping criteria
        for it in range(max_iterations):
            self.step(sess, feed_dict, verbose, it)

    def updateDict(self, sess, feed_dict):
        sess.run(self.update_D, feed_dict=feed_dict)
        sess.run(self.normalize_D)


