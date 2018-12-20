import tensorflow as tf
import models.utils as utils
import numpy as np
import pdb

class lcaSC(object):
    def __init__(self, inputNode, l1_weight, dict_size, sc_lr, dict_lr, layer_type=None, patch_size=None, stride=None, mask=None):
        curr_input = inputNode
        #Model variables and outputs
        self.model = {}

        assert(layer_type is not None)

        with tf.name_scope("lca_layer"):
            input_shape = curr_input.get_shape().as_list()

            if(len(input_shape) == 3):
                [batch, input_size, input_features] = input_shape
            else:
                [batch, input_features] = input_shape

            if("sc_fc" == layer_type):
                curr_input = tf.reshape(curr_input, [batch, -1])
                input_features = curr_input.get_shape().as_list()[1]
                D_shape = [input_features, dict_size]
                act_shape = [batch, dict_size]
                reduce_axis = [1]
            else:
                D_shape = [patch_size, input_features, dict_size]
                assert(input_size % stride == 0)
                act_shape = [batch, input_size//stride, dict_size]
                reduce_axis = [1, 2]

            curr_dict = utils.l2_weight_variable(D_shape, "dictionary")
            curr_potential = utils.weight_variable(act_shape, "potential", std=1e-3)
            curr_activation = utils.weight_variable(act_shape, "activation", std=1e-3)

            if("sc_fc" == layer_type):
                curr_recon = tf.matmul(curr_activation, curr_dict, transpose_b=True)
            elif("sc_conv" == layer_type):
                curr_recon = tf.contrib.nn.conv1d_transpose(curr_activation, curr_dict, [batch, input_size, input_features], stride, padding='SAME')
            else:
                assert(0)

            curr_error = curr_input - curr_recon
            curr_recon_error = 0.5 * tf.reduce_mean(tf.reduce_sum(curr_error**2, axis=reduce_axis))
            curr_l1_sparsity = tf.reduce_mean(tf.reduce_sum(tf.abs(curr_activation), axis=reduce_axis))
            curr_loss = curr_recon_error + 0.5 * l1_weight * curr_l1_sparsity

            self.model["error"] = curr_error
            self.model["recon_error"] = curr_recon_error
            self.model["potential"] = curr_potential
            self.model["activation"] = curr_activation
            self.model["recon"] = curr_recon
            self.model["l1_sparsity"] = curr_l1_sparsity
            self.model["loss"] = curr_loss

            #Ops
            calc_act = tf.nn.relu(curr_potential - l1_weight)
            self.calc_activation = curr_activation.assign(calc_act)

            low_init_val = .8*l1_weight
            high_init_val  = 1.1*l1_weight
            potential_init = tf.random_uniform(act_shape, low_init_val, high_init_val, dtype=tf.float32)
            self.reset_potential = curr_potential.assign(potential_init)

            #Save all variables
            self.model["dictionary"] = curr_dict
            self.model["output"] = curr_activation
            self.model["input"] = curr_input

        with tf.name_scope("stats"):
            #Calculate stats
            num_total_act = 1
            for s in act_shape:
                num_total_act *= s

            curr_nnz = tf.count_nonzero(curr_activation) / num_total_act

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

            input_norm = tf.norm(curr_input, axis=moment_reduce_axis)
            output_norm = tf.norm(curr_activation, axis=moment_reduce_axis)

            self.model["nnz"] = curr_nnz
            self.model["act_norm"] = act_norm
            self.model["act_mean"] = act_mean
            self.model["act_std"] = act_std
            self.model["act_max"] = act_max
            self.model["pot_norm"] = pot_norm
            self.model["pot_mean"] = pot_mean
            self.model["pot_std"] = pot_std
            self.model["input_norm"] = input_norm
            self.model["output_norm"] = output_norm

        with tf.name_scope("optimizer"):
            #Define optimizer
            #TODO different learning rates?
            opt = tf.train.AdamOptimizer(sc_lr)

            #Calculate recon gradient wrt activation
            recon_grad = opt.compute_gradients(self.model["recon_error"], self.model["activation"])

            #Apply gradient (plus shrinkage) to potential
            #Needs to be a list of number of gradients, each element as a tuple of (gradient, wrt)

            (grad, var) = recon_grad[0]

            shrink_term = (1/batch) * (self.model["potential"] - self.model["activation"])
            d_potential = [(grad + shrink_term, self.model["potential"])]

            self.train_step = opt.apply_gradients(d_potential)
            #Reset must be called after apply_gradients to define opt variables
            self.reset_opt = tf.group([v.initializer for v in opt.variables()])

            #Dictionary update variables
            opt_D = tf.train.AdamOptimizer(dict_lr)
            self.update_D = opt_D.minimize(self.model["recon_error"], var_list=[self.model["dictionary"]])

            #Normalize D
            curr_dict = self.model["dictionary"]
            dict_shape = curr_dict.get_shape().as_list()
            if(len(dict_shape) == 3):
                curr_norm = tf.norm(curr_dict, axis=(0, 1))
            elif(len(dict_shape) == 2):
                curr_norm = tf.norm(curr_dict, axis=0)
            else:
                assert(0)
            #curr_norm = tf.maximum(tf.ones(dict_shape), curr_norm)
            self.normalize_D = curr_dict.assign(curr_dict/curr_norm)

    def reset(self, sess):
        #Reset states
        sess.run(self.reset_opt)
        sess.run(self.reset_potential)
        sess.run(self.calc_activation)

    def step(self, sess, feed_dict, verbose, step, verbose_period=10):
        if(verbose):
            if((step+1) % verbose_period == 0):
                #TODO store this to disk
                #act_norm= [np.mean(a) for a in act_norm]
                stats_list = ["recon_error", "nnz", "output_norm", "input_norm"]
                stats_nodes = [self.model[stat] for stat in stats_list]
                output = sess.run(stats_nodes, feed_dict=feed_dict)

                outstr = ""
                outstr += "%4d"%(step+1) + ":"
                for name, out in zip(stats_list, output):
                    ndims = len(np.array(out).shape)
                    out = np.mean(out)
                    outstr += " \t" + name + "["
                    if(out is None):
                        outstr+= " None "
                    else:
                        outstr += " %7.5f "%out
                    outstr += "]"
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


