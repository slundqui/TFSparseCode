import tensorflow as tf
import models.utils as utils
import numpy as np
import pdb

class lcaSC(object):

    #TODO build conv discriminator here?
    def discriminator(self, X, hsize=[512, 256],reuse=False):
        with tf.variable_scope("GAN/Discriminator",reuse=reuse):
            h1 = tf.layers.dense(X,hsize[0],activation=tf.nn.leaky_relu)
            h2 = tf.layers.dense(h1,hsize[1],activation=tf.nn.leaky_relu)
            h3 = tf.layers.dense(h2,2)
            out = tf.layers.dense(h3,1)
        return out, h3

    def __init__(self, inputNode, l1_weight, dict_size, sc_lr, dict_lr, layer_type=None, patch_size=None, stride=None, mask=None, optimizer=tf.train.AdamOptimizer, gan=False, gan_weight=None):
        #Model variables and outputs
        self.model = {}
        self.model["input"] = inputNode
        self.gan = gan

        self.recon_built = False

        assert(layer_type is not None)

        with tf.name_scope("lca_layer"):
            input_shape = inputNode.get_shape().as_list()

            if(len(input_shape) == 3):
                [batch, input_size, input_features] = input_shape
            else:
                [batch, input_features] = input_shape

            if("sc_fc" == layer_type):
                reshape_input = tf.reshape(inputNode, [batch, -1])
                input_features = reshape_input.get_shape().as_list()[1]
                D_shape = [input_features, dict_size]
                act_shape = [batch, dict_size]
            elif("sc_conv" == layer_type):
                D_shape = [patch_size, input_features, dict_size]
                assert(input_size % stride == 0)
                act_shape = [batch, input_size//stride, dict_size]
            else:
                assert(False)

            self.model["dictionary"] = utils.l2_weight_variable(D_shape, "dictionary")
            self.model["potential"]  = utils.weight_variable(act_shape, "potential", std=1e-3)
            self.model["activation"] = utils.weight_variable(act_shape, "activation", std=1e-3)

            if("sc_fc" == layer_type):
                self.model["recon"] = tf.matmul(self.model["activation"],
                    self.model["dictionary"], transpose_b=True)
            elif("sc_conv" == layer_type):
                self.model["recon"] = tf.contrib.nn.conv1d_transpose(
                   self.model["activation"], self.model["dictionary"],
                   [batch, input_size, input_features], stride, padding='SAME')
            else:
                assert(0)

            self.model["error"] = inputNode - self.model["recon"]
            self.model["recon_error"] = 0.5 * tf.reduce_sum(self.model["error"]**2)
            self.model["l1_sparsity"] = 0.5 * tf.reduce_sum(tf.abs(self.model["activation"]))
            self.model["loss"] = self.model["recon_error"] + l1_weight * self.model["l1_sparsity"]

            #Ops
            calc_act = tf.nn.relu(self.model["potential"] - l1_weight)
            self.calc_activation = self.model["activation"].assign(calc_act)

            low_init_val = .8*l1_weight
            high_init_val  = 1.1*l1_weight
            potential_init = tf.random_uniform(act_shape, low_init_val, high_init_val, dtype=tf.float32)
            self.reset_potential = self.model["potential"].assign(potential_init)

        with tf.name_scope("inject_act"):
            #TODO can we make this independent of batch size?
            self.model["inject_act_placeholder"] = tf.placeholder(tf.float32,
                shape=[batch,] + act_shape[1:], name="inject_act")

            self.model["recon_from_act"] = tf.contrib.nn.conv1d_transpose(
                self.model["inject_act_placeholder"], self.model["dictionary"],
                [batch, input_size, input_features], stride, padding='SAME')

        with tf.name_scope("gan"):
            real_logits, real_rep = self.discriminator(inputNode)
            fake_logits, fake_rep = self.discriminator(self.model["recon"], reuse=True)

            real_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=real_logits,
                labels = tf.ones_like(real_logits))
            fake_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits,
                labels = tf.zeros_like(fake_logits))
            gan_disc_loss = gan_weight * tf.reduce_sum(real_loss + fake_loss)
            gan_gen_loss = gan_weight * tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits,
                labels=tf.ones_like(fake_logits)))

        with tf.name_scope("stats"):
            #Calculate stats
            num_total_act = 1
            for s in act_shape:
                num_total_act *= s

            nnz = tf.count_nonzero(self.model["activation"]) / num_total_act

            #Calculate means/std of activations
            #Do this across batches
            #Normalize each feature/dictionary element individually

            if(len(act_shape) == 3):
                moment_reduce_axis = [0, 1]
                non_batch_reduce_axis = [1, 2]
                tile_input = [act_shape[0], act_shape[1], 1]
            elif(len(act_shape) == 2):
                moment_reduce_axis = 0
                non_batch_reduce_axis = 1
                tile_input = [act_shape[0], 1]
            else:
                assert(0)

            act_norm = tf.norm(self.model["activation"], axis=moment_reduce_axis, keepdims=True)
            act_mean, act_var = tf.nn.moments(self.model["activation"], axes=moment_reduce_axis, keep_dims=True)
            act_std = tf.sqrt(act_var)
            act_max = tf.reduce_max(self.model["activation"])

            pot_norm = tf.norm(self.model["potential"], axis=moment_reduce_axis, keepdims=True)
            pot_mean, pot_var = tf.nn.moments(self.model["potential"], axes=moment_reduce_axis, keep_dims=True)
            pot_std = tf.sqrt(pot_var)

            input_norm = tf.norm(inputNode, axis=non_batch_reduce_axis)
            recon_norm = tf.norm(self.model["recon"], axis=non_batch_reduce_axis)
            output_norm = tf.norm(self.model["activation"], axis=moment_reduce_axis)
            dict_norm = tf.norm(self.model["dictionary"], axis=moment_reduce_axis)

            self.model["nnz"] = nnz
            self.model["act_norm"] = act_norm
            self.model["act_mean"] = act_mean
            self.model["act_std"] = act_std
            self.model["act_max"] = act_max
            self.model["pot_norm"] = pot_norm
            self.model["pot_mean"] = pot_mean
            self.model["pot_std"] = pot_std
            self.model["input_norm"] = input_norm
            self.model["recon_norm"] = recon_norm
            self.model["output_norm"] = output_norm
            self.model["dict_norm"] = dict_norm
            self.model["gan_gen_loss"] = gan_gen_loss
            self.model["gan_disc_loss"] = gan_disc_loss

        with tf.name_scope("optimizer"):
            #Define optimizer
            #TODO different learning rates?
            opt = optimizer(sc_lr)

            #Calculate recon gradient wrt activation
            if(gan):
                gen_loss = self.model["recon_error"] + gan_gen_loss
            else:
                gen_loss = self.model["recon_error"]

            recon_grad = opt.compute_gradients(gen_loss, [self.model["activation"]])

            #Apply gradient (plus shrinkage) to potential
            #Needs to be a list of number of gradients, each element as a tuple of (gradient, wrt)

            (grad, var) = recon_grad[0]

            shrink_term = self.model["potential"] - self.model["activation"]
            d_potential = [(grad + shrink_term, self.model["potential"])]

            self.train_step = opt.apply_gradients(d_potential)
            print("Opt variables that get reset per image:", [v.name for v in opt.variables()])
            #Reset must be called after apply_gradients to define opt variables
            self.reset_opt = tf.group([v.initializer for v in opt.variables()])

            #Dictionary update variables
            opt_D = tf.train.AdamOptimizer(dict_lr)
            #Update d based on mean across batch
            self.update_D = opt_D.minimize((1/batch) * gen_loss, var_list=[self.model["dictionary"]])

            disc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="GAN/Discriminator")
            #opt_disc = tf.train.AdamOptimizer(dict_lr)
            opt_disc = tf.train.AdamOptimizer(1e-3)
            self.update_disc = opt_disc.minimize((1/batch) * gan_disc_loss, var_list=disc_vars)


            #Normalize D
            dict_shape = self.model["dictionary"].get_shape().as_list()
            if(len(dict_shape) == 3):
                #dict_norm = tf.norm(self.model["dictionary"], axis=(0, 1), keepdims=True)
                #Normalize individually by input features?
                #TODO hard coded here
                dict_norm = tf.norm(self.model["dictionary"], axis=0, keepdims=True)
            elif(len(dict_shape) == 2):
                dict_norm= tf.norm(self.model["dictionary"], axis=0, keepdims=True)
            else:
                assert(0)
            dict_norm = tf.maximum(tf.ones(dict_shape), dict_norm)
            self.normalize_D = self.model["dictionary"].assign(self.model["dictionary"]/dict_norm)

    def reset(self, sess):
        #Reset states
        sess.run(self.reset_opt)
        sess.run(self.reset_potential)
        sess.run(self.calc_activation)

    def step(self, sess, feed_dict, verbose, step, verbose_period=200):
        if(verbose):
            if((step+1) % verbose_period == 0):
                #TODO store this to disk
                #act_norm= [np.mean(a) for a in act_norm]
                stats_list = ["recon_error", "nnz", "input_norm", "recon_norm", "dict_norm"]
                if(self.gan):
                    stats_list += ["gan_gen_loss", "gan_disc_loss"]

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
        if(self.gan):
            sess.run(self.update_disc, feed_dict=feed_dict)

