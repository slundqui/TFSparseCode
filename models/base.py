import pdb
import numpy as np
import tensorflow as tf
import os
import subprocess
import json
import time
import inspect

class base(object):
    #Constructor takes inputShape, which is a 3 tuple (ny, nx, nf) based on the size of the image being fed in
    def __init__(self, params):
        #Global timestep
        #self.timestep = 0
        self.timestep = tf.Variable(0, trainable=False, name='global_step')
        #Incrementer for timestep
        self.update_timestep = tf.assign_add(self.timestep, 1)

        #For storing model tensors
        self.imageDict = {}
        self.scalarDict = {}
        self.varDict = {}

        self.params = params
        self.tf_dir = self.params.run_dir + "/tfout/"
        self.ckpt_dir = self.params.run_dir + "/checkpoints/"
        self.save_file = self.ckpt_dir + "/save-model"
        self.plot_dir = self.params.run_dir + "/plots/"
        self.train_plot_dir = self.plot_dir + "/train/"
        self.test_plot_dir = self.plot_dir + "/test/"
        self.makeDirs()

        self.printParamsStr(params)
        self.printRepoDiffStr()

        #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.gpuPercent)
        #config = tf.ConfigProto(gpu_options=gpu_options)

        #If device is set to cpu, set gpu cout to 0
        config = tf.ConfigProto()

        if(self.params.device[1:4] == "cpu"):
            config.device_count['GPU'] = 0

        config.gpu_options.allow_growth=True
        #config.gpu_options.per_process_gpu_memory_fraction=self.gpuPercent
        config.allow_soft_placement=True
        self.sess = tf.Session(config=config)

        self.buildModel()
        self.buildSummaries()
        self.initialize()
        self.writeSummary()

    def printRepoDiffStr(self):
        repolabel = subprocess.check_output(["git", "log", "-n1"])
        diffStr = subprocess.check_output(["git", "diff", "HEAD"])
        outstr = repolabel.decode() + "\n\n" + diffStr.decode()

        outfile = self.params.run_dir+"/repo.diff"

        #Will replace current file if found
        f = open(outfile, 'w')
        f.write(outstr)
        f.close()

    def genParamsStr(self, params):
        param_dict = {i:getattr(params, i) for i in dir(params) if not inspect.ismethod(i) and "__" not in i}
        json_dict = {}
        for k in param_dict:
            try:
                json.dumps(param_dict[k])
                json_dict[k] = param_dict[k]
            except:
                json_dict[k] = "not_serializable"

        paramsStr = json.dumps(json_dict, indent=2)
        return paramsStr

    def printParamsStr(self, params):
        outstr = self.genParamsStr(params)
        outfile = self.params.run_dir+"/params.json"
        #Will replace current file if found
        f = open(outfile, 'w')
        f.write(outstr)
        f.close()

    def makeDir(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    #Make approperiate directories if they don't exist
    def makeDirs(self):
        self.makeDir(self.params.run_dir)
        self.makeDir(self.plot_dir)
        self.makeDir(self.train_plot_dir)
        self.makeDir(self.test_plot_dir)
        self.makeDir(self.ckpt_dir)

    def trainStepInit(self, train_feed_dict):
        #Default is do nothing
        pass

    def testStepInit(self, test_feed_dict):
        #Default is do nothing
        pass

    def trainModel(self, dataObj):
        progress_time = time.time()
        timestep = tf.train.global_step(self.sess, self.timestep)
        while timestep <= self.params.num_steps:
            timestep = tf.train.global_step(self.sess, self.timestep)

            #Progress print
            if(self.params.progress > 0 and timestep % self.params.progress == 0):
                tmp_time = time.time()
                print("Timestep ", timestep, ":", float(self.params.progress)/(tmp_time - progress_time), " iterations per second")
                progress_time = tmp_time

            need_plot = False

            if(self.params.plot_period > 0 and timestep % self.params.plot_period == 0):
                need_plot = True

            #####Training
            #Generate training feed_dict
            train_feed_dict = self.getTrainFeedDict(dataObj)
            self.trainStepInit(train_feed_dict)

            if(self.params.save_period > 0 and timestep % self.params.save_period == 0):
                #Save meta graph if beginning of run
                if(timestep == 0):
                    write_meta_graph = True
                else:
                    write_meta_graph = False
                save_path = self.saver.save(self.sess, self.save_file, global_step=self.timestep, write_meta_graph=write_meta_graph)

                print("Model saved in file: %s" % save_path)

            if(timestep%self.params.write_step == 0):
                self.writeTrainSummary(train_feed_dict)

            if(need_plot):
                fn_prefix = self.train_plot_dir + "/step_" + str(timestep) + "/"
                self.makeDir(fn_prefix)
                self.plot(timestep, train_feed_dict, fn_prefix, is_train=True)

            self.trainStep(timestep, train_feed_dict)


            #####Testing
            need_test_eval = False
            if(self.params.eval_period > 0 and timestep % self.params.eval_period == 0):
                need_test_eval = True

            #If either evaluating or plotting, need to initialize test step
            if(need_test_eval or need_plot):
                #TODO handle test data from other dataObj
                test_feed_dict = self.getTestFeedDict(dataObj)
                self.testStepInit(test_feed_dict)

            if(need_test_eval):
                #Evaluate test frame, providing gt so that it writes to summary
                self.evalModelSummary(test_feed_dict)
                print("Done test eval")

            if(need_plot):
                fn_prefix = self.test_plot_dir + "/step_" + str(timestep) + "/"
                self.makeDir(fn_prefix)
                self.plot(timestep, test_feed_dict, fn_prefix, is_train=False)


            #self.timestep+=1
            self.sess.run(self.update_timestep)

    def plot(self, step, train_feed_dict, fn_prefix, is_train):
        #Subclass should overwrite this
        pass

    def trainStep(self, step, dataObj):
        #Subclass must overwrite this
        assert(False)

    def buildModel(self):
        print("Cannot call base class buildModel")
        assert(0)

    #def addImageSummary(self, name, tensor, normalize=True):
    #    assert(len(tensor.get_shape()) == 4)
    #    self.imageDict[name] = (tensor, normalize)

    def buildSummaries(self):
        ##Summaries ops

        #TODO fix this
        ##Write all images as a grid
        #with tf.name_scope("Summary"):
        #    opsList = []
        #    opsList_test = []
        #    gridList = {}
        #    gridList_test = {}

        #    for key in self.imageDict.keys():
        #        (tensor, normalize) = self.imageDict[key]
        #        (grid_image, grid_op) = createImageBuf(tensor, key+"_grid")
        #        (grid_image_test, grid_op_test) = createImageBuf(tensor, key+"_grid_test")

        #        gridList[key] = (grid_image, normalize)
        #        gridList_test[key] = (grid_image_test, normalize)

        #        opsList.append(grid_op)
        #        opsList_test.append(grid_op_test)
        #    if(len(opsList)):
        #        self.updateImgOp = tf.tuple(opsList)
        #    else:
        #        self.updateImgOp = tf.no_op()
        #    if(len(opsList_test)):
        #        self.updateImgOp_test = tf.tuple(opsList_test)
        #    else:
        #        self.updateImgOp_test = tf.no_op()

        trainSummaries = []
        testSummaries = []
        bothSummaries = []
        #for key in gridList.keys():
        for key in self.imageDict.keys():
            #(tensor, normalize) = gridList[key]
            #(test_tensor, test_normalize) = gridList_test[key]
            #assert(test_normalize == normalize)
            #Create images per item in imageDict
            #trainSummaries.append(tf.summary.image(key+"_grid_train", normImage(tensor, normalize)))
            #testSummaries.append(tf.summary.image(key+"_grid_test", normImage(test_tensor, test_normalize)))
            image = self.imageDict[key]
            max_image = tf.reduce_max(image, axis=[1, 2, 3], keepdims=True)
            min_image = tf.reduce_min(image, axis=[1, 2, 3], keepdims=True)
            norm_image = (image + min_image)/(max_image - min_image)
            bothSummaries.append(tf.summary.image(key, norm_image))
            bothSummaries.append(tf.summary.histogram(key, image))

        #Output tensorboard summaries
        for key in self.scalarDict.keys():
            bothSummaries.append(tf.summary.scalar(key,  self.scalarDict[key]))

        #Generate histograms for all inputs in varDict
        for key in self.varDict.keys():
            bothSummaries.append(tf.summary.histogram(key, self.varDict[key]))

        #Merge summaries
        trainSummaries.extend(bothSummaries)
        testSummaries.extend(bothSummaries)
        if(len(trainSummaries) > 0):
            self.mergeTrainSummary = tf.summary.merge(trainSummaries)
        else:
            self.mergeTrainSummary = None
        if(len(testSummaries) > 0):
            self.mergeTestSummary = tf.summary.merge(testSummaries)
        else:
            self.mergeTestSummary = None

    def getTrainFeedDict(self, dataObj):
        #Subclass must overwrite this
        assert(0)

    def getTestFeedDict(self, dataObj):
        assert(0)

    def writeTrainSummary(self, train_feed_dict):
        if(self.mergeTrainSummary is not None):
            trainSummary = self.sess.run(self.mergeTrainSummary, feed_dict=train_feed_dict)
            timestep = tf.train.global_step(self.sess, self.timestep)
            self.train_writer.add_summary(trainSummary, timestep)

    def writeTestSummary(self, feed_dict):
        if(self.mergeTestSummary is not None):
            testSummary = self.sess.run(self.mergeTestSummary, feed_dict=feed_dict)
            timestep = tf.train.global_step(self.sess, self.timestep)
            self.test_writer.add_summary(testSummary, timestep)

    def getLoadVars(self):
        #vv = [v for v in tf.global_variables() if 'Adam' not in v.name]
        #return vv
        return tf.global_variables()

    def initialize(self):
        with tf.name_scope("Savers"):
            ##Define saver
            load_v = self.getLoadVars()
            ##Load specific variables, save all variables
            self.loader = tf.train.Saver(var_list=load_v, allow_empty=False)
            self.saver = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=2, allow_empty=True)

            #Initialize
            self.initSess()
            #Load checkpoint if flag set
            if(self.params.load):
                self.loadModel()

    def guarantee_initialized_variables(self, session, list_of_variables = None):
        if list_of_variables is None:
            list_of_variables = tf.all_variables()
        uninitialized_variables = list(tf.get_variable(name) for name in
                session.run(tf.report_uninitialized_variables(list_of_variables)))
        session.run(tf.initialize_variables(uninitialized_variables))
        return uninitialized_variables

    #Initializes session.
    def initSess(self):
        self.sess.run(tf.global_variables_initializer())

    #Allocates and specifies the output directory for tensorboard summaries
    def writeSummary(self):
        self.mergedSummary = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(self.tf_dir + "/train", self.sess.graph)
        self.test_writer = tf.summary.FileWriter(self.tf_dir + "/test")

    def closeSess(self):
        self.sess.close()
        tf.reset_default_graph()

    #This function should evalulate a given feed_dict
    def evalModel(self, feed_dict):
        assert(False)

    def getEvalFeedDict(self, data):
        assert(False)

    def evalSet(self, allData):
        numExamples = allData.shape[0]

        steps_per_epoch = int(np.ceil(numExamples / self.params.batch_size))
        out = []
        for test_step in range(steps_per_epoch):
            print(test_step,"out of", steps_per_epoch)
            start_idx = test_step*self.params.batch_size
            end_idx = start_idx + self.params.batch_size
            if(end_idx > numExamples):
                diff = int(end_idx - numExamples)
                end_idx = numExamples
                data = allData[start_idx:end_idx]
                data_shape = data.shape
                data = np.concatenate([data, np.zeros((diff,) + data_shape[1:])], axis=0)
                s_out = self.evalModel(self.getEvalFeedDict(data))
                s_out = s_out[:-diff]
            else:
                data = allData[start_idx:end_idx]
                s_out = self.evalModel(self.getEvalFeedDict(data))
            out.append(s_out)

        return np.concatenate(out, axis=0)

    def evalModelSummary(self, test_feed_dict):
        self.writeTestSummary(test_feed_dict)

    #Loads a tf checkpoint
    def loadModel(self):
        self.loader.restore(self.sess, self.params.load_file)
        #self.guarantee_initialized_variables(self.sess)

        print("Model %s loaded" % self.params.load_file)





