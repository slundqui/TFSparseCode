import pdb
import numpy as np
import tensorflow as tf
import os
from .utils import sparse_weight_variable, weight_variable, node_variable, conv2d, conv2d_oneToMany, convertToSparse4d, save_sparse_csr
#import matplotlib.pyplot as plt
#from pvtools import writepvpfile

class base(object):
    #Global timestep
    timestep = np.int64(0)
    plotTimestep = 0

    #Sets dictionary of params to member variables
    def loadParams(self, params):
        #Initialize tf parameters here
        self.outDir = params['outDir']
        self.runDir = self.outDir + params['runDir']
        self.ckptDir = self.runDir + params['ckptDir']
        self.plotDir = self.runDir + params['plotDir']
        self.tfDir = self.runDir + params['tfDir']
        self.saveFile = self.ckptDir + params['saveFile']
        self.load = params['load']
        self.loadFile = params['loadFile']
        self.numIterations= params['numIterations']
        self.displayPeriod = params['displayPeriod']
        self.savePeriod = params['savePeriod']
        self.plotReconPeriod = params['plotReconPeriod']
        self.plotWeightPeriod = params['plotWeightPeriod']
        self.device = params['device']
        self.batchSize = params['batchSize']
        self.writeStep = params['writeStep']
        self.progress = params['progress']

    #Make approperiate directories if they don't exist
    def makeDirs(self):
        if not os.path.exists(self.runDir):
           os.makedirs(self.runDir)
        if not os.path.exists(self.plotDir):
           os.makedirs(self.plotDir)
        if not os.path.exists(self.ckptDir):
           os.makedirs(self.ckptDir)

    #Constructor takes inputShape, which is a 3 tuple (ny, nx, nf) based on the size of the image being fed in
    def __init__(self, params, dataObj):
        self.loadParams(params)
        self.makeDirs()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        config.allow_soft_placement=True
        self.sess = tf.Session(config=config)
        self.dataObj = dataObj
        self.inputShape = self.dataObj.inputShape
        self.buildModel(self.dataObj.inputShape)
        self.initialize()
        self.writeSummary()

    #Allocates and specifies the output directory for tensorboard summaries
    def writeSummary(self):
        self.mergedSummary = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(self.tfDir + "/train", self.sess.graph)
        self.test_writer = tf.summary.FileWriter(self.tfDir + "/test")

    def initialize(self):
        ##Define saver
        load_v = self.getLoadVars()
        ##Load specific variables, save all variables
        self.loader = tf.train.Saver(var_list=load_v)
        self.saver = tf.train.Saver()

        #Initialize
        self.initSess()
        #Load checkpoint if flag set
        if(self.load):
           self.loadModel()

    def getLoadVars(self):
        return tf.global_variables()

    #Initializes session.
    def initSess(self):
        self.sess.run(tf.global_variables_initializer())

    def closeSess(self):
        self.sess.close()

    #Loads a tf checkpoint
    def loadModel(self):
        self.loader.restore(self.sess, self.loadFile)
        print("Model %s loaded" % self.loadFile)


