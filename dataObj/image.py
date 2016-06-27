import scipy.io as spio
from scipy.ndimage import imread
from scipy.misc import imresize
import numpy as np
import matplotlib.pyplot as plt
import pdb
import random

def readList(filename):
    f = open(filename, 'r')
    allLines = f.readlines()
    f.close()
    #Remove newlines from all lines
    return [line[:-1] for line in allLines]


"""
An object that handles data input
"""
class imageObj:
    imgIdx = 0
    #inputShape = (32, 32, 3)
    maxDim = 0

    #Constructor takes a text file containing a list of absolute filenames
    #Will calculate the mean/std of image for normalization
    #resizeMethod takes 3 different types of input:
    #"crop" will resize the smallest dimension to inputShape,
    #and crop the other dimension in the center
    #"pad" will resize the largest dimension to inputShape, and pad the other dimension
    #"max" will find the max dimension of the list of images, and pad the surrounding area
    #Additionally, if inMaxDim is set with resizeMethod of "max", it will explicitly set
    #the max dimension to inMaxDim
    def __init__(self, imgList, resizeMethod="crop", shuffle=True, skip=1, seed=None):
        self.resizeMethod=resizeMethod
        self.imgFiles = readList(imgList)
        self.numImages = len(self.imgFiles)
        self.shuffleIdx = range(self.numImages)
        self.skip = skip
        if(shuffle):
            #Initialize random seed
            if(seed):
                #Seed random
                random.seed(seed)
            random.shuffle(self.shuffleIdx)
        #This function will also set self.maxDim
        #self.getMeanVar()
        if(self.resizeMethod=="crop"):
            pass
        elif(self.resizeMethod=="pad"):
            pass
        elif(self.resizeMethod=="max"):
            #self.inputShape=(self.maxDim, self.maxDim, 3)
            print "Resize method max Not implemented"
            assert(0)
        else:
            print "Method ", resizeMethod, "not supported"
            assert(0)

    ##Explicitly sets the mean and standard deviation for normalization
    #def setMeanVar(self, inMean, inStd):
    #    self.mean = inMean
    #    self.std = inStd

    ##Explicitly sets the max dim for the input shape.
    #def setMaxDim(self, inMaxDim):
    #    if(self.maxDim > inMaxDim):
    #        print "Error, input maxDim (", inMaxDim, ") is smaller than the biggest dimension in input images (", self.maxDim, ")"
    #        assert(0)
    #    self.maxDim = inMaxDim
    #    self.inputShape=(self.maxDim, self.maxDim, 3)

    ##Calculates the mean and standard deviation from the images
    ##Will also calculate the max dimension of image
    #def getMeanVar(self):
    #    s = 0
    #    num = 0
    #    for f in self.imgFiles:
    #        img = (imread(f).astype(np.float32)/256)
    #        [ny, nx, nf] = img.shape
    #        if(ny > self.maxDim):
    #            self.maxDim = ny
    #        if(nx > self.maxDim):
    #            self.maxDim = nx
    #        s += np.sum(img)
    #        num += img.size
    #    self.mean = s / num
    #    print "img mean: ", self.mean
    #    ss = 0
    #    for f in self.imgFiles:
    #        img = (imread(f).astype(np.float32)/256)
    #        ss += np.sum(np.power(img-self.mean, 2))
    #    self.std = np.sqrt(float(ss)/num)
    #    print "depth std: ", self.std
    #    print "maxDim: ", self.maxDim

    #Function to resize image to inputShape
    def resizeImage(self, inImage):
        (ny, nx, nf) = inImage.shape
        if(self.resizeMethod == "crop"):
            if(ny > nx):
                #Get percentage of scale
                scale = float(self.inputShape[1])/nx
                targetNy = int(round(ny * scale))
                scaleImage = imresize(inImage, (targetNy, self.inputShape[1]))
                cropTop = (targetNy-self.inputShape[0])/2
                outImage = scaleImage[cropTop:cropTop+self.inputShape[0], :, :]
            elif(ny <= nx):
                #Get percentage of scale
                scale = float(self.inputShape[0])/ny
                targetNx = int(round(nx * scale))
                scaleImage = imresize(inImage, (self.inputShape[0], targetNx))
                cropLeft = (targetNx-self.inputShape[1])/2
                outImage = scaleImage[:, cropLeft:cropLeft+self.inputShape[1], :]
        elif(self.resizeMethod == "pad"):
            if(ny > nx):
                #Get percentage of scale
                scale = float(self.inputShape[0])/ny
                targetNx = int(round(nx * scale))
                scaleImage = imresize(inImage, (self.inputShape[0], targetNx))
                padLeft = (self.inputShape[1]-targetNx)/2
                padRight = self.inputShape[1] - (padLeft + targetNx)
                outImage = np.pad(scaleImage, ((0, 0), (padLeft, padRight), (0, 0)), 'constant')
            elif(ny <= nx):
                #Get percentage of scale
                scale = float(self.inputShape[1])/nx
                targetNy = int(round(ny * scale))
                scaleImage = imresize(inImage, (targetNy, self.inputShape[1]))
                padTop = (self.inputShape[0]-targetNy)/2
                padBot = self.inputShape[0] - (padTop + targetNy)
                outImage = np.pad(scaleImage, ((padTop, padBot), (0, 0), (0, 0)), 'constant')
        elif(self.resizeMethod=="max"):
            #We pad entire image with 0
            assert(ny <= self.inputShape[0])
            assert(nx <= self.inputShape[1])
            padTop   = (self.inputShape[0]-ny)/2
            padBot   = self.inputShape[0]-(padTop+ny)
            padLeft  = (self.inputShape[1]-nx)/2
            padRight = self.inputShape[1]-(padLeft+nx)
            outImage = np.pad(inImage, ((padTop, padBot), (padLeft, padRight), (0, 0)), 'constant')
        else:
            print "Method ", resizeMethod, "not supported"
            assert(0)
        return outImage

    #Reads image provided in the argument, resizes, and normalizes image
    #Returns the image
    def readImage(self, filename):
        image = imread(filename)
        image = (self.resizeImage(image).astype(np.float32)/255)
        image = (image-np.mean(image))/np.std(image)
        #gt = np.zeros((10))
        #s = filename.split('/')[-2]
        #gt[int(s)] = 1
        return image

    #Grabs the next image in the list. Will shuffle images when rewinding
    #Num frames define how many in the clip it will grab
    def nextImage(self, numFrames = 1):
        assert(numFrames >= 1)
        startIdx = self.shuffleIdx[self.imgIdx]
        if(numFrames == 1):
            imgFile = self.imgFiles[startIdx]
            outImg = self.readImage(imgFile)
        else:
            outImg = np.zeros((numFrames, self.inputShape[0], self.inputShape[1], self.inputShape[2]))
            for f in range(numFrames):
                imgFile = self.imgFiles[startIdx+f]
                outImg[f, :, :, :] = self.readImage(imgFile)
        #Update imgIdx
        self.imgIdx = self.imgIdx + self.skip

        if(self.imgIdx >= self.numImages):
            print "Rewinding"
            self.imgIdx = 0
            shuffle(range(self.numImages))
        return outImg

    ##Get all segments of current image. This is what evaluation calls for testing
    #def allImages(self):
    #    outData = np.zeros((self.numImages, self.inputShape[0], self.inputShape[1], self.inputShape[2]))
    #    #outGt = np.zeros((self.numImages, 10))
    #    for i, imgFile in enumerate(self.imgFiles):
    #        data = self.readImage(imgFile)
    #        outData[i, :, :, :] = data
    #        #outGt[i, :] = data[1]
    #    return outData

    #Gets numExample images and stores it into an outer dimension.
    #This is what TF object calls to get images for training
    def getData(self, numExample, numFrames=1):
        assert(numFrames >= 1)
        if(numFrames == 1):
            outData = np.zeros((numExample, self.inputShape[0], self.inputShape[1], self.inputShape[2]))
        else:
            outData = np.zeros((numExample, numFrames, self.inputShape[0], self.inputShape[1], self.inputShape[2]))
        #outGt = np.zeros((numExample, 10))
        for i in range(numExample):
            data = self.nextImage(numFrames)
            if(numFrames == 1):
                outData[i, :, :, :] = data
            else:
                outData[i, :, :, :, :] = data
            #outGt[i, :] = data[1]
        return outData

class cifarObj(imageObj):
    inputShape = (32, 32, 3)

class imageNetObj(imageObj):
    #inputShape = (64, 128, 3)
    inputShape = (64, 64, 3)
