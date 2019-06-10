import threading

#TODO threading uses GIL, use processes instead

#We define a factory function that creates a class which inherites the class
#passed in as a parameter. This way, any member variables and methods defined
#by the parent class (passed into this factory function) can be called directly
#from the new multithreaded output class

#cls is the object reference, input_batchSize is the batchSize that will be
#called from getData, *args is the list of arguments for the constructor of cls
def mtWrapper(cls, input_batchSize):
    #Note, cls must have a "getData(self, batchSize)" method

    #Inherite from input class, we overload getData
    class mtDataObj(cls):
        def __loadData__(self, batchSize):
            #Explicitly call parent class getData
            self.loadBuf = super().getData(batchSize)

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._input_batchSize = input_batchSize
            #Start first thread
            self.loadThread = threading.Thread(target=self.__loadData__, args=(self._input_batchSize,))
            self.loadThread.setDaemon(True)
            self.loadThread.start()

        #This function doesn't actually need numExample , but this api matches that of
        #cls. So all we do here is assert numExample is the same
        #TODO use threadpool
        #TODO use multiple threads
        def getData(self, numExample):
            assert(numExample == self._input_batchSize)
            #Block loadThread here
            self.loadThread.join()
            #Store loaded data into local variable
            #This should copy, not access reference
            returnBuf = self.loadBuf[:]
            #Launch new thread to load new buffer
            self.loadThread = threading.Thread(target=self.__loadData__, args=(self._input_batchSize,))
            self.loadThread.setDaemon(True)
            self.loadThread.start()
            #Return stored buffer
            return returnBuf

    return mtDataObj
