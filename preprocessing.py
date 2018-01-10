import numpy as np
from numpy import random
import glob
import scipy.io.wavfile

np.random.seed(4)


def preprocess(periods, testCategoryNum):
    periodList = periods
    catNum = len(periodList)

    def createpathlist():

        print("Loading file paths.")

        x = []
        y = []
        for recDirIndex in range(len(periodList)):
            rnge = periodList[recDirIndex]
            bottomrange = rnge[0]
            toprange = rnge[1]
            for i in range(bottomrange, toprange):
                recYearDirIndex = glob.glob("..//toserver//FINAL//" + str(i) + "//*.wav")
                for n in range(len(recYearDirIndex)):
                    path = recYearDirIndex[n]
                    x.append(path)
                    y.append(recDirIndex)


        #Created 2 original arrays for readability
        print("Done.")
        return np.array(x), np.array(y)



    def truncateData():
        x, y = createpathlist()

        #Least prevalent category
        originalLengths = [] 
        for n in range(catNum):
            originalLengths.append(np.count_nonzero(y == n))

        minimumNum = min(originalLengths)

        for n in range(catNum):
            while( y.tolist().count(n) > minimumNum ):
                #First occuring instance
                for q in range(y.size):
                    if y[q] == n:
                        y = np.delete(y, q)
                        x = np.delete(x, q)
                        break

        return x, y
        


    def psudoRandomOrder():

        x, y = truncateData()
        print("")
        print("Psudo-randomising Data")


        randOrder = np.random.permutation(x.shape[0])
        x, y = x[randOrder], y[randOrder]
        
        print("Shuffled.")
        return x, y
   

    def BatchSeparator():

        x, y = psudoRandomOrder()
        print("")
        print("Separating data into testing and training set.")


        x_test = []
        y_test = []


        for n in range(catNum):
            while( y_test.count(n) < testCategoryNum ):
                #first occuring instance
                for q in range(y.size):
                    if y[q] == n:
                                        
                        x_test.append(x[q])
                        x = np.delete(x, q)

                        y_test.append(y[q])
                        y = np.delete(y, q)
                        break

        x_test = np.array(y_test)
        y_test = np.array(y_test)

        x_train = x
        y_train = y

        return x_train, y_train, x_test, y_test


    x_train, y_train, x_test, y_test = BatchSeparator()

    print("Created training set of " + str(y_train.size) + " recordings and a testing set of " + str(y_test.size) + " recordings.")
    print("Preproccessing complete.")
    return x_train, y_train, x_test, y_test