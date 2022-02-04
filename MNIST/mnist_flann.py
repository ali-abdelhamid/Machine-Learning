from pyflann import *
import numpy as np
from numpy.random import *
from scipy import stats
import time

#Load Raw Data (NX784)
train_data = np.loadtxt("mnist_train.csv", 
                        delimiter=",")
test_data = np.loadtxt("mnist_test.csv", 
                       delimiter=",") 
#Isolate first 10K training instances and 1K testing instances
train_data = train_data[:10000]
test_data = test_data[:1000]

#Remove and store first (label) column
train_labels = train_data[:,0]
test_labels = np.array([test_data[:,0]])

dataset = train_data[:,1:]
testset = test_data[:,1:]

tic = time.time()
#Compile FLANN function to get index matrix
flann = FLANN()
result, dists = flann.nn(
    dataset, testset, 5, algorithm="kmeans", branching=32, iterations=7, checks=16)

#Find most ocurring neighbor label for each test instance
tran = np.transpose(train_labels[result[:,:]])
ypred = stats.mode(tran)

#Compare our prediction list with the test labels list
correct_pred = np.sum(ypred == test_labels)
Accuracy = (float(correct_pred)/float(1000))*100
Error = 100 - Accuracy

toc = time.time()

print("Error Rate", Error, "%")
print("Total Time: ", toc-tic)
print("#pred", correct_pred)


