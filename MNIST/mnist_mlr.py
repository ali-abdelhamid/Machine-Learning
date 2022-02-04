from pyflann import *
import numpy as np
from numpy.random import *
from scipy import stats
import time
from sklearn.model_selection import *
from sklearn.linear_model import LogisticRegression

#Load Raw Data (NX784)
train_data = np.loadtxt("mnist_train.csv", 
                        delimiter=",")
test_data = np.loadtxt("mnist_test.csv", 
                       delimiter=",") 
#Isolate first 10K training instances and 1K testing instances
train_data = train_data[:10000]
test_data = test_data[:1000]

lr = LogisticRegression(solver='saga')

tic = time.time()

lr.fit(train_data[:,1:], train_data[:,0])

predd = lr.predict(test_data[:,1:])
pred = np.array([predd])

#Remove and store first (label) column
train_labels = train_data[:,0]
test_labels = np.array([test_data[:,0]])

#dataset = train_data[:,1:]
#testset = test_data[:,1:]

#Compare our prediction list with the test labels list
correct_pred = np.sum(pred == test_labels)
Accuracy = (float(correct_pred)/float(1000))*100
Error = 100 - Accuracy

toc = time.time()

print("Error Rate: ", Error, "%")
print("Total Time: ", toc-tic)



