import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from joblib import Parallel, delayed

from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import validation_curve
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV



#Load Data
train_data = pd.read_csv("train.csv") #reading the csv files using pandas
test_data = pd.read_csv("test.csv")

#rescaling the feature values
round(train_data.drop('label', axis=1).mean(), 2)

#Assigning data to variables
Y = train_data['label']
X = train_data.drop(columns = 'label')

#Normalizing Pixel Values
X = X/255.0
test_data = test_data/255.0
X_scaled = scale(X)

#Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y, test_size = 0.3, random_state = 10)


##LINEAR KERNELS (91%)
#Fit
linear_kernel = SVC(kernel='linear')
linear_kernel.fit(X_train, y_train)
# predict
y_pred = linear_kernel.predict(X_test)
#Evaluate
print("accuracy:", metrics.accuracy_score(y_true=y_test, y_pred=y_pred), "\n")

#RBF KERNELS (93%)
#Fit
rbf_kernel = SVC(kernel='rbf', C=5, gamma=0.001)
rbf_kernel.fit(X_train, y_train)
#Predict
y_pred = rbf_kernel.predict(X_test)
#Evaluate
print("accuracy:", metrics.accuracy_score(y_true=y_test, y_pred=y_pred), "\n")


#Given the higher accuracy of the rbf kernel (non-linear model), we will continue forward using this model for the optimization phase. We seek to optimize C and gamma in this case, using K-fold cross validation:
# creating a KFold object with 5 splits 

folds = KFold(n_splits = 5, shuffle = True, random_state = 10)
print("folds done")


# specify range of hyperparameters
# Set the parameters by cross-validation
hyper_params = [ {'gamma': [1e-2, 1e-3, 1e-4],
                     'C': [1,5,10,15]}]


# specify model
model = SVC(kernel="rbf")

t0 = time.time()
# set up GridSearchCV()
model_cv = GridSearchCV(estimator = model, 
                        param_grid = hyper_params, 
                        scoring= 'accuracy', n_jobs = -1, 
                        cv = folds, 
                        verbose = 2,
                        return_train_score=True)      

model_cv.fit(X_train, y_train)
t1 = time.time()
tt = (t1-t0)/60
print("Total elapsed time for Grid Search:", tt, "minutes")

#Plot Results of cross validation
cv_results = pd.DataFrame(model_cv.cv_results_)
cv_results['param_C'] = cv_results['param_C'].astype('int')

# # plotting
plt.figure(figsize=(16,8))

# subplot 1/3
plt.subplot(131)
gamma_01 = cv_results[cv_results['param_gamma']==0.01]

plt.plot(gamma_01["param_C"], gamma_01["mean_test_score"])
plt.plot(gamma_01["param_C"], gamma_01["mean_train_score"])
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.title("Gamma=0.01")
plt.ylim([0.60, 1])
plt.legend(['test accuracy', 'train accuracy'], loc='upper left')
plt.xscale('log')

# subplot 2/3
plt.subplot(132)
gamma_001 = cv_results[cv_results['param_gamma']==0.001]

plt.plot(gamma_001["param_C"], gamma_001["mean_test_score"])
plt.plot(gamma_001["param_C"], gamma_001["mean_train_score"])
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.title("Gamma=0.001")
plt.ylim([0.60, 1])
plt.legend(['test accuracy', 'train accuracy'], loc='upper left')
plt.xscale('log')


# subplot 3/3
plt.subplot(133)
gamma_0001 = cv_results[cv_results['param_gamma']==0.0001]

plt.plot(gamma_0001["param_C"], gamma_0001["mean_test_score"])
plt.plot(gamma_0001["param_C"], gamma_0001["mean_train_score"])
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.title("Gamma=0.0001")
plt.ylim([0.60, 1])
plt.legend(['test accuracy', 'train accuracy'], loc='upper left')
plt.xscale('log')

plt.show()

