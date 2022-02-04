import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

from sklearn.model_selection import train_test_split

#Load dataset
iris = load_iris()

#############################################################################
#Decision Tree Classifier
#############################################################################

#number of classes
classes = 3
#color scheme
colors = 'cmb'
#marker scheme
markers = 'ovs'
#plotting resolution
res = 0.01


#Plotting Decision Tree
plt.figure(figsize=(20,10))

models = []

#Multiple pair feature representation
for pair_index, pair in enumerate([[0, 1], [0, 2], [0, 3], 
                                           [1, 2], [1, 3], 
                                                   [2, 3] ]):

    X, y = iris.data[:, pair] , iris.target
    model = DecisionTreeClassifier(min_impurity_decrease = 0.01).fit(X, y)
    models.append(model)
 
    plt.subplot(2, 3, pair_index + 1)
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, 
                                   x_max, 
                                   res),
                         np.arange(y_min, 
                                   y_max, 
                                   res) )
    plt.tight_layout(h_pad = 0.5, 
                     w_pad = 0.5, 
                       pad = 4.0 )
    
    #Performing Prediction
    Z = model.predict(np.c_[xx.ravel(), 
                            yy.ravel() ])

    Z = Z.reshape(xx.shape)

    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.brg)
    
    plt.xlabel(iris.feature_names[pair[0]].title()[0:-4] + iris.feature_names[pair[0]][-4:])
    plt.ylabel(iris.feature_names[pair[1]].title()[0:-4] + iris.feature_names[pair[1]][-4:])
    
    # Plot the training points for each species
    for i, color, marker in zip(range(classes), colors, markers):
        index = np.where(y == i)
        plt.scatter(X[index, 0], 
                    X[index, 1], 
                    c = color,
                    marker = marker,
                    label = iris.target_names[i],
                    cmap = plt.cm.brg, 
                    edgecolor = 'black', 
                    s = 15                       )

plt.legend(loc = 'lower right',
           fontsize = 16,
           borderpad = 0.1, 
           handletextpad = 0.1 )

plt.axis("tight")

plt.show()



# Apply the decision tree classifier model to the data using all four parameters at once.
model_all_params = DecisionTreeClassifier(min_impurity_decrease = 0.01).fit(iris.data, iris.target)
# Prepare a plot figure with set size.
plt.figure(figsize = (20,10))
# Plot the decision tree, showing the decisive values and the improvements in Gini impurity along the way.
plot_tree(model_all_params, rounded = True, 
          filled=True      )
# Display the tree plot figure.
plt.show()


#############################################################################
#Random Forrest Classifier
#############################################################################

import pandas as pd
data=pd.DataFrame({
    'sepal length':iris.data[:,0],
    'sepal width':iris.data[:,1],
    'petal length':iris.data[:,2],
    'petal width':iris.data[:,3],
    'species':iris.target
})

#Train Test Split
X=data[['sepal length', 'sepal width', 'petal length', 'petal width']]
y=data['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) 

#Random Forrest Classifier (Gaussian)
clf=RandomForestClassifier(n_estimators=100)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
print("Random Forrest Classifier Accuracy:",metrics.accuracy_score(y_test, y_pred))
