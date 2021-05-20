import numpy as np
from sklearn.neighbors import KNeighborsClassifier
DATA_PATH = "./None"

data = np.loadtxt(DATA_PATH, delimiter=",")

x, y_3weeksm, y_5weeks = data[:,15], data[:,15], data[:, 16]


# Create KNN classifier
knn = KNeighborsClassifier(n_neighbors = 3)
# Fit the classifier to the data
knn.fit(X_train,y_train)
