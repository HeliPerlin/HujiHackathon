import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.semi_supervised.tests.test_self_training

DATA_PATH = "./None"

data = np.loadtxt(DATA_PATH, delimiter=",")

x, y_3weeksm, y_5weeks = data[:,15], data[:,15], data[:, 16]
X_train_3, X_test_3, y_train_3, y_test_3 = train_test_split(X, y_3weeks, test_size=0.2)
X_train_5, X_test_5, y_train_5, y_test_5 = train_test_split(X, y_5weeks, test_size=0.2)



knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train_3,y_train_3)
y_predicted = knn.predict(X_test_3)

score(y_test_3, y_predicted)


clf = RandomForestClassifier(n_estimators = 100)
clf.fit(X_train_3, y_train_3)
y_pred = clf.predict(X_test_3)

score(y_test_3, y_predicted)
