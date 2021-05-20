import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from sklearn.metrics import plot_confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# load data
DATA_PATH = "./None"
data = np.loadtxt(DATA_PATH, delimiter=",")
# separate X and y - X is the features and y is the result
X, y_3weeks, y_5weeks = data[:, :15], data[:, 15], data[:,16]
X_train_3, X_test_3, y_train_3, y_test_3 = train_test_split(X, y_3weeks, test_size=0.2)
X_train_5, X_test_5, y_train_5, y_test_5 = train_test_split(X, y_5weeks, test_size=0.2)
all_data_sets = [[X_train_3, X_test_3, y_train_3, y_test_3],
                 [X_train_5, X_test_5, y_train_5, y_test_5]]


# cnn model
cnn_model = tf.keras.models.Sequential()
cnn_model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
cnn_model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
cnn_model.add(tf.keras.layers.Dense(128, activation=tf.nn.softmax))
# TODO: maybe change the loss to MSE
cnn_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# knn model
knn = KNeighborsClassifier(n_neighbors = 3)

# random forest model
clf = RandomForestClassifier(n_estimators = 100)


"""
    TESTING
"""
list_of_models = [cnn_model, knn, clf]
for model_index in range(len(list_of_models)):
    for data_set in all_data_sets:
        model = list_of_models[model_index]
        if model_index == 0:
            model.fit(x=data_set[0], y=data_set[2], batch_size=60,
                      verbose=1, epochs=200, validation_split=0.25)
            predicted_y = model.predict(data_set[1])
            tf.math.confusion_matrix(data_set[3], predicted_y)
            plot_confusion_matrix(model, data_set[1], data_set[3],
                                  cmap=plt.cm.get_cmap("Blues"),
                                  normalize=True)
            plt.show()
        else:
            model.fit(data_set[0], data_set[2])
            predicted_y = model.predict(data_set[1])
            plot_confusion_matrix(model, data_set[1], data_set[3],
                                  cmap=plt.cm.get_cmap("Blues"), normalize=True)
            plt.show()



# cnn_model_log_3 = cnn_model.fit(x=X_train_3, y=y_train_3, batch_size=60,
#                                 verbose=1, epochs=200, validation_split=0.25)
# predicted_y = cnn_model.predict(X_test_3)
# tf.math.confusion_matrix(y_test_3, predicted_y)
# test_loss, test_acc = cnn_model.evaluate(x=X_test_3, y=y_test_3)
# print("Accuracy: {}, Loss: {} \n".format(test_acc, test_loss))
# cnn_model_log_5 = cnn_model.fit(x=X_train_5, y=y_train_5, batch_size=60,
#                                 verbose=1, epochs=200, validation_split=0.25)
# predicted_y = cnn_model.predict(X_test_5)
# tf.math.confusion_matrix(y_test_5, predicted_y)
# test_loss, test_acc = cnn_model.evaluate(x=X_test_5, y=y_test_5)
# print("Accuracy: {}, Loss: {} \n".format(test_acc, test_loss))



