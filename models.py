import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

# load data
DATA_PATH = "./None"
data = np.loadtxt(DATA_PATH, delimiter=",")
# separate X and y - X is the features and y is the result
X, y_3weeks, y_5weeks = data[:, :15], data[:, 15], data[:,16]
X_train_3, X_test_3, y_train_3, y_test_3 = train_test_split(X, y_3weeks, test_size=0.2)
X_train_5, X_test_5, y_train_5, y_test_5 = train_test_split(X, y_5weeks, test_size=0.2)

# cnn model
cnn_model = tf.keras.models.Sequential()
cnn_model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
cnn_model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
cnn_model.add(tf.keras.layers.Dense(128, activation=tf.nn.softmax))
# TODO: maybe change the loss to MSE
cnn_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
cnn_model_log_3 = cnn_model.fit(x=X_train_3, y=y_train_3, batch_size=60,
                                verbose=1, epochs=200, validation_split=0.25)
test_loss, test_acc = cnn_model.evaluate(x=X_test_3)
cnn_model_log_5 = cnn_model.fit(x=X_train_5, y=y_train_5, batch_size=60,
                                verbose=1, epochs=200, validation_split=0.25)


