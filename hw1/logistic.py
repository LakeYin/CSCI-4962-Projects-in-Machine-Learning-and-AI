import threading
import tensorflow as tf
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

def sigmoid(x):
    return 1 / (1 + tf.math.exp(-x))

# load the data
data = pd.read_csv(sys.argv[1])

X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :-1], data.iloc[:, -1], test_size=0.3)
X_train = tf.cast(tf.convert_to_tensor(X_train), tf.float32)
X_test = tf.cast(tf.convert_to_tensor(X_test), tf.float32)
y_train = tf.cast(tf.convert_to_tensor(y_train), tf.float32)
y_test = tf.cast(tf.convert_to_tensor(y_test), tf.float32)

train = tf.data.Dataset.from_tensor_slices((X_train, y_train))
test = tf.data.Dataset.from_tensor_slices((X_test, y_test))

# initialize values
weight = tf.random.uniform((1, X_train.shape[1]), minval=0.0001, maxval=0.001)
bias = tf.random.uniform((1, 1), minval=0.0001, maxval=0.001)
eta = 0.001

for i in range(5):
    # iterate and update at each value
    train = train.shuffle(len(train))
    for x, y in train:
        y_i = sigmoid(tf.tensordot(weight, x, 1) + bias)

        dw = x * (y_i - y)
        db = y_i - y

        weight = weight - eta * dw
        bias = bias - eta * db

# final prediction and error rate
errors = 0
for x, y in test:
    y_i = 1 if sigmoid(tf.tensordot(weight, x, 1) + bias) >= 0.5 else 0

    if y_i != y:
        errors += 1

print("Error rate: ")
print(errors / len(test))
