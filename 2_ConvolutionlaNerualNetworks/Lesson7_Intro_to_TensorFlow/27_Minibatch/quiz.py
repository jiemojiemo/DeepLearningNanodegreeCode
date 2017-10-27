from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
from helper import batches

learning_rate = 0.001
n_input = 784
n_classes = 10

mnist = input_data.read_data_sets('/datasets/ud730/mnist', one_hot=True)

# The features are already scaled and the data is shuffled
train_features = mnist.train.images
test_features = mnist.test.images

train_labels = mnist.train.labels
test_labels = mnist.test.labels

# Build the model
# Features and Labels
features = tf.placeholder(tf.float32, [None, n_input])
labels = tf.placeholder(tf.float32, [None, n_classes])

weights = tf.Variable(tf.random_normal([n_input, n_classes]))
bias = tf.Variable(tf.zeros([n_classes]))

# XW + b
logits = tf.add( tf.matmul(features, weights), bias )

# Define loss and optimizer
cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels) )
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Calculate accuracy
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

batch_size = 128

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for batch_features, batch_labels in batches(batch_size, train_features, train_labels):
        sess.run(optimizer, feed_dict={features:batch_features, labels:batch_labels})

    test_accuracy = sess.run(
        accuracy,
        feed_dict={features:test_features, labels:test_labels}
    )
print('Test Accuracy: {}'.format(test_accuracy))
