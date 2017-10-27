import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def get_weights(n_features, n_labels):

    return tf.Variable(tf.truncated_normal((n_features, n_labels)))

def get_biases(n_labels):
    return tf.Variable( tf.zeros(n_labels) )

def linear(input, w, b):
    return tf.add( tf.matmul(input, w), b )

def mnist_features_labels(n_labels):
    mnist_features = []
    mnist_labels = []

    #mnist = input_data.read_data_sets('/datasets/ud730/mnist', one_hot=True)
    mnist = input_data.read_data_sets('mnist', one_hot=True)
    # In order to make quizzes run faster, we're only looking at 10000 images
    for mnist_feature, mnist_label in zip(*mnist.train.next_batch(10000)):

        # Add features and labels if it's for the first <n>th labels
        if mnist_label[:n_labels].any():
            mnist_features.append(mnist_feature)
            mnist_labels.append(mnist_label[:n_labels])

    return mnist_features, mnist_labels
print('Begin the training')
n_features = 784
n_labels = 3

features = tf.placeholder(tf.float32)
labels = tf.placeholder(tf.float32)

w = get_weights(n_features, n_labels)
b = get_biases(n_labels)

logits = linear(features, w, b)

train_features, train_labels = mnist_features_labels(n_labels)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    prediction = tf.nn.softmax(logits)

    cross_entropy = -tf.reduce_sum( labels * tf.log(prediction), reduction_indices = 1 )

    loss = tf.reduce_mean(cross_entropy)

    learning_rate = 0.08

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    _,l = sess.run([optimizer,loss], feed_dict={features: train_features, labels: train_labels})

print('Loss={}'.format(l))
