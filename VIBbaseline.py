# Baseline of the Alemi (2017) paper
# Baseline is a 784-1024-1024-10 neural network trained on the full 60,000 image MNIST database
# Training over 200 epochs with Adam optimizer
# Optimizer parameters, exponential decay, and Polyak averaging used, as described in paper

import tensorflow as tf
import math


tf.reset_default_graph()

# Turn on xla optimization
config = tf.ConfigProto()
config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
sess = tf.InteractiveSession(config=config)


# Import mnist data. Contains train images, test images, train labels, test labels
from tensorflow.examples.tutorials.mnist import input_data
mnist_data = input_data.read_data_sets('/tmp/mnistdata', validation_size=0)

# Placeholder for images. Shape is dimension of tensor.
# Shape is matrix with number of rows equal to number of images
# Number of columns = 784, number of pixels in an image
images = tf.placeholder(tf.float32, shape=[None, 784], name='images')

# Placeholders for images
# Shape is array with dimension equal to number of images
labels = tf.placeholder(tf.int64, shape=[None], name='labels')

# One-hot encoding of the labels
# Matrix with number of rows equal to number of images
# Number of columns = 10, number of classes
one_hot_labels = tf.one_hot(labels, 10)

layers = tf.contrib.layers

# Images go into first hidden layer
first_hidden_layer = layers.relu(2 * images - 1, 1024)

# Connect the second hidden layer to first hidden layer with relu
second_hidden_layer = layers.relu(first_hidden_layer, 1024)


# Connect second hidden layer to final ouput layer
# Linear activation function is identity function
output_layer = layers.linear(second_hidden_layer, 10)

# Logits are the output of the network before the softmax function
logits = output_layer

# Apply softmax on logits and then find cross entropy
class_loss = tf.losses.softmax_cross_entropy(
    logits=logits, onehot_labels=one_hot_labels) / math.log(2)

total_loss = class_loss

# Returns True if output from network (logits) is equal to label, False otherwise
# Argmax returns index of highest entry along axis 1 of the logits tensor
# Logits are output from final output layer with no activation function applied
# They are not normalized, but this does not matter for predictions
correct_prediction = tf.equal(
    tf.argmax(logits, 1), labels)

# Cast lists of booleans to floating point numbers, True becomes 1 and False becomes 0
correct_prediction = tf.cast(correct_prediction, tf.float32)

# Mean of the correct_predictions array is the percentage of correct predictions
accuracy = tf.reduce_mean(correct_prediction)



IZY_bound = math.log(10, 2) - class_loss
# IZX_bound = info_loss

batch_size = 100
steps_per_batch = int(mnist_data.train.num_examples / batch_size)
#In [9]:
global_step = tf.contrib.framework.get_or_create_global_step()
learning_rate = tf.train.exponential_decay(learning_rate=1e-4, global_step= global_step,
                                           decay_steps=2*steps_per_batch,
                                           decay_rate=0.97, staircase=True)

# Adam optimizer
opt = tf.train.AdamOptimizer(learning_rate, 0.5)

# Polyak averaging
ma = tf.train.ExponentialMovingAverage(0.999, zero_debias=True)
ma_update = ma.apply(tf.model_variables())

saver = tf.train.Saver()
saver_polyak = tf.train.Saver(ma.variables_to_restore())

# Total loss is cost, opt is optimizer
train_tensor = tf.contrib.training.create_train_op(total_loss, opt,
                                                   global_step,
                                                   update_ops=[ma_update])
#In [10]:
tf.global_variables_initializer().run()

# Ignore these, they are not used, left over from VIB code snippet
IZX_bound = IZY_bound
avg_accuracy = accuracy

def evaluate():
    IZY, IZX, acc, avg_acc = sess.run([IZY_bound, IZX_bound, accuracy, avg_accuracy],
                                      feed_dict={images: mnist_data.test.images, labels: mnist_data.test.labels})
    return IZY, IZX, acc, avg_acc, 1 - acc, 1 - avg_acc


#In[12]:
import sys

# Number of epochs of training
for epoch in range(200):
    for step in range(steps_per_batch):
        im, ls = mnist_data.train.next_batch(batch_size)
        sess.run(train_tensor, feed_dict={images: im, labels: ls})
    print("{}: IZY={:.2f}\tIZX={:.2f}\tacc={:.4f}\tavg_acc={:.4f}\terr={:.4f}\tavg_err={:.4f}".format(
        epoch, *evaluate()))
    sys.stdout.flush()

savepth = saver.save(sess, '/tmp/mnistvib', global_step)

saver_polyak.restore(sess, savepth)
print(evaluate())

saver.restore(sess, savepth)
print(evaluate())