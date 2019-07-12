import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf


tf.reset_default_graph()

# Turn on xla optimization
config = tf.ConfigProto()
config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
sess = tf.InteractiveSession(config=config)

from tensorflow.examples.tutorials.mnist import input_data
mnist_data = input_data.read_data_sets('/tmp/mnistdata', validation_size=0)

#images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, IMAGE_PIXELS))
images = tf.placeholder(tf.float32, shape=[None, 784], name='images')

#labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
labels = tf.placeholder(tf.int64, shape=[None], name='labels')
one_hot_labels = tf.one_hot(labels, 10)

layers = tf.contrib.layers
ds = tf.contrib.distributions


def encoder(images):
    # Connect the first hidden layer to input layer
    # (features) with relu activation
    first_hidden_layer = layers.relu(2 * images - 1, 1024)
    # Connect the second hidden layer to first hidden layer with relu
    second_hidden_layer = layers.relu(first_hidden_layer, 1024)
    # Connect the output layer to second hidden layer (no activation fn)
    output_layer = layers.linear(second_hidden_layer, 512)
    mu, rho = output_layer[:, :256], output_layer[:, 256:]
    encoding = ds.NormalWithSoftplusScale(mu, rho - 5.0)
    return encoding


def decoder(encoding_sample):
    net = layers.linear(encoding_sample, 10)
    return net


prior = ds.Normal(0.0, 1.0)
#In[6]:
import math

with tf.variable_scope('encoder'):
    encoding = encoder(images)

with tf.variable_scope('decoder'):
    logits = decoder(encoding.sample())

with tf.variable_scope('decoder', reuse=True):
    many_logits = decoder(encoding.sample(12))

class_loss = tf.losses.softmax_cross_entropy(
    logits=logits, onehot_labels=one_hot_labels) / math.log(2)

BETA = 1e-3

info_loss = tf.reduce_sum(tf.reduce_mean(
    ds.kl_divergence(encoding, prior), 0)) / math.log(2)

total_loss = class_loss + BETA * info_loss

accuracy = tf.reduce_mean(tf.cast(tf.equal(
    tf.argmax(logits, 1), labels), tf.float32))
avg_accuracy = tf.reduce_mean(tf.cast(tf.equal(
    tf.argmax(tf.reduce_mean(tf.nn.softmax(many_logits), 0), 1), labels), tf.float32))
IZY_bound = math.log(10, 2) - class_loss
IZX_bound = info_loss

batch_size = 100
steps_per_batch = int(mnist_data.train.num_examples / batch_size)
#In [9]:
global_step = tf.contrib.framework.get_or_create_global_step()
learning_rate = tf.train.exponential_decay(learning_rate=1e-4, global_step= global_step,
                                           decay_steps=2*steps_per_batch,
                                           decay_rate=0.97, staircase=True)
# opt = tf.train.AdamOptimizer(learning_rate, 0.5)
#
# ma = tf.train.ExponentialMovingAverage(0.999, zero_debias=True)
# ma_update = ma.apply(tf.model_variables())
#
# saver = tf.train.Saver()
# saver_polyak = tf.train.Saver(ma.variables_to_restore())
#
# train_tensor = tf.contrib.training.create_train_op(total_loss, opt,
#                                                    global_step,
#                                                    update_ops=[ma_update])
# #In [10]:
# tf.global_variables_initializer().run()
#
#
# def evaluate():
#     IZY, IZX, acc, avg_acc = sess.run([IZY_bound, IZX_bound, accuracy, avg_accuracy],
#                                       feed_dict={images: mnist_data.test.images, labels: mnist_data.test.labels})
#     return IZY, IZX, acc, avg_acc, 1 - acc, 1 - avg_acc
#
#
# #In[12]:
# import sys
#
# for epoch in range(2):
#     for step in range(steps_per_batch):
#         im, ls = mnist_data.train.next_batch(batch_size)
#         sess.run(train_tensor, feed_dict={images: im, labels: ls})
#     print
#     ("{}: IZY={:.2f}\tIZX={:.2f}\tacc={:.4f}\tavg_acc={:.4f}\terr={:.4f}\tavg_err={:.4f}".format(
#         epoch, *evaluate()))
#     sys.stdout.flush()
#
# savepth = saver.save(sess, '/tmp/mnistvib', global_step)
#
# saver_polyak.restore(sess, savepth)
# evaluate()
#
# saver.restore(sess, savepth)
# evaluate()