# Variational Informational Bottleneck from Alemi (2017)
# Original code by authors of paper at https://github.com/alexalemi/vib_demo
# Here comments have been added and things rewritten to improve readability
# Mean of distribution is used

import tensorflow as tf


tf.reset_default_graph()

# Turn on xla optimization
config = tf.ConfigProto()
config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
sess = tf.InteractiveSession(config=config)


# Import mnist data. Contains train images, test images, train labels, test labels
from tensorflow.examples.tutorials.mnist import input_data
mnist_data = input_data.read_data_sets('/tmp/mnistdata', validation_size=0)

# Placeholder for images; value stays constant, added during runtime
# Shape is dimension of tensor, a matrix with number of rows equal to number of images
# None is number of images, currently unknown
# Number of columns = 784, tthe number of pixels in an image
images = tf.placeholder(tf.float32, shape=[None, 784], name='images')

# Placeholders for labels
# Shape is array with dimension equal to number of images
# None is number of images, currently unknown
labels = tf.placeholder(tf.int64, shape=[None], name='labels')

# One-hot encoding of the labels
# Matrix with number of rows equal to number of images
# Number of columns = 10, number of classes
one_hot_labels = tf.one_hot(labels, 10)


layers = tf.contrib.layers
# Probability distributions
ds = tf.contrib.distributions


def encoder(images):
    # Each layer is tensor of rank 2, 1024 columns

    # Connect the first hidden layer to input layer
    # (features) with relu activation
    first_hidden_layer = layers.relu(2 * images - 1, 1024)
    # Connect the second hidden layer to first hidden layer with relu
    second_hidden_layer = layers.relu(first_hidden_layer, 1024)
    # Connect the output layer to second hidden layer (no activation fn)
    # Linear activation function is just identity function
    output_layer = layers.linear(second_hidden_layer, 512)

    # mu used as mean, first 256 nodes
    # rho used as standard deviation, last 256 nodes
    # rank 2 tensor (matrix) 256 columns
    mu, rho = output_layer[:, :256], output_layer[:, 256:]


    # Normal distribution: mean mu,
    # standard deviationL softplus(rho - 5)


    # encoding = ds.NormalWithSoftplusScale(mu, rho - 5.0)

    encoding = output_layer[:, :256]
    return encoding


def decoder(encoding_sample):
    # linear activation function
    net = layers.linear(encoding_sample, 10)
    return net


# Standard normal distribution
prior = ds.Normal(0.0, 1.0)

import math

with tf.variable_scope('encoder'):
    encoding = encoder(images)


with tf.variable_scope('decoder'):
    logits = decoder(encoding)

# with tf.variable_scope('decoder', reuse=True):
#     many_logits = decoder(encoding.sample(12))

# passes logits into softmax function and then funds cross entropy with one hot encoded labels
class_loss = tf.losses.softmax_cross_entropy(
    logits=logits, onehot_labels=one_hot_labels) / math.log(2)

BETA = 0

# tf.reduce_mean finds the mean
# tf.reduce_sum(x,0) finds sum of rows
# KL divergence between encoding and standard normal prior

info_loss = tf.reduce_sum(tf.reduce_mean(
    ds.kl_divergence(encoding, prior), 0)) / math.log(2)


total_loss = class_loss + BETA * info_loss

# Returns True if output from network (logits) is equal to label, False otherwise
# Argmax returns index of highest entry along axis 1 of the logits tensor
correct_prediction = tf.equal(
    tf.argmax(logits, 1), labels)

# Cast lists of booleans to floating point numbers, True becomes 1 and False becomes 0
correct_prediction = tf.cast(correct_prediction, tf.float32)

# Mean of the correct_predictions array is the percentage of correct predictions
accuracy = tf.reduce_mean(correct_prediction)

# # Calculate average accuracy using Monte Carlo average of 12 samples from the encoder
# correct_predictions_2 = tf.equal(
#     tf.argmax(tf.reduce_mean(tf.nn.softmax(many_logits), 0), 1), labels)
#
# # Mean is the number of correct predictions
# avg_accuracy = tf.reduce_mean(tf.cast(correct_predictions_2, tf.float32))

# Bound of mutual information of Z and Y
IZY_bound = math.log(10, 2) - class_loss

# Bound of mutual information of Z and X
IZX_bound = info_loss

# Mini-batch size of 100
batch_size = 100

# Number of steps in each epoch
# 600 mini-batches in 60,000 image training set = 600 steps each epoch
steps_per_batch = int(mnist_data.train.num_examples / batch_size)

# Variable incremented by 1 at every step
global_step = tf.contrib.framework.get_or_create_global_step()

# Step decay learning rate with initial lr = 1e-4, then decaying by 0.97 every 2 epochs
# decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
learning_rate = tf.train.exponential_decay(learning_rate=1e-4, global_step=global_step,
                                           decay_steps=2*steps_per_batch,
                                           decay_rate=0.97, staircase=True)

# Adam optimizer. Beta_1 = 0.5, Beta_2 = 0.999 (default)
opt = tf.train.AdamOptimizer(learning_rate, 0.5)

# Polyark averaging with decay constant 0.999
# Moving average of trained parameters with exponential decay
# shadow_variable = decay * shadow_variable + (1 - decay) * variable
ma = tf.train.ExponentialMovingAverage(0.999, zero_debias=True)
# Create shadow copy of trained variables
ma_update = ma.apply(tf.model_variables())

# Save variables
saver = tf.train.Saver()
saver_polyak = tf.train.Saver(ma.variables_to_restore())


# Computes gradients and returns loss
train_tensor = tf.contrib.training.create_train_op(total_loss, opt,
                                                   global_step,
                                                   update_ops=[ma_update])

tf.global_variables_initializer().run()


enc, cp, IZY, IZX, acc = sess.run([encoding.sample(),correct_prediction, IZY_bound, IZX_bound, accuracy],
                                      feed_dict={images: mnist_data.test.images, labels: mnist_data.test.labels})
print(enc)

# Evaluate performance of the network on the 10,000 images testing set
# Accuracy uses the stochastic encoder with one sample
# Average accuracy uses a Monte Carlo average of 12 samples from the encoder
def evaluate():
    cp, IZY, IZX, acc = sess.run([correct_prediction,IZY_bound, IZX_bound, accuracy],
                                      feed_dict={images: mnist_data.test.images, labels: mnist_data.test.labels})

    return IZY, IZX, acc, avg_acc, 1 - acc, 1 - avg_acc

# Same as previous evaluate function expect with train images instead of test images
def evaluate2():
    IZY, IZX, acc, avg_acc = sess.run([IZY_bound, IZX_bound, accuracy],
                                      feed_dict={images: mnist_data.train.images, labels: mnist_data.train.labels})
    return IZY, IZX, acc, avg_acc, 1 - acc, 1 - avg_acc

import sys

# Keep track of history
# Test set
IZY_array = []
IZX_array = []
acc_array = []
avg_acc_array = []
error_array = []
avg_error_array = []

# Train set history
IZY_array2 = []
IZX_array2 = []
acc_array2= []
avg_acc_array2= []
error_array2 = []
avg_error_array2 = []

# Train for 200 epochs
for epoch in range(0):

    # Train each minibatch
    # Number of minibatches is total train images (60000) / minibatch size (100) = 600
    for step in range(steps_per_batch):
        # Get next 600 images and labels from training set
        im, ls = mnist_data.train.next_batch(batch_size)
        # Feed them into the train tensor
        sess.run(train_tensor, feed_dict={images: im, labels: ls})


    # Keep track of information bounds, accuracy and error
    IZY, IZX, acc, avg_acc, error, avg_error = evaluate()
    IZY_array.append(IZY)
    IZX_array.append(IZX)
    acc_array.append(acc)
    avg_acc_array.append(avg_acc)
    error_array.append(error)
    avg_error_array.append(avg_error)

    IZY2, IZX2, acc2, avg_acc2, error2, avg_error2 = evaluate2()
    IZY_array2.append(IZY2)
    IZX_array2.append(IZX2)
    acc_array2.append(acc2)
    avg_acc_array2.append(avg_acc2)
    error_array2.append(error2)
    avg_error_array2.append(avg_error2)



    # # Print after every epoch
    # # Testing set
    # print("{}: IZY={:.2f}\tIZX={:.2f}\tacc={:.4f}\tavg_acc={:.4f}\terr={:.4f}\tavg_err={:.4f}".format(
    #     epoch, *(IZY, IZX, acc, avg_acc, error, avg_error) ))
    #
    # # Training set
    # print("{}: IZY={:.2f}\tIZX={:.2f}\tacc={:.4f}\tavg_acc={:.4f}\terr={:.4f}\tavg_err={:.4f}".format(
    #     epoch, *(IZY2, IZX2, acc2, avg_acc2, error2, avg_error2) ))

    sys.stdout.flush()

savepth = saver.save(sess, '/tmp/mnistvib', global_step)


saver_polyak.restore(sess, savepth)
#print(evaluate())
#
saver.restore(sess, savepth)
# print(evaluate())

# Save history to text file
file1 = open("alemiExampleHistory.txt", "a")
file1.write('\nBeta ' + str(BETA) + '\n')
file1.write(str(IZY_array) + '\n')
file1.write(str(IZX_array) + '\n')
file1.write(str(acc_array) + '\n')
file1.write(str(avg_acc_array) + '\n')
file1.write(str(error_array) + '\n')
file1.write(str(avg_error_array) + '\n')
file1.write('training set\n')
file1.write(str(IZY_array2) + '\n')
file1.write(str(IZX_array2) + '\n')
file1.write(str(acc_array2) + '\n')
file1.write(str(avg_acc_array2) + '\n')
file1.write(str(error_array2) + '\n')
file1.write(str(avg_error_array2) + '\n')


file1.close()