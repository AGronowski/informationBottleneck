from __future__ import absolute_import, division, print_function, unicode_literals

import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import plotting


train = True
epoch = 20
hidden_layer_size = 32
optimizer = 'adamDefault'
decay = False
description = str(hidden_layer_size) + "-" + str(hidden_layer_size) + optimizer + "decay: " + str(decay)

# Create a new neural network
def create_model():
    initializer = 'he_uniform'
    model = keras.models.Sequential([
        # Input layer. Image is 28 x 28 pixels, so layer has 784 nodes
        keras.layers.Flatten(input_shape=(28, 28)),
        # 16 node hidden layer with reLu activation function
        keras.layers.Dense(hidden_layer_size, activation=tf.nn.relu,kernel_initializer=initializer),
        # 16 node hidden layer with reLu activation function
        keras.layers.Dense(hidden_layer_size, activation=tf.nn.relu,kernel_initializer=initializer),
        # 512 node hidden layer, no activation function (linear function, o(x) = x)
        # keras.layers.Dense(512, activation=None),
        # 10 node output layer with softmax activation function
        keras.layers.Dense(10, activation=tf.nn.softmax,kernel_initializer=initializer)
    ])

    # Adam optimizer, lr set to 0 to indicate that it's not being used here; lr controlled by lr_sched callback
    if optimizer == 'adamPaper':
      adam = keras.optimizers.Adam(lr=0, beta_1=0.5, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    if optimizer == 'adamDefault':
      adam = keras.optimizers.Adam(lr=0.001,beta_1=0.9,beta_2=0.999)
    else:
        raise Exception("invalid name for optimizer")

    batch_size = 100
    steps_per_batch = int(60000 / batch_size)

    # global_step = tf.contrib.framework.get_or_create_global_step()
    # learning_rate = tf.train.exponential_decay(learning_rate=1e-4, global_step=global_step,
    #                                            decay_steps=2 * steps_per_batch,
    #                                            decay_rate=0.97, staircase=True)
    #sgd = keras.optimizers.SGD(lr=1e-4, decay=0)

    model.compile(optimizer=adam,  # Variation of stochastic gradient descent
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])  # Percentage classified correctly

    return model


# Learning rate schedule with step decay
def step_decay_schedule(initial_lr, decay_factor, step_size):
    '''
    Wrapper function to create a LearningRateScheduler with step decay schedule.
    '''

    def schedule(epoch):
        return initial_lr * (decay_factor ** np.floor(epoch / step_size))

    return keras.callbacks.LearningRateScheduler(schedule)

def history():
    # Get history
    train_loss = history_callback.history["loss"]
    train_acc = history_callback.history["acc"]
    val_loss = history_callback.history["val_loss"]
    val_acc = history_callback.history["val_acc"]

    # Graph history
    plotting.plot_4_history(train_loss, train_acc, val_loss, val_acc,"testDescription")

    # History strings
    train_loss_string = "loss=" + str(train_loss)
    train_acc_string = "acc=" + str(train_acc)
    val_loss_string = "val_loss=" + str(val_loss)
    val_acc_string = "val_acc=" + str(val_acc)

    # Final accuracy strings
    end = len(val_acc)
    final_val_acc_string = "final validation accuracy: " + str(val_acc[end-1])
    final_test_acc_string = "final_test_accuracy: " + str(val_acc[end-1])

    # Get number of wrongly classified images
    numWrongTest, percentageTest = plotting.get_num_wrong(test_images, test_labels, model)
    numWrongTrain, percentageTrain = plotting.get_num_wrong(train_images, train_labels, model)

    # Number of wrong strings
    test_wrong_string = "Test set: " + str(numWrongTest) + " wrong, " + str(percentageTest) + "%"
    train_wrong_string = "Training set: " + str(numWrongTrain) + " wrong, " + str(percentageTrain) + "%"

    # Print loss
    print(train_loss_string)
    print(train_acc_string)
    print(val_loss_string)
    print(val_acc_string)

    # Print final accuracy
    print(final_val_acc_string)
    print(final_test_acc_string)

    # Print number of wrong
    print(test_wrong_string)
    print(train_wrong_string)

    # Save history, final accuracy, and  to text file
    file1 = open("full_history.txt", "a")
    file1.write('\n' + description +'\n')
    file1.write(train_loss_string + '\n')
    file1.write(train_acc_string + '\n')
    file1.write(val_loss_string + '\n')
    file1.write(val_acc_string + '\n')
    file1.write(final_test_acc_string + '\n')
    file1.write(final_val_acc_string + '\n')
    file1.write(test_wrong_string + '\n')
    file1.write(train_wrong_string + '\n')
    file1.close()


# Import MNIST data
fashion_mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# Train on only first numMNIST images to reduce computational time
numMNIST = 60000
numTest = 10000
train_images = train_images[:numMNIST]
test_images = test_images[:numTest]
train_labels = train_labels[:numMNIST]
test_labels = test_labels[:numTest]

# Scale images to have values between 0 and 1
train_images = train_images / 255.0
test_images = test_images / 255.0

# Create checkpoint callback
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
# cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
#                                                  save_weights_only=True,
#                                                  verbose=1,
#                                                  period=100)  # Save every 100 epochs

# Create new model
model = create_model()

# One-hot labels are used with categorical_crossentropy, not used with sparse_categorical_crossentropy
train_labels_onehot = keras.utils.to_categorical(train_labels)
test_labels_onehot = keras.utils.to_categorical(test_labels)

# Callback adds learning rate decay every step_size epochs
lr_sched = step_decay_schedule(initial_lr=1e-4, decay_factor=0.97, step_size=2)

if decay:
    callbacks = [lr_sched]
else:
    callbacks = None

# Train model
if train:
    history_callback = model.fit(train_images, train_labels, batch_size=100, epochs=epoch,
                                 validation_data=(test_images, test_labels),
                                 callbacks=callbacks)  # Pass callback to training
    # Save entire model
    model.save('my_model.h5')
    # Print loss and accuracy and save to text file
    history()

# Load saved model
model3 = keras.models.load_model('my_model.h5')

print(model3.summary)

plotting.plot_images_with_prediction(model3,class_names,test_labels,test_images,4,4,True)
# loss, acc = model3.evaluate(test_images, test_labels)
# print("Restored model, accuracy: {:5.2f}%".format(100 * acc))