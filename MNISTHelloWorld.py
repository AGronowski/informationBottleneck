from __future__ import absolute_import, division, print_function, unicode_literals

import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import plotting


# Create a new neural network
def create_model():
    model = keras.models.Sequential([
        # Input layer. Image is 28 x 28 pixels, so layer has 784 nodes
        keras.layers.Flatten(input_shape=(28, 28)),
        # 16 node hidden layer with reLu activation function
        keras.layers.Dense(10, activation=tf.nn.relu, ),
        # 16 node hidden layer with reLu activation function
        keras.layers.Dense(10, activation=tf.nn.relu),
        # 512 node hidden layer, no activation function (linear function, o(x) = x)
        # keras.layers.Dense(512, activation=None),
        # 10 node output layer with softmax activation function
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    adam = keras.optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    batch_size = 100
    steps_per_batch = int(60000 / batch_size)

    # global_step = tf.contrib.framework.get_or_create_global_step()
    # learning_rate = tf.train.exponential_decay(learning_rate=1e-4, global_step=global_step,
    #                                            decay_steps=2 * steps_per_batch,
    #                                            decay_rate=0.97, staircase=True)
    sgd = keras.optimizers.SGD(lr=1e-4, decay=0)

    model.compile(optimizer=adam,  # Variation of stochastic gradient descent
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])  # Percentage classified correctly

    return model


# Learning rate schedule with step decay
def step_decay_schedule(initial_lr=1e-3, decay_factor=0.75, step_size=10):
    '''
    Wrapper function to create a LearningRateScheduler with step decay schedule.
    '''

    def schedule(epoch):
        return initial_lr * (decay_factor ** np.floor(epoch / step_size))

    return keras.callbacks.LearningRateScheduler(schedule)

def history():
    # Get history
    full_history = history_callback.history
    train_loss = history_callback.history["loss"]
    train_acc = history_callback.history["acc"]
    val_loss = history_callback.history["val_loss"]
    val_acc = history_callback.history["val_acc"]

    plotting.plot_4_history(train_loss, train_acc, val_loss, val_acc)

    print("loss=" + str(train_loss))
    print("acc=" + str(train_acc))
    print("val_loss=" + str(val_loss))
    print("val_acc=" + str(val_acc))

    numpy_loss_history = np.array(val_loss)
    # np.savetxt("loss_history.txt", [numpy_loss_history], delimiter=', ', header='[',footer=']')

    with open('loss_history.txt', 'a') as f:
        np.savetxt(f, [numpy_loss_history], delimiter=', ', header='[', footer=']')

    with open('full_history.txt', 'a') as f:
        np.savetxt(f, [full_history], delimiter='}', fmt="%s")



# Import MNIST data
fashion_mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# Train on only first numMNIST images to reduce computational time
numMNIST = 10000
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
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1,
                                                 period=2)  # Save every 2 epochs

# Create new model
model = create_model()

# One-hot labels are used with categorical_crossentropy, not used with sparse_categorical_crossentropy
train_labels_onehot = keras.utils.to_categorical(train_labels)
test_labels_onehot = keras.utils.to_categorical(test_labels)

decay = 0.97

lr_sched = step_decay_schedule(initial_lr=1e-4, decay_factor=decay, step_size=2)

train = True
epoch = 3
# Train model
if train:
    history_callback = model.fit(train_images, train_labels, batch_size=100, epochs=epoch,
                                 validation_data=(test_images, test_labels),
                                 callbacks=[cp_callback])  # Pass callback to training
    # Save entire model
    model.save('my_model.h5')
    history()



# Recreate another model from saved 1st model, check accuracy (should be the same)
model3 = keras.models.load_model('my_model.h5')
# print(model3.summary)
# loss, acc = model3.evaluate(test_images, test_labels)
# print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
