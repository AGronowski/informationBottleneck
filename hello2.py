from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt


def create_model():
    model = keras.models.Sequential([
        # Input layer. Image is 28 x 28 pixels, so layer has 784 nodes
        keras.layers.Flatten(input_shape=(28, 28)),
        # 128 node hidden layer with reLu activation function
        keras.layers.Dense(1024, activation=tf.nn.relu),
        # 128 node hidden layer with reLu activation function
        keras.layers.Dense(1024, activation=tf.nn.relu),
        # 10 node output layer
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    # adam optimizer = variation of stochastic gradient descent,
    # accuracy = percentage classified correctly
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# print(tf.__version__)


fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# print(train_images.shape)
# print(len(train_labels))
# print(train_labels)
# print(test_images.shape)
#
# plt.figure()
# plt.imshow(train_images[5])
# plt.colorbar()
# plt.grid(False)
# plt.show()
# print(train_images[0])
# train_images = train_images[:1000] / 255.0
# test_images = test_images[:1000] / 255.0
# train_labels = train_labels[:1000]
# test_labels = test_labels[:1000]
train_images = train_images / 255.0
test_images = test_images / 255.0
# print(train_images[0])



#
# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()
#

# !ls {checkpoint_dir}
# checkpoint_path = "training_1/cp.ckpt"
# checkpoint_dir = os.path.dirname(checkpoint_path)
#
# # Create checkpoint callback
# cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
#                                                  save_weights_only=True,
#                                                  verbose=1)

model = create_model()
model.summary()

model.fit(train_images, train_labels, epochs=1)
#
# test_loss, test_acc = model.evaluate(test_images, test_labels)
# print('Test accuracy:', test_acc)
#
# predictions = model.predict(test_images)
# print(predictions[0])
# print(np.argmax(predictions[0]))
#
# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()
