# import tensorflow as tf
#
# a = tf.placeholder(tf.float32, shape=(3,2))
# b = tf.placeholder(tf.float32, shape=(2,3))
# # We provide the shape of each type of input as the second argument.
# # Here, we expect `a` to be a 2-dimensional matrix, with 2 rows and 1 column
# # and `b` to be a matrix with 1 row and 2 columns
#
# c = tf.matmul(a, b)
# # Instead of addition, we define `c` as the matrix multiplication operation,
# # with the inputs coming from `a` and `b`
#
# sess = tf.Session()
#
# output = sess.run(c, {a:[[1,2],[1,3],[1,4]], b:[[1,1,1],[2,3,4]]})
# print(output)
#
# # sess = tf.Session(c, {a:[[1,2],[1,3],[1,4]], b:[[1,1,1],[2,3,4]]})
#

import tensorflow as tf
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)


