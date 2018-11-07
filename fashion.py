# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
#import matplotlib
import matplotlib.pyplot as plt

#matplotlib.use('agg') # for X11

print('Tensorflow version:', tf.__version__)

# map class indices to names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# load fashion dataset
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# scale pixel values (0,255) => (0,1)
train_images = train_images / 255.0
test_images = test_images / 255.0

print('Train images shape:', np.shape(train_images))

# build a simple model
if (0):
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(optimizer=tf.train.AdamOptimizer(), 
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(train_images, train_labels, epochs=5)
    model.summary()

    # evaluate accuracy
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('Test accuracy:', test_acc)

# build a better model
model = keras.Sequential([
    keras.layers.Conv2D(32, (2,2), padding='same', activation=tf.nn.relu, kernel_initializer='random_uniform', input_shape=(28, 28, 1)),
    keras.layers.MaxPool2D(padding='same'),
    keras.layers.Conv2D(32, (2,2), padding='same', kernel_initializer='random_uniform', input_shape=(1, 28, 28)),
    keras.layers.MaxPool2D(padding='same'),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation=tf.nn.relu, kernel_initializer='random_uniform'),
    keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# reshape input from (28,28) to (28,28,1) - first layer Conv2D wants 4D tensor as input
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

model.fit(train_images, train_labels, epochs=5)
model.summary()

# evaluate accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

