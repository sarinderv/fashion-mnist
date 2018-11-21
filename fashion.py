# Dependencies:
# pip3 install tensorflow
# pip3 install keras
# pip3 install git+https://github.com/sarinderv/cleverhans.git@fashion-mnist#egg=cleverhans

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
#import matplotlib
import matplotlib.pyplot as plt
from cleverhans.attacks import BasicIterativeMethod
from cleverhans.dataset import Fashion_MNIST
from cleverhans.loss import CrossEntropy
from cleverhans.train import train
from cleverhans.utils import pair_visual, grid_visual, AccuracyReport
from cleverhans.utils_keras import cnn_model
from cleverhans.utils_keras import KerasModelWrapper
from cleverhans.utils_tf import model_eval

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

# Obtain Image Parameters
# img_rows, img_cols = train_images.shape[1:3]
nchannels = 1 # greyscale
# nb_classes = 10 #train_labels.shape[1]
# # convert to one-hot
# train_labels = keras.utils.to_categorical(train_labels, nb_classes)
# test_labels = keras.utils.to_categorical(test_labels, nb_classes)

# print('Images rows: {} columns: {}'.format(img_rows, img_cols))
# print('Train images shape:', train_images.shape)
# print('Train labels shape:', train_labels.shape)
# print('Test images shape:', test_images.shape)
# print('Test labels shape:', test_labels.shape)

# build a simple model
if (0):
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.summary()
    model.compile(optimizer=tf.train.AdamOptimizer(), 
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(train_images, train_labels, epochs=5)

    # evaluate accuracy
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('Test accuracy:', test_acc)

# reshape input from (28,28) to (28,28,1) - first layer Conv2D wants 4D tensor as input
train_images = train_images.reshape(train_images.shape[0], 28, 28, nchannels)
test_images = test_images.reshape(test_images.shape[0], 28, 28, nchannels)

filename = 'model.h5'
try: # load a previously trained model
  model = tf.keras.models.load_model(filename)
  print("Model loaded from: {}".format(filename))
except OSError as e:
  print(e)
  # build a model which can have >90% accuracy
  model = keras.Sequential([
    keras.layers.Conv2D(32, (2,2), padding='same', activation=tf.nn.relu, input_shape=(28, 28, 1)),
    keras.layers.MaxPool2D(padding='same'),
    keras.layers.Conv2D(32, (2,2), padding='same', input_shape=(28, 28, 1)),
    keras.layers.MaxPool2D(padding='same'),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
  ])
  model.compile(optimizer=keras.optimizers.Adam(), 
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
  model.fit(train_images, train_labels, epochs=5)
  model.save(filename)

model.summary()

# evaluate accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

# Initialize the Fast Gradient Sign Method (FGSM) attack object and graph
bim = BasicIterativeMethod(KerasModelWrapper(model))
bim_params = {'eps': 0.3,
              'eps_iter': 0.05,
              'nb_iter': 5,
              'clip_min': 0.,
              'clip_max': 1.}

# Visualize adversarial images from a trained network
def adv_viz(attack, attack_params):
  print(np.where(np.argmax(test_labels, axis=1) == 0))

  # Generate adversarial images ...
  idxs = [np.where(np.argmax(test_labels, axis=1) == i)[0][0]
                for i in range(nb_classes)]
  adv_inputs = test_images[idxs]
  adv_images = attack.generate_np(adv_inputs, **attack_params)

  # Initialize our array for grid visualization
  grid_shape = (nb_classes, 2, img_rows, img_cols, nchannels)
  grid_viz_data = np.zeros(grid_shape, dtype='f')
  for j in range(nb_classes):
    grid_viz_data[j, 0] = adv_inputs[j]
    grid_viz_data[j, 1] = adv_images[j]
  print('grid_viz_data.shape=', grid_viz_data.shape)

  # Compute the average distortion introduced by the algorithm
  percent_perturbed = np.mean(np.sum((adv_images - adv_inputs)**2,
                                     axis=(1, 2, 3))**.5)
  print('Avg. L_2 norm of perturbations {0:.4f}'.format(percent_perturbed))

  # Finally, block & display a grid of all the adversarial examples
  #import matplotlib.pyplot as plt
  from cleverhans.plot.pyplot_image import grid_visual
  return grid_visual(grid_viz_data)

# Display a grid of some adversarial examples
adv_viz(bim, bim_params)

