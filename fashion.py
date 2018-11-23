#
# This code implements the 6 steps required to complete HW2:
# https://gradebot.org/course/ecs235a/18f/homework/2/
#
# - Step 1 -
# Dependencies:
# pip install tensorflow
# pip install keras
# pip install cleverhans

# TensorFlow and tf.keras
import tensorflow as tf
import keras
from keras.datasets import fashion_mnist

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
from cleverhans.attacks import BasicIterativeMethod
from cleverhans.utils_keras import KerasModelWrapper

print('Tensorflow version:', tf.__version__)

# map class indices to names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# load fashion dataset
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# scale pixel values (0,255) => (0,1)
train_images = train_images / 255.0
test_images = test_images / 255.0

# Obtain Image Parameters
img_rows, img_cols = train_images.shape[1:3]
nchannels = 1 # greyscale
nb_classes = 10
nb_iter = 10

print('Images rows: {} columns: {}'.format(img_rows, img_cols))

# - Step 2 - build a simple model
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

# Create TF session and set as Keras backend session
sess = tf.Session()
keras.backend.set_session(sess)

# reshape input from (28,28) to (28,28,1) - first layer Conv2D wants 4D tensor as input
train_images = train_images.reshape(train_images.shape[0], 28, 28, nchannels)
test_images2 = test_images.reshape(test_images.shape[0], 28, 28, nchannels)

filename = 'model.h5'
try:  # load a previously trained model
  model = keras.models.load_model(filename)
  print("Model loaded from: {}".format(filename))
except OSError as e:
  print(e)
  # - Step 3 - build a model which can have >90% accuracy
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
test_loss, test_acc = model.evaluate(test_images2, test_labels)
print('Test accuracy:', test_acc)

# - Step 4 - Implement untargeted attack using the Basic Iterative Method
wrap = KerasModelWrapper(model)
bim = BasicIterativeMethod(wrap, sess=sess)
bim_params = {'eps': 0.3,
              'eps_iter': 0.05,
              'nb_iter': nb_iter,
              'clip_min': 0.,
              'clip_max': 1.}

# - Step 5 - From each category in the Fashion-MNIST test set, select 10 images
# to generate adversarial examples using 5 and 10 iterations, respectively.
def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}%\n({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         i), color=color)

# Visualize adversarial images from a trained network
def adv_viz(attack, attack_params):
  # Generate adversarial images ...
  idxs = [np.where(test_labels == i)[0][0]
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
  fig = grid_visual(grid_viz_data)
  fig.savefig('grid_viz.png')
  return adv_images

predictions = model.predict(test_images2)

# Plot the 10 test images from each category, their predicted label, and the true label
# Color correct predictions in blue, incorrect predictions in red
pos = 0
plt.figure(figsize=(20, 20))
for clz in range(nb_classes):
    idxs = np.where(test_labels == clz)[0][:10]
    for i in idxs:
      adv_inputs = test_images[idxs]
      #adv_images = bim.generate_np(adv_inputs, **bim_params)
      plt.subplot(10, 10, pos+1)
      pos += 1
      plot_image(i, predictions, test_labels, test_images)
plt.savefig('original-images.png'.format(nb_iter))
#plt.savefig('{}-iterations.png'.format(nb_iter))
