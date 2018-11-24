#
# This code implements the 6 steps required to complete HW2:
# https://gradebot.org/course/ecs235a/18f/homework/2/
#
# - Step 1 - Install TensorFlow.
# Dependencies:
# pip install tensorflow
# pip install keras
# pip install cleverhans

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras
from cleverhans.attacks import BasicIterativeMethod
from cleverhans.utils_keras import KerasModelWrapper
from keras.datasets import fashion_mnist

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
nchannels = 1  # greyscale
nb_classes = 10

print('Images rows: {} columns: {}'.format(img_rows, img_cols))

# - Step 2 - Implement the basic classification neural network.
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
    # - Step 3 - Improve the classification accuracy above 90% by adding 2D convolution layers and max pooling layers.
    model = keras.Sequential([
        keras.layers.Conv2D(32, (2, 2), padding='same', activation=tf.nn.relu, input_shape=(28, 28, 1)),
        keras.layers.MaxPool2D(padding='same'),
        keras.layers.Conv2D(32, (2, 2), padding='same', input_shape=(28, 28, 1)),
        keras.layers.MaxPool2D(padding='same'),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(optimizer=keras.optimizers.Adam(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(train_images, train_labels, epochs=5)
    model.save(filename)  # Save your model in case of retraining.

model.summary()

# evaluate accuracy
test_loss, test_acc = model.evaluate(test_images2, test_labels)
print('Test accuracy:', test_acc)

# - Step 4 - Implement untargeted attack using the Basic Iterative Method
wrap = KerasModelWrapper(model)
bim = BasicIterativeMethod(wrap, sess=sess)
bim_params = {'eps': 0.2,
              'eps_iter': 0.05,
              'clip_min': 0.,
              'clip_max': 1.}


# - Step 5 - From each category in the Fashion-MNIST test set,
# select 10 images to generate adversarial examples using 5 and 10 iterations, respectively.
orig_images = np.empty([0, 28, 28, 1])
orig_labels = np.empty([0])
for clz in range(nb_classes):
    idxs = np.where(test_labels == clz)[0][:10]
    orig_images = np.append(orig_images, test_images2[idxs], axis=0)
    orig_labels = np.append(orig_labels, test_labels[idxs], axis=0)

bim_params["nb_iter"] = 5
adv_images5 = bim.generate_np(orig_images, **bim_params)
bim_params["nb_iter"] = 10
adv_images10 = bim.generate_np(orig_images, **bim_params)

# Compute the average distortion introduced by the algorithm
percent_perturbed = np.mean(np.sum((adv_images5 - orig_images) ** 2,
                                   axis=(1, 2, 3)) ** .5)
print('Avg. L_2 norm of perturbations (5 iterations) {0:.4f}'.format(percent_perturbed))
percent_perturbed = np.mean(np.sum((adv_images10 - orig_images) ** 2,
                                   axis=(1, 2, 3)) ** .5)
print('Avg. L_2 norm of perturbations (10 iterations) {0:.4f}'.format(percent_perturbed))

# Evaluate accuracy on adversarial images
adv_loss, adv_acc5 = model.evaluate(adv_images5, orig_labels)
adv_loss, adv_acc10 = model.evaluate(adv_images10, orig_labels)
orig_loss, orig_acc = model.evaluate(orig_images, orig_labels)
print('Adv accuracy (5): {}, Adv accuracy (10): {}, Original accuracy: {}'.format(adv_acc5, adv_acc10, orig_acc))


# - Step 6 - Use your model trained in Step 3 to classify the adversarial examples.
adv_preds10 = model.predict(adv_images5)
adv_preds5 = model.predict(adv_images5)
orig_preds = model.predict(orig_images)

# Plot the 10 test images from each category, the true label, and their predicted labels.
# Color correct predictions in blue, incorrect predictions in red
fig = plt.figure(figsize=(5, 100))
for i, img in enumerate(orig_images):
    fig.add_subplot(100, 3, i * 3 + 1)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(np.squeeze(img, axis=2), cmap=plt.cm.binary)
    plt.ylabel("{}".format(class_names[int(orig_labels[i])]), rotation=0)
for i, img in enumerate(adv_images5):
    fig.add_subplot(100, 3, i * 3 + 2)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(np.squeeze(img, axis=2), cmap=plt.cm.binary)
    adv_pred_label = np.argmax(adv_preds5[i])
    if adv_pred_label == orig_labels[i]:
        color = 'blue'
    else:
        color = 'red'
    confidence = np.max(adv_preds5[i])
    if confidence < .99:
        plt.ylabel("{} {:.2f}".format(class_names[adv_pred_label],
                                      confidence), color=color, rotation=0)
    else:
        plt.ylabel("{}".format(class_names[adv_pred_label]), color=color, rotation=0)
for i, img in enumerate(adv_images10):
    fig.add_subplot(100, 3, i * 3 + 3)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(np.squeeze(img, axis=2), cmap=plt.cm.binary)
    adv_pred_label = np.argmax(adv_preds10[i])
    if adv_pred_label == orig_labels[i]:
        color = 'blue'
    else:
        color = 'red'
    confidence = np.max(adv_preds10[i])
    if confidence < .99:
        plt.ylabel("{} {:.2f}".format(class_names[adv_pred_label],
                                      confidence), color=color, rotation=0)
    else:
        plt.ylabel("{}".format(class_names[adv_pred_label]), color=color, rotation=0)

plt.savefig('output.png')
