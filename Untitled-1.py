# %%
# Import all the necessary files!
import os
import tensorflow as tf
import keras
from tensorflow.keras import layers
from tensorflow.keras import Model
from os import getcwd

# %%
# Selecting the dataset as CIFAR10
cifar10 = tf.keras.datasets.cifar10

# %%
#Train and Test Images Partitioning along with their labels
(training_images, training_labels), (test_images, test_labels) = cifar10.load_data()

# %%
#Finding total number of images in training and test set
print(len(training_images))
print(len(test_images))

# %%
#Shape of the image before reshaping
training_images.shape

# %%
training_images = training_images.reshape(50000, 1024, 3)

# %%
#Shape of the image after reshaping
print(training_images[1].shape)
print(training_labels[1])

# %%
#Reshaping and Normalizing training and test images
training_images = training_images.reshape(50000, 1024, 3)
training_images = training_images/255.0
test_images = test_images.reshape(10000, 1024, 3)
test_images = test_images/255.0

# %%
model = tf.keras.models.Sequential([
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, input_shape=(1024, 3), return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
        ])

# %%
model.compile(optimizer='adam',
              loss = 'sparse_categorical_crossentropy',
              metrics=["acc"])
model.fit(training_images, training_labels, batch_size = 50, epochs=20)

# %%



