import tensorflow as tf
from tensorflow import keras

import numpy as np
import os
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

tf.random.set_seed(42)
np.random.seed(42)
batch_size = 32
codings_size = 100
data_dir_letters = 'data/notMNIST_small'

# Load images from one folder (A to J) in a numpy array
def notMNIST_load_data(letter) : 
    x_train = []
    for image in glob.iglob(os.path.join(data_dir_letters, letter, "*.png")):
        pixels_array = mpimg.imread(image)
        x_train.append(pixels_array)
    return np.array(x_train)

# Build generateur
def build_generator():
    model = keras.models.Sequential([
        keras.layers.Dense(7 * 7 * 128, input_shape=[codings_size]),
        keras.layers.Reshape([7, 7, 128]),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2DTranspose(64, kernel_size=5, strides=2, padding="SAME", activation="selu"),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2DTranspose(1, kernel_size=5, strides=2, padding="SAME", activation="tanh"),
    ])
    return model

# Build discriminateur
def build_discriminator():
    model = keras.models.Sequential([
        keras.layers.Conv2D(64, kernel_size=5, strides=2, padding="SAME", activation=keras.layers.LeakyReLU(0.2), input_shape=[28, 28, 1]),
        keras.layers.Dropout(0.4),
        keras.layers.Conv2D(128, kernel_size=5, strides=2, padding="SAME", activation=keras.layers.LeakyReLU(0.2)),
        keras.layers.Dropout(0.4),
        keras.layers.Flatten(),
        keras.layers.Dense(1, activation="sigmoid")
    ])
    return model

# Build gan (generator + discriminator)
def build_gan(generator, discriminator):
    gan = keras.models.Sequential([generator, discriminator])
    discriminator.compile(loss="binary_crossentropy", optimizer="rmsprop")
    discriminator.trainable = False
    gan.compile(loss="binary_crossentropy", optimizer="rmsprop")
    return gan

# Save generated images individually
def save_generated_images(save_path, generated_images, letter, arg):
    for i, generated_image in enumerate(generated_images):
        img = tf.keras.preprocessing.image.array_to_img(generated_image)
        img.save(os.path.join(save_path, "{}_generated_{}_{}.png".format(os.path.basename(letter), arg, i)))

# Train gan on each letter and save generated images individually
def train_gan(save_path, letter, gan, dataset, batch_size, codings_size, n_epochs=50):
    generator, discriminator = gan.layers
    for epoch in range(n_epochs):
        print("Epoch {}/{}".format(epoch + 1, n_epochs)) 

        for X_batch in dataset:
            # phase 1 - training the discriminator
            noise = tf.random.normal(shape=[batch_size, codings_size])
            generated_images = generator(noise)
            X_fake_and_real = tf.concat([generated_images, X_batch], axis=0)
            y1 = tf.constant([[0.]] * batch_size + [[1.]] * batch_size)
            discriminator.trainable = True
            discriminator.train_on_batch(X_fake_and_real, y1)

            # phase 2 - training the generator
            noise = tf.random.normal(shape=[batch_size, codings_size])
            y2 = tf.constant([[1.]] * batch_size)
            discriminator.trainable = False
            gan.train_on_batch(noise, y2)
            
        # Save generated images individually
        save_generated_images(save_path, generated_images, letter, epoch)


# Training and generating images for each letter (A to J)
for letter in os.listdir(data_dir_letters):
    X_train = notMNIST_load_data(letter)
    X_train_dcgan = X_train.reshape(-1, 28, 28, 1) * 2. - 1. # reshape and rescale

    dataset = tf.data.Dataset.from_tensor_slices(X_train_dcgan)
    dataset = dataset.shuffle(1000)
    dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(1)

    save_path = os.path.join('data/generated', os.path.basename(letter))
    os.makedirs(save_path, exist_ok=True)

    generator = build_generator()
    discriminator = build_discriminator()
    gan = build_gan(generator, discriminator)
    train_gan(save_path, letter, gan, dataset, batch_size, codings_size)

    noise = tf.random.normal(shape=[batch_size, codings_size])
    generated_images = generator(noise)

    # Save generated images individually
    save_generated_images(save_path, generated_images, letter, os.path.basename(letter))
