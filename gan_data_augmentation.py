# Ce programme permet de générer des images à partir d'un dataset d'images de lettres (A à J) en utilisant un modèle GAN.
# Les images générées sont sauvegardées dans un dossier data/notMNIST_small_augmented qui contient un dossier par lettre.
# Le modèle GAN est entrainé sur chaque lettre du dataset et génère des images de la lettre correspondante.
# Un modèle GAN est composé d'un générateur et d'un discriminateur, qui sont des réseaux de neurones.
# Le générateur prend en entrée un vecteur de nombres aléatoires et génère une image.
# Le discriminateur prend en entrée une image et prédit si elle est réelle ou générée.
# Le générateur et le discriminateur sont entrainés en même temps. 
# Le générateur essaye de tromper le discriminateur en générant des images qui ressemblent à des images réelles.
# Le discriminateur essaye de distinguer les images réelles des images générées par le générateur.
# Au fur et à mesure de l'entrainement, le générateur génère des images de plus en plus réalistes.
# Les deux réseaux de neurones s'améliorent mutuellement.


# Importation des librairies
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


# Paramètres
tf.random.set_seed(42)
np.random.seed(42)
batch_size = 32
codings_size = 100
data_dir_letters = 'data/notMNIST_small'
nb_epochs = 50


# On charge les images d'un dossier (A à J) dans un tableau numpy
def notMNIST_load_data(letter) : 
    x_train = []
    for image in glob.iglob(os.path.join(data_dir_letters, letter, "*.png")):
        pixels_array = mpimg.imread(image)
        x_train.append(pixels_array)
    return np.array(x_train)

# Création du générateur
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

# Création du discriminateur
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

# Création du modèle GAN (générateur + discriminateur)
def build_gan(generator, discriminator):
    gan = keras.models.Sequential([generator, discriminator])
    discriminator.compile(loss="binary_crossentropy", optimizer="rmsprop")
    discriminator.trainable = False
    gan.compile(loss="binary_crossentropy", optimizer="rmsprop")
    return gan

# Sauvegarde des images générées individuellement
def save_generated_images(save_path, generated_images, letter, arg):
    for i, generated_image in enumerate(generated_images):
        img = tf.keras.preprocessing.image.array_to_img(generated_image)
        img.save(os.path.join(save_path, "{}_generated_{}_{}.png".format(os.path.basename(letter), arg, i)))

# Entrainement du modèle GAN sur chaque lettre du dataset et sauvegarde des images générées individuellement
def train_gan(save_path, letter, gan, dataset, batch_size, codings_size, n_epochs=nb_epochs):
    generator, discriminator = gan.layers
    for epoch in range(n_epochs):
        print("Epoch {}/{}".format(epoch + 1, n_epochs)) 

        for X_batch in dataset:
            # phase 1 - entraînement du discriminateur
            noise = tf.random.normal(shape=[batch_size, codings_size])
            generated_images = generator(noise)
            X_fake_and_real = tf.concat([generated_images, X_batch], axis=0)
            y1 = tf.constant([[0.]] * batch_size + [[1.]] * batch_size)
            discriminator.trainable = True
            discriminator.train_on_batch(X_fake_and_real, y1)

            # phase 2 - entraînement du générateur
            noise = tf.random.normal(shape=[batch_size, codings_size])
            y2 = tf.constant([[1.]] * batch_size)
            discriminator.trainable = False
            gan.train_on_batch(noise, y2)
            
        # Sauvegarde des images générées individuellement
        save_generated_images(save_path, generated_images, letter, epoch)


# Entrainement du modèle GAN sur chaque lettre du dataset et sauvegarde des images générées individuellement
for letter in os.listdir(data_dir_letters):
    X_train = notMNIST_load_data(letter)
    X_train_dcgan = X_train.reshape(-1, 28, 28, 1) * 2. - 1.

    dataset = tf.data.Dataset.from_tensor_slices(X_train_dcgan)
    dataset = dataset.shuffle(1000)
    dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(1)

    save_path = os.path.join('data/notMNIST_small_augmented', os.path.basename(letter))
    os.makedirs(save_path, exist_ok=True)

    generator = build_generator()
    discriminator = build_discriminator()
    gan = build_gan(generator, discriminator)
    train_gan(save_path, letter, gan, dataset, batch_size, codings_size)

    noise = tf.random.normal(shape=[batch_size, codings_size])
    generated_images = generator(noise)

    # Sauvegarde des images générées individuellement
    save_generated_images(save_path, generated_images, letter, os.path.basename(letter))
