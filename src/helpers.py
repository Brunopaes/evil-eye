# -*- coding: utf-8 -*-
from keras.optimizers import Adam
from keras.datasets import mnist

import matplotlib.pyplot as plt
import datetime
import filecmp
import numpy
import cv2
import os


def load_data():
    """This function loads the mnist dataset.

    Returns
    -------

    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = (x_train.astype(numpy.float32) - 127.5) / 127.5
    x_train = x_train.reshape(60000, 784)

    return x_train, y_train, x_test, y_test


def adam_optimizer():
    """Adam optmizer used in Keras models.

    Returns
    -------

    """
    return Adam(lr=0.0002, beta_1=0.5)


def plot_generated_images(epoch, generator, examples=100, dim=(10, 10),
                          fig_size=(10, 10)):
    """This function plots GAN-generated images.

    Parameters
    ----------
    epoch : int
        Training epoch.
    generator :  keras.engine.sequential.Sequential
        GAN-generator model.
    examples : int
        Number of examples.
    dim : tuple
        Figure dimension.
    fig_size : tuple
        Figure size.

    Returns
    -------

    """
    noise = numpy.random.normal(loc=0, scale=1, size=[examples, 100])
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(100, 28, 28)
    plt.figure(figsize=fig_size)

    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i], interpolation='nearest')
        plt.axis('off')

    plt.tight_layout()
    path = (os.path.abspath('../data/generated_data/{}'.
                            format(datetime.date.today())))
    if not os.path.exists(path):
        os.makedirs(path)

    plt.savefig('{}/gan_generated_{}.png'.format(path, epoch))


def resize(max_width, max_height, image_path, out_dir):
    """This function resizes images into a directory.

    Parameters
    ----------
    max_width : int
        Max resized width.
    max_height : int
        Max resized height.
    image_path : str
        Images path.
    out_dir : str
        Goal directory.

    Returns
    -------

    """
    dir_path = os.path.dirname(os.path.abspath(__file__))

    image_path = os.path.join(dir_path, image_path)
    images = os.listdir(image_path)

    counter = 0
    for im in images:
        image = cv2.imread(os.path.join(image_path, im))
        height = image.shape[0]
        width = image.shape[1]

        if width > max_width:
            ratio = max_width / width
            new_height = int(ratio * height)
            image = cv2.resize(image, (max_width, new_height))

        if new_height > max_height:
            ratio = max_height / new_height
            new_width = int(ratio * max_width)
            image = cv2.resize(image, (new_width, max_height))

        out_path = os.path.join(out_dir, im)
        cv2.imwrite(out_path, image)
        counter += 1


def comparing(path):
    """This function compares files into a given directory.

    Parameters
    ----------
    path : str
        Directory path.
    Returns
    -------

    """
    file_list = os.listdir(path)
    for file_1 in file_list:
        for file_2 in list(set(os.listdir(path)) - {file_1}):
            if filecmp.cmp(
                os.path.join(path, file_1),
                os.path.join(path, file_2)
            ):
                print('{} == {}: {}'.format(file_1, file_2, True))

        file_list.pop(file_list.index(file_1))
