# -*- coding: utf-8 -*-
from keras.optimizers import Adam
from keras.datasets import mnist
import matplotlib.pyplot as plt

import numpy


def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = (x_train.astype(numpy.float32) - 127.5) / 127.5

    # convert shape of x_train from (60000, 28, 28) to (60000, 784)
    # 784 columns per row
    x_train = x_train.reshape(60000, 784)
    return x_train, y_train, x_test, y_test


def adam_optimizer():
    return Adam(lr=0.0002, beta_1=0.5)


def plot_generated_images(epoch, generator, examples=100, dim=(10, 10),
                          fig_size=(10, 10)):
    noise = numpy.random.normal(loc=0, scale=1, size=[examples, 100])
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(100, 28, 28)
    plt.figure(figsize=fig_size)

    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i], interpolation='nearest')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('../data/2020-04-02/gan_generated_{}.png'.format(epoch))
