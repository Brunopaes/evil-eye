# -*- coding: utf-8 -*-
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Dense, Dropout, Input
from keras.models import Model, Sequential

import helpers


def create_generator():
    """This function instantiates the generator model.

    Returns
    -------
    generator : keras.engine.sequential.Sequential
        The generator model.

    """
    generator = Sequential()
    generator.add(Dense(units=256, input_dim=100))
    generator.add(LeakyReLU(0.2))

    generator.add(Dense(units=512))
    generator.add(LeakyReLU(0.2))

    generator.add(Dense(units=1024))
    generator.add(LeakyReLU(0.2))

    generator.add(Dense(units=784, activation='tanh'))

    generator.compile(loss='binary_crossentropy',
                      optimizer=helpers.adam_optimizer())

    return generator


def create_discriminator():
    """This function instantiates the discriminator model.

    Returns
    -------
    discriminator : keras.engine.sequential.Sequential
        The discriminator model.

    """
    discriminator = Sequential()
    discriminator.add(Dense(units=1024, input_dim=784))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))

    discriminator.add(Dense(units=512))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))

    discriminator.add(Dense(units=256))
    discriminator.add(LeakyReLU(0.2))

    discriminator.add(Dense(units=1, activation='sigmoid'))

    discriminator.compile(loss='binary_crossentropy',
                          optimizer=helpers.adam_optimizer())
    return discriminator


def create_gan(discriminator, generator):
    """Using discriminator and generator models, instantiate a gan model.

    Parameters
    ----------
    discriminator : keras.engine.sequential.Sequential
        The discriminator model.

    generator : keras.engine.sequential.Sequential
        The generator model.

    Returns
    -------
    gan : keras.engine.sequential.Sequential
        Gan model.

    """
    discriminator.trainable = False

    gan_input = Input(shape=(100, ))
    gan_output = discriminator(generator(gan_input))

    gan = Model(inputs=gan_input, outputs=gan_output)
    gan.compile(loss='binary_crossentropy', optimizer='adam')

    return gan
