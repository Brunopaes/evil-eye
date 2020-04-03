# -*- coding: utf-8 -*-
from tqdm import tqdm

import helpers
import models
import numpy


def training(epochs=1, batch_size=128):
    # Loading the data
    (X_train, y_train, X_test, y_test) = helpers.load_data()
    batch_count = X_train.shape[0] / batch_size

    # Creating GAN
    generator = models.create_generator()
    discriminator = models.create_discriminator()
    gan = models.create_gan(discriminator, generator)

    for e in range(1, epochs + 1):
        for _ in tqdm(range(batch_size), desc="Epoch %d" % e):
            # generate  random noise as an input  to  initialize the  generator
            noise = numpy.random.normal(0, 1, [batch_size, 100])

            # Generate fake MNIST images from noised input
            generated_images = generator.predict(noise)

            # Get a random set of  real images
            image_batch = X_train[
                numpy.random.randint(low=0, high=X_train.shape[0],
                                     size=batch_size)]

            # Construct different batches of  real and fake data
            x = numpy.concatenate([image_batch, generated_images])

            # Labels for generated and real data
            y_dis = numpy.zeros(2 * batch_size)
            y_dis[:batch_size] = 0.9

            # Pre train discriminator on  fake and real data  before starting the gan.
            discriminator.trainable = True
            discriminator.train_on_batch(x, y_dis)

            # Tricking the noised input of the Generator as real data
            noise = numpy.random.normal(0, 1, [batch_size, 100])
            y_gen = numpy.ones(batch_size)

            # During the training of gan,
            # the weights of discriminator should be fixed.
            # We can enforce that by setting the trainable flag
            discriminator.trainable = False

            # training  the GAN by alternating the training of the Discriminator
            # and training the chained GAN model with Discriminator’s weights freezed.
            gan.train_on_batch(noise, y_gen)

        if e == 1 or e % 20 == 0:
            helpers.plot_generated_images(e, generator)


if __name__ == '__main__':
    training(400, 128)
