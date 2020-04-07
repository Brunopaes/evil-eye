# -*- coding: utf-8 -*-
from tqdm import tqdm

import helpers
import models
import numpy


def training(epochs=1, batch_size=128):
    (X_train, y_train, X_test, y_test) = helpers.load_data()

    generator = models.create_generator()
    discriminator = models.create_discriminator()
    gan = models.create_gan(discriminator, generator)

    for e in range(1, epochs + 1):
        for _ in tqdm(range(batch_size), desc="Epoch {}".format(e)):
            noise = numpy.random.normal(0, 1, [batch_size, 100])

            generated_images = generator.predict(noise)

            image_batch = X_train[
                numpy.random.randint(low=0, high=X_train.shape[0],
                                     size=batch_size)]

            x = numpy.concatenate([image_batch, generated_images])

            y_dis = numpy.zeros(2 * batch_size)
            y_dis[:batch_size] = 0.9

            discriminator.trainable = True
            discriminator.train_on_batch(x, y_dis)

            noise = numpy.random.normal(0, 1, [batch_size, 100])
            y_gen = numpy.ones(batch_size)

            discriminator.trainable = False

            gan.train_on_batch(noise, y_gen)

        if e == 1 or e % 20 == 0:
            helpers.plot_generated_images(e, generator)


if __name__ == '__main__':
    training(400, 128)
