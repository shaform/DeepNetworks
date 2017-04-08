"""convert_to_records

Convert images to TFRecords
"""

import argparse

import numpy as np
import tensorflow as tf
import tqdm

from scipy.misc import imresize
from tensorflow.examples.tutorials.mnist import input_data

from deep_networks import data_util


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', help='Directory of images')
    parser.add_argument('outfile', help='Output tfrecords')
    parser.add_argument(
        '--use-mnist', action='store_true', help='Use MNIST dataset')
    parser.add_argument(
        '--target-height', type=int, default=64, help='Target height')
    parser.add_argument(
        '--target-width', type=int, default=64, help='Target width')
    return parser.parse_args()


def read_mnist(path):
    """Read MNIST images and normalize the data

    :param path: directory of MNIST data
    """
    mnist = input_data.read_data_sets(path)
    images = (mnist.train.images.reshape((-1, 28,
                                          28)) * 255.0).astype(np.uint8)
    labels = mnist.train.labels
    return images, labels


def main():
    args = parse_args()
    if args.use_mnist:
        images, labels = read_mnist(args.directory)

        if args.target_width != 28 or args.target_width != 28:
            images = np.vstack([
                imresize(image, (args.target_height, args.target_width))
                for image in images
            ])
        images = images.reshape((-1, args.target_height, args.target_width, 1))

        data_util.save_image_as_tfrecords(args.outfile, images, labels)
    else:
        filename_queue = data_util.list_files_as_filename_queue(
            args.directory, num_epochs=1)
        image = data_util.read_images(filename_queue)
        image = data_util.crop_and_resize(
            image,
            target_height=args.target_height,
            target_width=args.target_width)

        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        with tf.Session() as sess:
            sess.run(init_op)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            def _produce_images():
                try:
                    while True:
                        yield sess.run(image)
                except tf.errors.OutOfRangeError:
                    pass

            data_util.save_image_as_tfrecords(args.outfile,
                                              tqdm.tqdm(_produce_images()))

            coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':
    main()
