import argparse

import tensorflow as tf
import tqdm

from deep_networks import data_util


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', help='Directory of images')
    parser.add_argument('outfile', help='Output tfrecords')
    parser.add_argument(
        '--target-height', type=int, default=64, help='Target height')
    parser.add_argument(
        '--target-width', type=int, default=64, help='Target width')
    return parser.parse_args()


def main():
    args = parse_args()
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

        def produce_images():
            try:
                while True:
                    yield sess.run(image)
            except tf.errors.OutOfRangeError:
                pass

        data_util.save_image_as_tfrecords(args.outfile,
                                          tqdm.tqdm(produce_images()))

        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    main()