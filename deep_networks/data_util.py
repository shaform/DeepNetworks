import glob
import itertools
import math
import os

import numpy as np
import tensorflow as tf


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def gaussian_mixture(num_clusters=5,
                     scale=1.0,
                     stddev=0.2,
                     batch_size=64,
                     minval=None,
                     maxval=None,
                     name='gaussian_mixture'):
    with tf.name_scope(name):
        if minval is None:
            minval = 0
        if maxval is None:
            maxval = num_clusters
        assert minval < maxval
        mixture_indices = tf.random_uniform(
            (batch_size, 1), minval=minval, maxval=maxval, dtype=tf.int64)
        mixture_indices = tf.mod(mixture_indices, num_clusters)
        angles = tf.cast(mixture_indices,
                         tf.float32) / num_clusters * 2 * math.pi + math.pi / 2
        means = tf.concat([tf.cos(angles), tf.sin(angles)], 1)
        X = means * scale + tf.random_normal(
            (batch_size, 2), mean=0.0, stddev=stddev)
        y = tf.squeeze(mixture_indices)
        return X, y


def crop_and_resize(image, target_height, target_width):
    image_height = tf.shape(image)[0]
    image_width = tf.shape(image)[1]
    scale = tf.maximum(target_height * image_width,
                       target_width * image_height)
    size = tf.stack([scale // image_width, scale // image_height])
    image = tf.image.resize_images(image, size)
    image = tf.image.resize_image_with_crop_or_pad(image, target_height,
                                                   target_width)
    image = tf.cast(image, tf.uint8)
    return image


def list_files(file_path, allow_regex=None):
    if isinstance(file_path, list):
        paths = []
        for path in file_path:
            paths.extend(list_files(path, allow_regex=allow_regex))
        return paths
    else:
        if os.path.isdir(file_path):
            paths = [path for path in glob.glob(os.path.join(file_path, '*'))]
        else:
            paths = [file_path]
        if allow_regex:
            paths = [path for path in paths if allow_regex.match(path)]
        return paths


def list_files_as_filename_queue(file_path, allow_regex=None, num_epochs=None):
    paths = list_files(file_path, allow_regex=allow_regex)
    filename_queue = tf.train.string_input_producer(
        paths, num_epochs=num_epochs)
    return filename_queue


def read_images(filename_queue, set_shape=(None, None, None)):
    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)

    image = tf.image.decode_image(value)
    if set_shape:
        image.set_shape(set_shape)
    return image


def read_image_from_tfrecords(
        filename_queue,
        with_labels=False,
        target_height=None,
        target_width=None,
        compression_type=tf.python_io.TFRecordCompressionType.GZIP):
    reader = tf.TFRecordReader(
        options=tf.python_io.TFRecordOptions(compression_type))

    _, serialized_example = reader.read(filename_queue)

    features = {
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'channel': tf.FixedLenFeature([], tf.int64),
        'image_raw': tf.FixedLenFeature([], tf.string),
    }
    if with_labels:
        features['label'] = tf.FixedLenFeature([], tf.int64)

    example = tf.parse_single_example(serialized_example, features=features)

    image = tf.decode_raw(example['image_raw'], tf.uint8)

    height = tf.cast(example['height'], tf.int32)
    width = tf.cast(example['width'], tf.int32)
    channel = tf.cast(example['channel'], tf.int32)

    image_shape = tf.stack([height, width, channel])
    image = tf.reshape(image, image_shape)

    if target_height is None:
        target_height = height
    if target_width is None:
        target_width = width

    if target_height != height or target_width != width:
        image = tf.image.resize_images(image, (target_height, target_width))

    if with_labels:
        label = tf.cast(example['label'], tf.int64)
        return image, label
    else:
        return image


def save_image_as_tfrecords(
        path,
        images,
        labels=None,
        num_examples=None,
        compression_type=tf.python_io.TFRecordCompressionType.GZIP):
    if labels is None:
        labels = itertools.repeat(None)

    writer = tf.python_io.TFRecordWriter(
        path, options=tf.python_io.TFRecordOptions(compression_type))

    index = 0
    for image, label in zip(images, labels):
        height, width, channel = image.shape
        image_raw = image.tostring()
        feature = {
            'height': _int64_feature(height),
            'width': _int64_feature(width),
            'channel': _int64_feature(channel),
            'image_raw': _bytes_feature(image_raw),
        }

        if label is not None:
            feature['label'] = _int64_feature(int(label))

        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())

        index += 1
        if num_examples is not None and index >= num_examples:
            break
    writer.close()


def norm_image(image):
    return tf.cast(image, tf.float32) / 127.5 - 1.


def denorm_image(image):
    return tf.cast((image + 1.) * 127.5, tf.uint8)


def np_norm_image(image):
    return image.astype(np.float32) / 127.5 - 1.


def np_denorm_image(image):
    return ((image + 1.) * 127.5).astype(np.uint8)
