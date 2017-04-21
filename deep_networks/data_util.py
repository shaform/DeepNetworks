"""Utils for data input/output"""
import glob
import itertools
import math
import os

from collections import defaultdict

import numpy as np
import tensorflow as tf


def gaussian_mixture(num_clusters=5,
                     scale=1.0,
                     stddev=0.2,
                     batch_size=64,
                     minval=None,
                     maxval=None,
                     name='gaussian_mixture'):
    """Gaussian mixture data set

    :param num_clusters: number of clusters
    :param scale: scale of the cluster centers
    :param stddev: stddev of the clusters
    :param batch_size: batch size of the resulting queue
    :param minval: select part of the clusters
    :param maxval: select part of the clusters
    :param name: name
    """
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
    """Scale image to target size and crop remaining part at center

    :param image: input image tensor
    :param target_height: target height
    :param target_width: target width
    """
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
    """Recursively list files

    :param file_path: file paths or directory paths
    :param allow_regex: filter paths by regex
    """
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
    """List files in directory as filename queue

    :param file_path: file paths or directory paths
    :param allow_regex: filter paths by regex
    :param num_epochs: epochs of the resulting queue
    """
    paths = list_files(file_path, allow_regex=allow_regex)
    filename_queue = tf.train.string_input_producer(
        paths, num_epochs=num_epochs)
    return filename_queue


def read_images(filename_queue, set_shape=(None, None, None)):
    """Read and decode image files

    :param filename_queue: filename queue
    :param set_shape: output tensor shape
    """
    reader = tf.WholeFileReader()
    _, value = reader.read(filename_queue)

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
    """Read images from TFRecords

    :param filename_queue: filename queue
    :param with_labels: whether to read labels as well
    :param target_height: resize to target_hight
    :param target_width: resize to target_with
    :param compression_type: compression of the tfrecords
    """
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


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def save_image_as_tfrecords(
        path,
        images,
        labels=None,
        num_examples=None,
        num_examples_per_label=None,
        compression_type=tf.python_io.TFRecordCompressionType.GZIP):
    """Convert all images to TFRecords and save to path

    :param path: output location
    :param images: input image numpy arrays
    :param labels: input image labels
    :param num_examples: total number of images
    :param compression_type: compression for tfrecords
    """
    if labels is None:
        labels = itertools.repeat(None)

    writer = tf.python_io.TFRecordWriter(
        path, options=tf.python_io.TFRecordOptions(compression_type))

    index = 0
    counts = defaultdict(int)
    for image, label in zip(images, labels):
        if num_examples_per_label is not None:
            if counts[label] < num_examples_per_label:
                counts[label] += 1
            else:
                continue
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
    return index


def norm_image(image):
    """Normalize a image tensor of range [0, 255] to [-1.0, 1.0]

    :param image: input tensor
    """
    return tf.cast(image, tf.float32) / 127.5 - 1.


def denorm_image(image):
    """Undo normalization of a tensor of range [-1.0, 1.0] to [0, 255]

    :param image: input tensor
    """
    return tf.cast((image + 1.) * 127.5, tf.uint8)


def np_norm_image(image):
    """Normalize a image numpy array of range [0, 255] to [-1.0, 1.0]

    :param image: input numpy array
    """
    return image.astype(np.float32) / 127.5 - 1.


def np_denorm_image(image):
    """Undo normalization of a numpy array of range [-1.0, 1.0] to [0, 255]

    :param image: input numpy array
    """
    return ((image + 1.) * 127.5).astype(np.uint8)
