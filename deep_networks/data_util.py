import math

import tensorflow as tf


def gaussian_mixture(num_cluster=5,
                     scale=1.0,
                     stddev=0.2,
                     batch_size=64,
                     name='gaussian_mixture'):
    with tf.variable_scope(name):
        mixture_indices = tf.random_uniform(
            (batch_size, 1), minval=0, maxval=num_cluster, dtype=tf.int32)
        angles = tf.cast(mixture_indices,
                         tf.float32) / num_cluster * 2 * math.pi + math.pi / 2
        means = tf.concat([tf.cos(angles), tf.sin(angles)], 1)
        return means * scale + tf.random_normal(
            (batch_size, 2), mean=0.0, stddev=stddev)
