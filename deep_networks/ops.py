"""Tensor operations"""
import tensorflow as tf


def lrelu(x, leak=0.2, name='lrelu'):
    """Leaky ReLU"""
    with tf.name_scope(name):
        return tf.nn.relu(x) - leak * tf.nn.relu(-x)
