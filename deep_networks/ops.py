"""Tensor operations"""
import tensorflow as tf


def lrelu(x, leak=0.2, name='lrelu'):
    """Leaky ReLU"""
    with tf.name_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * tf.abs(x)
