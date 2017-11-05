import functools
import logging
import operator

import tensorflow as tf

from ..layers import conv2d_with_weight_norm
from ..layers import conv2d_transpose_with_weight_norm
from ..layers import dense_with_weight_norm
from ..ops import conv2d_subpixel
from ..ops import opt_activation
from ..ops import std_eps
from .base import BaseDiscriminator
from .base import BaseGenerator
from .base import BaseImageDiscriminator
from .base import BaseImageGenerator

logger = logging.getLogger(__name__)


class BasicGenerator(BaseGenerator):
    """BasicGenerator

    A generator with only fully-connected layers.
    """

    def __init__(self,
                 inputs,
                 output_shape,
                 c=None,
                 initializer=tf.contrib.layers.xavier_initializer(
                     uniform=False),
                 dim=300,
                 num_layers=3,
                 activation_fn=None,
                 name='generator',
                 reuse=False):
        assert num_layers > 0
        self.output_shape = output_shape
        self.output_size = functools.reduce(operator.mul, output_shape)

        with tf.variable_scope(name, reuse=reuse) as scope:
            super().__init__(scope, reuse)
            self.log_name()

            if c is not None:
                inputs = self.build_latents(inputs, c)

            outputs = inputs
            for i in range(num_layers - 1):
                with tf.variable_scope('fc{}'.format(i + 1)):
                    outputs = dense_with_weight_norm(
                        inputs=outputs,
                        units=dim,
                        activation=tf.nn.relu,
                        kernel_initializer=initializer,
                        use_bias=True,
                        bias_initializer=tf.zeros_initializer(),
                        scale=True)
                    self.log_msg('WN-FC %d-Relu', dim)

            with tf.variable_scope('outputs'):
                self.outputs = dense_with_weight_norm(
                    inputs=outputs,
                    units=self.output_size,
                    activation=None,
                    kernel_initializer=initializer,
                    use_bias=True,
                    bias_initializer=tf.zeros_initializer())
                self.activations = opt_activation(self.outputs, activation_fn)
                self.log_msg('WN-FC %d', self.output_size)


class BasicDiscriminator(BaseDiscriminator):
    """BasicDiscriminator

    A discriminator with only fully-connected layers.
    """

    def __init__(self,
                 inputs,
                 input_shape=None,
                 num_classes=None,
                 initializer=tf.contrib.layers.xavier_initializer(
                     uniform=False),
                 regularizer=None,
                 dim=300,
                 num_layers=3,
                 disc_activation_fn=tf.nn.sigmoid,
                 cls_activation_fn=tf.nn.softmax,
                 name='discriminator',
                 reuse=False):
        assert num_layers > 0
        self.inputs = inputs
        self.input_shape = input_shape
        self.input_size = functools.reduce(operator.mul, input_shape)
        with tf.variable_scope(name, reuse=reuse) as scope:
            super().__init__(scope, reuse)
            self.log_name()
            outputs = inputs
            self.features = []
            for i in range(num_layers - 1):
                with tf.variable_scope('fc{}'.format(i + 1)):
                    if i == num_layers - 2:
                        stds = std_eps(outputs)
                        stds = tf.tile(stds,
                                       tf.concat(
                                           [tf.shape(outputs)[:-1], [1]],
                                           axis=0))
                        outputs = tf.concat([outputs, stds], axis=-1)
                    outputs = dense_with_weight_norm(
                        inputs=outputs,
                        units=dim,
                        activation=tf.nn.leaky_relu,
                        kernel_initializer=initializer,
                        kernel_regularizer=regularizer,
                        use_bias=True,
                        bias_initializer=tf.zeros_initializer())
                    self.features.append(outputs)
                    self.log_msg('WN-FC %d-LRelu', dim)

            with tf.variable_scope('disc_outputs'):
                self.disc_outputs = dense_with_weight_norm(
                    inputs=outputs,
                    units=1,
                    activation=None,
                    kernel_initializer=initializer,
                    kernel_regularizer=regularizer,
                    use_bias=True,
                    bias_initializer=tf.zeros_initializer())
                self.disc_activations = opt_activation(self.disc_outputs,
                                                       disc_activation_fn)
                self.log_msg('WN-FC %d-LRelu (disc_outputs)', 1)

            if num_classes is not None:
                with tf.variable_scope('cls_outputs'):
                    self.cls_outputs = dense_with_weight_norm(
                        inputs=outputs,
                        units=num_classes,
                        activation=None,
                        kernel_initializer=initializer,
                        kernel_regularizer=regularizer,
                        use_bias=True,
                        bias_initializer=tf.zeros_initializer())
                    self.cls_activations = opt_activation(
                        self.cls_outputs, cls_activation_fn)
                    self.log_msg('WN-FC %d-LRelu (cls_outputs)', num_classes)


class ConvTransposeGenerator(BaseImageGenerator):
    """ConvTransposeGenerator

    A generator with transpose convolutions.
    """

    def __init__(self,
                 inputs,
                 output_shape,
                 c=None,
                 initializer=tf.contrib.layers.xavier_initializer(
                     uniform=False),
                 regularizer=None,
                 min_size=4,
                 min_dim=16,
                 max_dim=512,
                 activation_fn=tf.nn.tanh,
                 name='generator',
                 reuse=False):

        self.output_shape = output_shape
        self.output_size = functools.reduce(operator.mul, output_shape)
        start_shape, upsamples = self.compute_upsamples(
            output_shape, min_size, min_dim, max_dim)
        channels = output_shape[2]

        with tf.variable_scope(name, reuse=reuse) as scope:
            super().__init__(scope, reuse)
            self.log_name()

            if c is not None:
                inputs = self.build_latents(inputs, c)

            outputs = inputs
            with tf.variable_scope('fc'):
                outputs = dense_with_weight_norm(
                    inputs=outputs,
                    units=start_shape[0] * start_shape[1] * upsamples[0],
                    kernel_initializer=initializer,
                    activation=tf.nn.relu,
                    use_bias=True,
                    bias_initializer=tf.zeros_initializer(),
                    scale=True)
                outputs = tf.reshape(outputs, (-1, start_shape[0],
                                               start_shape[1], upsamples[0]))
                self.log_msg('WN-FC %dx%dx%d-Relu', start_shape[0],
                             start_shape[1], upsamples[0])

            for i, dim in enumerate(upsamples[1:]):
                with tf.variable_scope('conv_transpose_{}'.format(i + 1)):
                    outputs = conv2d_transpose_with_weight_norm(
                        inputs=outputs,
                        filters=dim,
                        kernel_size=(3, 3),
                        strides=(2, 2),
                        padding='same',
                        activation=tf.nn.relu,
                        kernel_initializer=initializer,
                        use_bias=True,
                        bias_initializer=tf.zeros_initializer(),
                        scale=True)
                    self.log_msg('WN-CONV-T k3n%ds2-Relu', dim)

            with tf.variable_scope('outputs'):
                outputs = conv2d_with_weight_norm(
                    inputs=outputs,
                    filters=channels,
                    kernel_size=(1, 1),
                    strides=(1, 1),
                    padding='same',
                    activation=None,
                    kernel_initializer=initializer,
                    use_bias=True,
                    bias_initializer=tf.zeros_initializer(),
                    scale=True)
                self.outputs = tf.layers.flatten(outputs)
                self.activations = opt_activation(self.outputs, activation_fn)
                self.log_msg('WN-CONV k1n%ds1', channels)


class SubpixelConvGenerator(BaseImageGenerator):
    """SubpixelConvGenerator

    A generator with subpixel convolutions.
    """

    def __init__(self,
                 inputs,
                 output_shape,
                 c=None,
                 initializer=tf.contrib.layers.xavier_initializer(
                     uniform=False),
                 regularizer=None,
                 min_size=4,
                 min_dim=16,
                 max_dim=512,
                 activation_fn=tf.nn.tanh,
                 name='generator',
                 reuse=False):

        self.output_shape = output_shape
        self.output_size = functools.reduce(operator.mul, output_shape)
        start_shape, upsamples = self.compute_upsamples(
            output_shape, min_size, min_dim, max_dim)
        channels = output_shape[2]

        with tf.variable_scope(name, reuse=reuse) as scope:
            super().__init__(scope, reuse)
            self.log_name()

            if c is not None:
                inputs = self.build_latents(inputs, c)

            outputs = inputs
            with tf.variable_scope('fc'):
                outputs = dense_with_weight_norm(
                    inputs=outputs,
                    units=start_shape[0] * start_shape[1] * upsamples[0],
                    kernel_initializer=initializer,
                    activation=tf.nn.relu,
                    use_bias=True,
                    bias_initializer=tf.zeros_initializer(),
                    scale=True)
                outputs = tf.reshape(outputs, (-1, start_shape[0],
                                               start_shape[1], upsamples[0]))
                self.log_msg('WN-FC %dx%dx%d-Relu', start_shape[0],
                             start_shape[1], upsamples[0])

            for i, dim in enumerate(upsamples[1:]):
                with tf.variable_scope('conv_subpixel_{}'.format(i + 1)):
                    outputs = conv2d_with_weight_norm(
                        inputs=outputs,
                        filters=dim,
                        kernel_size=(3, 3),
                        strides=(1, 1),
                        padding='same',
                        activation=None,
                        kernel_initializer=initializer,
                        use_bias=True,
                        bias_initializer=tf.zeros_initializer(),
                        scale=True)
                    outputs = conv2d_subpixel(inputs=outputs, scale=2)
                    outputs = tf.nn.relu(outputs)
                    self.log_msg('WN-CONV-Subpixel k3n%ds1-Relu', dim)

            with tf.variable_scope('outputs'):
                outputs = conv2d_with_weight_norm(
                    inputs=outputs,
                    filters=channels,
                    kernel_size=(1, 1),
                    strides=(1, 1),
                    padding='same',
                    activation=None,
                    kernel_initializer=initializer,
                    use_bias=True,
                    bias_initializer=tf.zeros_initializer(),
                    scale=True)
                self.outputs = tf.layers.flatten(outputs)
                self.activations = opt_activation(self.outputs, activation_fn)
                self.log_msg('WN-CONV k1n%ds1', channels)


class ConvDiscriminator(BaseImageDiscriminator):
    def __init__(self,
                 inputs,
                 input_shape,
                 num_classes=None,
                 initializer=tf.contrib.layers.xavier_initializer(
                     uniform=False),
                 regularizer=None,
                 min_size=4,
                 min_dim=16,
                 max_dim=512,
                 disc_activation_fn=tf.nn.sigmoid,
                 cls_activation_fn=tf.nn.softmax,
                 name='discriminator',
                 reuse=False):
        self.inputs = inputs
        self.input_shape = input_shape
        self.input_size = functools.reduce(operator.mul, input_shape)
        self.num_classes = num_classes
        _, downsamples = self.compute_downsamples(input_shape, min_size,
                                                  min_dim * 2, max_dim)
        with tf.variable_scope(name, reuse=reuse) as scope:
            super().__init__(scope, reuse)
            self.log_name()
            outputs = tf.reshape(inputs, (-1, ) + input_shape)
            self.features = []

            with tf.variable_scope('conv_start'):
                outputs = conv2d_with_weight_norm(
                    inputs=outputs,
                    filters=min_dim,
                    kernel_size=(1, 1),
                    strides=(1, 1),
                    padding='same',
                    activation=tf.nn.leaky_relu,
                    kernel_initializer=initializer,
                    use_bias=True,
                    bias_initializer=tf.zeros_initializer(),
                    scale=True)
                self.features.append(outputs)
                self.log_msg('WN-CONV k1n%ds1-LRelu', min_dim)

            for i, dim in enumerate(downsamples):
                with tf.variable_scope('conv{}'.format(i + 1)):
                    if i == len(downsamples) - 1:
                        stds = std_eps(outputs)
                        stds = tf.reduce_mean(stds, axis=-1, keep_dims=True)
                        stds = tf.tile(stds,
                                       tf.concat(
                                           [tf.shape(outputs)[:1], [1, 1, 1]],
                                           axis=0))
                        outputs = tf.concat([outputs, stds], axis=-1)

                    outputs = conv2d_with_weight_norm(
                        inputs=outputs,
                        filters=dim,
                        kernel_size=(3, 3),
                        strides=(2, 2),
                        padding='same',
                        activation=tf.nn.leaky_relu,
                        kernel_initializer=initializer,
                        use_bias=True,
                        bias_initializer=tf.zeros_initializer(),
                        scale=True)
                    self.features.append(outputs)
                    self.log_msg('WN-CONV k3n%ds2-LRelu', dim)

            outputs = tf.layers.flatten(outputs)

            with tf.variable_scope('disc_outputs'):
                self.disc_outputs = self.build_disc_outputs(
                    outputs, initializer, regularizer)
                self.disc_activations = opt_activation(self.disc_outputs,
                                                       disc_activation_fn)
                self.log_msg('WN-FC %d-LRelu (disc_outputs)', 1)

            if self.num_classes:
                with tf.variable_scope('cls_outputs'):
                    self.cls_outputs = self.build_cls_outputs(
                        outputs, self.num_classes, initializer, regularizer)
                    self.cls_activations = opt_activation(
                        self.cls_outputs, cls_activation_fn)
                    self.log_msg('WN-FC %d-LRelu (cls_outputs)', num_classes)
