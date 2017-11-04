"""
Generative Adversarial Networks
"""

import datetime
import functools
import operator
import os

import numpy as np
import tensorflow as tf

from .base import GANModel
from ..ops import leaky_relu
from ..train import IncrementalAverage


class BasicGenerator(object):
    """BasicGenerator"""

    def __init__(self,
                 z,
                 is_training,
                 output_shape,
                 updates_collections=tf.GraphKeys.UPDATE_OPS,
                 initializer=tf.contrib.layers.xavier_initializer(
                     uniform=False),
                 regularizer=None,
                 name='generator',
                 reuse=False,
                 dim=128,
                 num_layers=3,
                 skip_first_batch=False,
                 use_fused_batch_norm=True,
                 activation_fn=None):
        assert num_layers > 0
        self.output_shape = output_shape
        self.output_size = functools.reduce(operator.mul, output_shape)
        normalizer_fn = tf.contrib.layers.batch_norm
        normalizer_params = {
            'is_training': is_training,
            'updates_collections': updates_collections
        }
        if use_fused_batch_norm:
            normalizer_params['fused'] = True

        with tf.variable_scope(name, reuse=reuse):
            outputs = z
            for i in range(num_layers - 1):
                with tf.variable_scope('fc{}'.format(i + 1)):
                    if skip_first_batch and i == 0:
                        layer_normalizer_fn = layer_normalizer_params = None
                    else:
                        layer_normalizer_fn = normalizer_fn
                        layer_normalizer_params = normalizer_params

                    outputs = tf.contrib.layers.fully_connected(
                        inputs=outputs,
                        num_outputs=dim,
                        activation_fn=tf.nn.relu,
                        normalizer_fn=layer_normalizer_fn,
                        normalizer_params=layer_normalizer_params,
                        weights_initializer=initializer,
                        weights_regularizer=regularizer,
                        biases_initializer=tf.zeros_initializer())

            with tf.variable_scope('fc{}'.format(num_layers)):
                self.outputs = tf.contrib.layers.fully_connected(
                    inputs=outputs,
                    num_outputs=self.output_size,
                    activation_fn=activation_fn,
                    weights_initializer=initializer,
                    weights_regularizer=regularizer,
                    biases_initializer=tf.zeros_initializer())


class BasicDiscriminator(object):
    """BasicDiscriminator"""

    def __init__(self,
                 X,
                 is_training,
                 num_classes=None,
                 input_shape=None,
                 updates_collections=tf.GraphKeys.UPDATE_OPS,
                 initializer=tf.contrib.layers.xavier_initializer(
                     uniform=False),
                 regularizer=None,
                 name='discriminator',
                 reuse=False,
                 dim=128,
                 num_layers=3,
                 use_fused_batch_norm=True,
                 skip_last_biases=False,
                 use_layer_norm=False,
                 activation_fn=tf.nn.sigmoid,
                 class_activation_fn=tf.nn.softmax):
        assert num_layers > 0
        self.input_shape = input_shape
        self.input_size = functools.reduce(operator.mul, input_shape)
        if use_layer_norm:
            normalizer_fn = tf.contrib.layers.layer_norm
            normalizer_params = {'scale': False}
        else:
            normalizer_fn = tf.contrib.layers.batch_norm
            normalizer_params = {
                'is_training': is_training,
                'updates_collections': updates_collections
            }
            if use_fused_batch_norm:
                normalizer_params['fused'] = True

        with tf.variable_scope(name, reuse=reuse):
            outputs = X
            self.features = []
            for i in range(num_layers - 1):
                with tf.variable_scope('fc{}'.format(i + 1)):
                    if i == 0:
                        layer_normalizer_fn = layer_normalizer_params = None
                    else:
                        layer_normalizer_fn = normalizer_fn
                        layer_normalizer_params = normalizer_params
                    outputs = tf.contrib.layers.fully_connected(
                        inputs=outputs,
                        num_outputs=dim,
                        activation_fn=leaky_relu,
                        normalizer_fn=layer_normalizer_fn,
                        normalizer_params=layer_normalizer_params,
                        weights_initializer=initializer,
                        weights_regularizer=regularizer,
                        biases_initializer=tf.zeros_initializer())
                    self.features.append(outputs)

            with tf.variable_scope('outputs_d'):
                biases_initializer = tf.zeros_initializer(
                ) if not skip_last_biases else None
                self.outputs_d = tf.contrib.layers.fully_connected(
                    inputs=outputs,
                    num_outputs=1,
                    activation_fn=None,
                    weights_initializer=initializer,
                    weights_regularizer=regularizer,
                    biases_initializer=biases_initializer)
                self.activations_d = activation_fn(
                    self.outputs_d) if activation_fn else self.outputs_d

            if num_classes is not None:
                with tf.variable_scope('outputs_c'):
                    self.outputs_c = tf.contrib.layers.fully_connected(
                        inputs=outputs,
                        num_outputs=num_classes,
                        activation_fn=None,
                        weights_initializer=initializer,
                        weights_regularizer=regularizer,
                        biases_initializer=tf.zeros_initializer())
                    self.activations_c = class_activation_fn(
                        self.outputs_c
                    ) if class_activation_fn else self.outputs_c


class ConvTransposeGenerator(object):
    def __init__(self,
                 z,
                 is_training,
                 output_shape,
                 updates_collections=tf.GraphKeys.UPDATE_OPS,
                 initializer=tf.contrib.layers.xavier_initializer(
                     uniform=False),
                 regularizer=None,
                 name='generator',
                 reuse=False,
                 min_size=4,
                 dim=32,
                 max_dim=64,
                 num_layers=3,
                 skip_first_batch=False,
                 use_fused_batch_norm=True,
                 activation_fn=tf.nn.tanh):
        assert num_layers > 1
        self.output_shape = output_shape
        self.output_size = functools.reduce(operator.mul, output_shape)
        target_h, target_w, target_c = output_shape
        normalizer_fn = tf.contrib.layers.batch_norm
        normalizer_params = {
            'is_training': is_training,
            'updates_collections': updates_collections
        }
        if use_fused_batch_norm:
            normalizer_params['fused'] = True

        min_h, min_w = target_h, target_w
        for _ in range(num_layers - 1):
            if min_h % 2 == 0 and min_h / 2 >= min_size:
                min_h //= 2
            if min_w % 2 == 0 and min_w / 2 >= min_size:
                min_w //= 2

        with tf.variable_scope(name, reuse=reuse):
            outputs = z
            with tf.variable_scope('fc'):
                if skip_first_batch:
                    layer_normalizer_fn = layer_normalizer_params = None
                else:
                    layer_normalizer_fn = normalizer_fn
                    layer_normalizer_params = normalizer_params
                outputs = tf.contrib.layers.fully_connected(
                    inputs=outputs,
                    num_outputs=dim * min_h * min_w,
                    activation_fn=tf.nn.relu,
                    normalizer_fn=layer_normalizer_fn,
                    normalizer_params=layer_normalizer_params,
                    weights_initializer=initializer,
                    weights_regularizer=regularizer,
                    biases_initializer=tf.zeros_initializer())
                outputs = tf.reshape(outputs, (-1, min_h, min_w, dim))
            for i in range(num_layers - 1):
                with tf.variable_scope('convt{}'.format(i + 1)):
                    stride = [2, 2]
                    if min_h == target_h:
                        stride[0] = 1
                    if min_w == target_w:
                        stride[1] = 1
                    min_h = min(target_h, min_h * 2)
                    min_w = min(target_w, min_w * 2)

                    if i == num_layers - 2:
                        layer_normalizer_fn = layer_normalizer_params = None
                        dim = target_c
                        layer_activation_fn = activation_fn
                    else:
                        layer_normalizer_fn = normalizer_fn
                        layer_normalizer_params = normalizer_params
                        layer_activation_fn = tf.nn.relu
                        dim = min(2 * dim, max_dim)

                    outputs = tf.contrib.layers.conv2d_transpose(
                        inputs=outputs,
                        num_outputs=dim,
                        kernel_size=(4, 4),
                        stride=stride,
                        padding='SAME',
                        activation_fn=layer_activation_fn,
                        normalizer_fn=layer_normalizer_fn,
                        normalizer_params=layer_normalizer_params,
                        weights_initializer=initializer,
                        weights_regularizer=regularizer)

            self.outputs = tf.contrib.layers.flatten(outputs)


class ResizeConvGenerator(object):
    def __init__(self,
                 z,
                 is_training,
                 output_shape,
                 updates_collections=tf.GraphKeys.UPDATE_OPS,
                 initializer=tf.contrib.layers.xavier_initializer(
                     uniform=False),
                 regularizer=None,
                 name='generator',
                 reuse=False,
                 min_size=4,
                 dim=32,
                 max_dim=64,
                 num_layers=3,
                 skip_first_batch=False,
                 use_fused_batch_norm=True,
                 activation_fn=tf.nn.tanh):
        assert num_layers > 1
        self.output_shape = output_shape
        self.output_size = functools.reduce(operator.mul, output_shape)
        target_h, target_w, target_c = output_shape
        normalizer_fn = tf.contrib.layers.batch_norm
        normalizer_params = {
            'is_training': is_training,
            'updates_collections': updates_collections
        }
        if use_fused_batch_norm:
            normalizer_params['fused'] = True

        min_h, min_w = target_h, target_w
        for _ in range(num_layers - 1):
            if min_h % 2 == 0 and min_h / 2 >= min_size:
                min_h //= 2
            if min_w % 2 == 0 and min_w / 2 >= min_size:
                min_w //= 2

        with tf.variable_scope(name, reuse=reuse):
            outputs = z
            with tf.variable_scope('fc'):
                if skip_first_batch:
                    layer_normalizer_fn = layer_normalizer_params = None
                else:
                    layer_normalizer_fn = normalizer_fn
                    layer_normalizer_params = normalizer_params
                outputs = tf.contrib.layers.fully_connected(
                    inputs=outputs,
                    num_outputs=dim * min_h * min_w,
                    activation_fn=tf.nn.relu,
                    normalizer_fn=layer_normalizer_fn,
                    normalizer_params=layer_normalizer_params,
                    weights_initializer=initializer,
                    weights_regularizer=regularizer,
                    biases_initializer=tf.zeros_initializer())
                outputs = tf.reshape(outputs, (-1, min_h, min_w, dim))
            for i in range(num_layers - 1):
                with tf.variable_scope('rs_conv{}'.format(i + 1)):
                    min_h = min(target_h, min_h * 2)
                    min_w = min(target_w, min_w * 2)

                    if i == num_layers - 2:
                        layer_normalizer_fn = layer_normalizer_params = None
                        dim = target_c
                        layer_activation_fn = activation_fn
                    else:
                        layer_normalizer_fn = normalizer_fn
                        layer_normalizer_params = normalizer_params
                        layer_activation_fn = tf.nn.relu
                        dim = min(2 * dim, max_dim)

                    outputs = tf.image.resize_nearest_neighbor(
                        outputs, (min_h, min_w), name='resize')
                    outputs = tf.contrib.layers.conv2d(
                        inputs=outputs,
                        num_outputs=dim,
                        kernel_size=(3, 3),
                        stride=(1, 1),
                        padding='SAME',
                        activation_fn=layer_activation_fn,
                        normalizer_fn=layer_normalizer_fn,
                        normalizer_params=layer_normalizer_params,
                        weights_initializer=initializer,
                        weights_regularizer=regularizer)

            self.outputs = tf.contrib.layers.flatten(outputs)


class ConvDiscriminator(object):
    def __init__(self,
                 X,
                 is_training,
                 input_shape,
                 num_classes=None,
                 updates_collections=tf.GraphKeys.UPDATE_OPS,
                 initializer=tf.contrib.layers.xavier_initializer(
                     uniform=False),
                 regularizer=None,
                 name='discriminator',
                 reuse=False,
                 dim=32,
                 max_dim=64,
                 num_layers=3,
                 use_fused_batch_norm=True,
                 skip_last_biases=False,
                 use_layer_norm=False,
                 activation_fn=tf.nn.sigmoid,
                 class_activation_fn=tf.nn.softmax):
        assert num_layers > 0
        self.input_shape = input_shape
        self.input_size = functools.reduce(operator.mul, input_shape)
        if use_layer_norm:
            normalizer_fn = tf.contrib.layers.layer_norm
            normalizer_params = {'scale': False}
        else:
            normalizer_fn = tf.contrib.layers.batch_norm
            normalizer_params = {
                'is_training': is_training,
                'updates_collections': updates_collections
            }
            if use_fused_batch_norm:
                normalizer_params['fused'] = True

        with tf.variable_scope(name, reuse=reuse):
            outputs = tf.reshape(X, (-1, ) + input_shape)
            self.features = []
            for i in range(num_layers - 1):
                with tf.variable_scope('conv{}'.format(i + 1)):
                    if i == 0:
                        layer_normalizer_fn = layer_normalizer_params = None
                    else:
                        layer_normalizer_fn = normalizer_fn
                        layer_normalizer_params = normalizer_params
                    outputs = tf.contrib.layers.conv2d(
                        inputs=outputs,
                        num_outputs=dim,
                        kernel_size=(4, 4),
                        stride=(2, 2),
                        padding='SAME',
                        activation_fn=leaky_relleaky_relu,
                        normalizer_fn=layer_normalizer_fn,
                        normalizer_params=layer_normalizer_params,
                        weights_initializer=initializer,
                        weights_regularizer=regularizer)
                    self.features.append(outputs)
                    dim = min(2 * dim, max_dim)

            outputs = tf.contrib.layers.flatten(outputs)

            with tf.variable_scope('outputs_d'):
                biases_initializer = tf.zeros_initializer(
                ) if not skip_last_biases else None
                self.outputs_d = tf.contrib.layers.fully_connected(
                    inputs=outputs,
                    num_outputs=1,
                    activation_fn=None,
                    weights_initializer=initializer,
                    biases_initializer=biases_initializer)
                self.activations_d = activation_fn(
                    self.outputs_d) if activation_fn else self.outputs_d

            if num_classes is not None:
                with tf.variable_scope('outputs_c'):
                    self.outputs_c = tf.contrib.layers.fully_connected(
                        inputs=outputs,
                        num_outputs=num_classes,
                        activation_fn=None,
                        weights_initializer=initializer,
                        weights_regularizer=regularizer,
                        biases_initializer=tf.zeros_initializer())
                    self.activations_c = class_activation_fn(
                        self.outputs_c
                    ) if class_activation_fn else self.outputs_c


class GAN(GANModel):
    def __init__(self,
                 sess,
                 X_real,
                 num_examples,
                 output_shape,
                 reg_const=5e-5,
                 stddev=None,
                 z_dim=10,
                 g_dim=32,
                 d_dim=32,
                 z_stddev=1.,
                 batch_size=128,
                 g_learning_rate=0.0002,
                 g_beta1=0.5,
                 d_learning_rate=0.0002,
                 d_beta1=0.5,
                 d_label_smooth=0.25,
                 generator_cls=BasicGenerator,
                 discriminator_cls=BasicDiscriminator,
                 image_summary=False,
                 name='GAN'):
        with tf.variable_scope(name):
            super().__init__(
                sess=sess,
                name=name,
                num_examples=num_examples,
                output_shape=output_shape,
                reg_const=reg_const,
                stddev=stddev,
                batch_size=batch_size,
                image_summary=image_summary)

            self.z_stddev = z_stddev
            self.z_dim = z_dim

            self.g_dim = g_dim
            self.g_learning_rate = g_learning_rate
            self.g_beta1 = g_beta1

            self.d_dim = d_dim
            self.d_learning_rate = d_learning_rate
            self.d_beta1 = d_beta1
            self.d_label_smooth = d_label_smooth

            self.X = X_real
            self.z = tf.random_normal(
                (batch_size, z_dim),
                mean=0.0,
                stddev=z_stddev,
                name='z',
                dtype=tf.float32)
            self.z = tf.placeholder_with_default(self.z, [None, z_dim])

            self._build_GAN(generator_cls, discriminator_cls)
            self._build_losses()
            self._build_optimizer()
            self._build_summary()

            self.saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())

    def _build_GAN(self, generator_cls, discriminator_cls):
        self.g = generator_cls(
            z=self.z,
            is_training=self.is_training,
            output_shape=self.output_shape,
            regularizer=self.regularizer,
            initializer=self.initializer,
            dim=self.g_dim,
            name='generator')

        self.d_real = discriminator_cls(
            X=self.X,
            is_training=self.is_training,
            input_shape=self.output_shape,
            regularizer=self.regularizer,
            initializer=self.initializer,
            dim=self.d_dim,
            name='discriminator')
        self.d_fake = discriminator_cls(
            X=self.g.outputs,
            is_training=self.is_training,
            input_shape=self.output_shape,
            regularizer=self.regularizer,
            initializer=self.initializer,
            dim=self.d_dim,
            reuse=True,
            name='discriminator')

        with tf.variable_scope('generator') as scope:
            self.g_vars = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)
        with tf.variable_scope('discriminator') as scope:
            self.d_vars = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)

    def _build_losses(self):
        with tf.variable_scope('generator') as scope:
            self.g_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.d_fake.outputs_d,
                    labels=tf.ones_like(self.d_fake.outputs_d)))

            g_reg_ops = tf.get_collection(
                tf.GraphKeys.REGULARIZATION_LOSSES, scope=scope.name)
            self.g_reg_loss = tf.add_n(g_reg_ops) if g_reg_ops else 0.0

            self.g_total_loss = self.g_loss + self.g_reg_loss

        with tf.variable_scope('discriminator') as scope:
            if self.d_label_smooth > 0.0:
                labels_real = tf.ones_like(
                    self.d_real.outputs_d) - self.d_label_smooth
            else:
                labels_real = tf.ones_like(self.d_real.outputs_d)

            self.d_loss_real = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.d_real.outputs_d, labels=labels_real))
            self.d_loss_fake = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.d_fake.outputs_d,
                    labels=tf.zeros_like(self.d_fake.outputs_d)))
            self.d_loss = self.d_loss_real + self.d_loss_fake

            d_reg_ops = tf.get_collection(
                tf.GraphKeys.REGULARIZATION_LOSSES, scope=scope.name)
            self.d_reg_loss = tf.add_n(d_reg_ops) if d_reg_ops else 0.0

            self.d_total_loss = self.d_loss + self.d_reg_loss

    def _build_summary(self):
        with tf.variable_scope('summary') as scope:
            self.z_sum = tf.summary.histogram('z', self.z)
            if self.image_summary:
                self.g_sum = tf.summary.image('g',
                                              tf.reshape(
                                                  self.g.outputs,
                                                  (-1, ) + self.output_shape))
            else:
                self.g_sum = tf.summary.histogram('g', self.g.outputs)
            self.d_real_sum = tf.summary.histogram('d_real',
                                                   self.d_real.activations_d)
            self.d_fake_sum = tf.summary.histogram('d_fake',
                                                   self.d_fake.activations_d)

            self.g_loss_sum = tf.summary.scalar('g_loss', self.g_loss)
            self.g_reg_loss_sum = tf.summary.scalar('g_reg_loss',
                                                    self.g_reg_loss)
            self.g_total_loss_sum = tf.summary.scalar('g_total_loss',
                                                      self.g_total_loss)
            self.d_loss_sum = tf.summary.scalar('d_loss', self.d_loss)
            self.d_loss_real_sum = tf.summary.scalar('d_loss_real',
                                                     self.d_loss_real)
            self.d_loss_fake_sum = tf.summary.scalar('d_loss_fake',
                                                     self.d_loss_fake)
            self.d_reg_loss_sum = tf.summary.scalar('d_reg_loss',
                                                    self.d_reg_loss)
            self.d_total_loss_sum = tf.summary.scalar('d_total_loss',
                                                      self.d_total_loss)

            self.summary = tf.summary.merge(
                tf.get_collection(tf.GraphKeys.SUMMARIES, scope=scope.name))

    def _build_optimizer(self):
        with tf.variable_scope('generator') as scope:
            update_ops_g = tf.get_collection(
                tf.GraphKeys.UPDATE_OPS, scope=scope.name)
            with tf.control_dependencies(update_ops_g):
                self.g_optim = tf.train.AdamOptimizer(
                    self.g_learning_rate, beta1=self.g_beta1).minimize(
                        self.g_total_loss, var_list=self.g_vars)

        with tf.variable_scope('discriminator') as scope:
            update_ops_d = tf.get_collection(
                tf.GraphKeys.UPDATE_OPS, scope=scope.name)
            with tf.control_dependencies(update_ops_d + update_ops_g):
                self.d_optim = tf.train.AdamOptimizer(
                    self.d_learning_rate, beta1=self.d_beta1).minimize(
                        self.d_total_loss, var_list=self.d_vars)

    def train(self,
              num_epochs,
              resume=True,
              resume_step=None,
              checkpoint_dir=None,
              save_step=500,
              sample_step=100,
              sample_fn=None,
              log_dir='logs'):
        with tf.variable_scope(self.name):
            if log_dir is not None:
                log_dir = os.path.join(log_dir, self.name)
                os.makedirs(log_dir, exist_ok=True)
                run_name = '{}_{}'.format(self.name,
                                          datetime.datetime.now().isoformat())
                log_path = os.path.join(log_dir, run_name)
                self.writer = tf.summary.FileWriter(log_path, self.sess.graph)
            else:
                self.writer = None

            num_batches = self.num_examples // self.batch_size

            success, step = False, 0
            if resume and checkpoint_dir:
                success, saved_step = self.load(checkpoint_dir, resume_step)

            if success:
                step = saved_step
                start_epoch = step // num_batches
            else:
                start_epoch = 0

            for epoch in range(start_epoch, num_epochs):
                start_idx = step % num_batches
                epoch_g_total_loss = IncrementalAverage()
                epoch_d_total_loss = IncrementalAverage()
                t = self._trange(
                    start_idx, num_batches, desc='Epoch #{}'.format(epoch + 1))
                for idx in t:
                    (_, _, d_total_loss, g_total_loss,
                     summary_str) = self.sess.run([
                         self.d_optim, self.g_optim, self.d_total_loss,
                         self.g_total_loss, self.summary
                     ])
                    epoch_d_total_loss.add(d_total_loss)
                    epoch_g_total_loss.add(g_total_loss)

                    if self.writer:
                        self.writer.add_summary(summary_str, step)
                    step += 1

                    # Save checkpoint
                    if checkpoint_dir and save_step and step % save_step == 0:
                        self.save(checkpoint_dir, step)

                    # Sample
                    if sample_fn and sample_step and (
                        (isinstance(sample_step, int) and
                         step % sample_step == 0) or
                        (not isinstance(sample_step, int) and
                         step in sample_step)):
                        sample_fn(self, step)

                    t.set_postfix(
                        g_loss=epoch_g_total_loss.average,
                        d_loss=epoch_d_total_loss.average)

            # Save final checkpoint
            if checkpoint_dir:
                self.save(checkpoint_dir, step)

    def sample(self, num_samples=None, z=None):
        if z is not None:
            return self.sess.run(
                self.g.outputs, feed_dict={self.is_training: False,
                                           self.z: z})
        elif num_samples is not None:
            return self.sess.run(
                self.g.outputs,
                feed_dict={
                    self.is_training: False,
                    self.z: self.sample_z(num_samples)
                })
        else:
            return self.sess.run(
                self.g.outputs, feed_dict={self.is_training: False})

    def sample_z(self, num_samples):
        return np.random.normal(0.0, self.z_stddev, (num_samples, self.z_dim))
