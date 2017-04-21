"""
Generative Adversarial Networks
"""

import functools
import math
import operator
import os
import time

import numpy as np
import tensorflow as tf

from .base import Model
from ..ops import lrelu
from ..train import IncrementalAverage


def build_basic_generator(z,
                          is_training,
                          updates_collections,
                          output_shape,
                          name='generator',
                          reuse=False,
                          dim=128,
                          num_layers=3,
                          skip_first_batch=False,
                          activation_fn=None):
    assert num_layers > 0
    output_size = functools.reduce(operator.mul, output_shape)
    initializer = tf.contrib.layers.xavier_initializer()

    with tf.variable_scope(name, reuse=reuse):
        outputs = z
        for i in range(num_layers - 1):
            with tf.variable_scope('fc{}'.format(i + 1)):
                if skip_first_batch and i == 0:
                    normalizer_fn = normalizer_params = None
                else:
                    normalizer_fn = tf.contrib.layers.batch_norm
                    normalizer_params = {
                        'is_training': is_training,
                        'updates_collections': updates_collections
                    }
                outputs = tf.contrib.layers.fully_connected(
                    inputs=outputs,
                    num_outputs=dim,
                    activation_fn=tf.nn.relu,
                    normalizer_fn=normalizer_fn,
                    normalizer_params=normalizer_params,
                    weights_initializer=initializer,
                    biases_initializer=tf.zeros_initializer())

        with tf.variable_scope('fc{}'.format(num_layers)):
            outputs = tf.contrib.layers.fully_connected(
                inputs=outputs,
                num_outputs=output_size,
                activation_fn=activation_fn,
                weights_initializer=initializer,
                biases_initializer=tf.zeros_initializer())

        return outputs


def build_basic_discriminator(X,
                              is_training,
                              updates_collections,
                              num_classes=None,
                              input_shape=None,
                              return_features=False,
                              name='discriminator',
                              reuse=False,
                              dim=128,
                              num_layers=3,
                              activation_fn=tf.nn.sigmoid,
                              class_activation_fn=tf.nn.softmax):
    assert num_layers > 0
    initializer = tf.contrib.layers.xavier_initializer()

    with tf.variable_scope(name, reuse=reuse):
        outputs = X
        features = []
        for i in range(num_layers - 1):
            with tf.variable_scope('fc{}'.format(i + 1)):
                if i == 0:
                    normalizer_fn = normalizer_params = None
                else:
                    normalizer_fn = tf.contrib.layers.batch_norm
                    normalizer_params = {
                        'is_training': is_training,
                        'updates_collections': updates_collections
                    }
                outputs = tf.contrib.layers.fully_connected(
                    inputs=outputs,
                    num_outputs=dim,
                    activation_fn=lrelu,
                    normalizer_fn=normalizer_fn,
                    normalizer_params=normalizer_params,
                    weights_initializer=initializer,
                    biases_initializer=tf.zeros_initializer())
                features.append(outputs)

        with tf.variable_scope('fc_d'):
            fc_d = tf.contrib.layers.fully_connected(
                inputs=outputs,
                num_outputs=1,
                activation_fn=None,
                weights_initializer=initializer,
                biases_initializer=tf.zeros_initializer())
            act_d = activation_fn(fc_d) if activation_fn else fc_d

        if num_classes is None:
            if return_features:
                return act_d, fc_d, features
            else:
                return act_d, fc_d
        else:
            with tf.variable_scope('fc_c'):
                fc_c = tf.contrib.layers.fully_connected(
                    inputs=outputs,
                    num_outputs=num_classes,
                    activation_fn=None,
                    weights_initializer=initializer,
                    biases_initializer=tf.zeros_initializer())
                act_c = class_activation_fn(
                    fc_c) if class_activation_fn else fc_c
            if return_features:
                return act_d, fc_d, act_c, fc_c, features
            else:
                return act_d, fc_d, act_c, fc_c


def build_resize_conv_generator(z,
                                is_training,
                                updates_collections,
                                output_shape,
                                name='generator',
                                reuse=False,
                                min_size=4,
                                dim=128,
                                num_layers=3,
                                skip_first_batch=False,
                                activation_fn=None):
    assert num_layers > 1
    target_h, target_w, target_c = output_shape
    initializer = tf.contrib.layers.xavier_initializer()

    with tf.variable_scope(name, reuse=reuse):
        fc = z
        h = max(min_size, int(math.ceil(target_h / (2**(num_layers - 1)))))
        w = max(min_size, int(math.ceil(target_w / (2**(num_layers - 1)))))
        c = dim * (2**(num_layers - 2))

        with tf.variable_scope('fc'):
            if skip_first_batch:
                normalizer_fn = normalizer_params = None
            else:
                normalizer_fn = tf.contrib.layers.batch_norm
                normalizer_params = {
                    'is_training': is_training,
                    'updates_collections': updates_collections
                }
            fc = tf.contrib.layers.fully_connected(
                inputs=fc,
                num_outputs=h * w * c,
                activation_fn=tf.nn.relu,
                normalizer_fn=normalizer_fn,
                normalizer_params=normalizer_params,
                weights_initializer=initializer,
                biases_initializer=tf.zeros_initializer())
            fc = tf.reshape(fc, (-1, h, w, c))
        for i in range(num_layers - 1):
            if i == num_layers - 2:
                normalizer_fn = normalizer_params = None
            else:
                normalizer_fn = tf.contrib.layers.batch_norm
                normalizer_params = {
                    'is_training': is_training,
                    'updates_collections': updates_collections
                }

            h = max(min_size,
                    int(math.ceil(target_h / (2**(num_layers - 2 - i)))))
            w = max(min_size,
                    int(math.ceil(target_w / (2**(num_layers - 2 - i)))))
            c = dim * (2**(
                num_layers - 3 - i)) if i != num_layers - 2 else target_c
            fc = tf.image.resize_nearest_neighbor(
                fc, (h, w), name='g_rs_{}'.format(i + 1))
            fc = tf.layers.conv2d(
                inputs=fc,
                filters=c,
                kernel_size=(5, 5),
                strides=(1, 1),
                padding='SAME',
                activation=None,
                kernel_initializer=initializer,
                name='d_conv{}'.format(i))
            if normalizer_fn is not None:
                fc = normalizer_fn(fc, **normalizer_params)
                fc = tf.nn.relu(fc)
        return tf.nn.tanh(tf.contrib.layers.flatten(fc))


def build_conv_discriminator(X,
                             is_training,
                             updates_collections,
                             input_shape,
                             num_classes=None,
                             return_features=False,
                             name='discriminator',
                             reuse=False,
                             dim=64,
                             num_layers=4,
                             activation_fn=tf.nn.sigmoid,
                             class_activation_fn=tf.nn.softmax):
    assert num_layers > 0
    initializer = tf.contrib.layers.xavier_initializer()

    with tf.variable_scope(name, reuse=reuse):
        outputs = tf.reshape(X, (-1, ) + input_shape)
        features = []
        for i in range(num_layers - 1):
            with tf.variable_scope('d_conv{}'.format(i)):
                if i == 0:
                    normalizer_fn = normalizer_params = None
                else:
                    normalizer_fn = tf.contrib.layers.batch_norm
                    normalizer_params = {
                        'is_training': is_training,
                        'updates_collections': updates_collections
                    }
                outputs = tf.layers.conv2d(
                    inputs=outputs,
                    filters=dim,
                    kernel_size=(5, 5),
                    strides=(2, 2),
                    padding='SAME',
                    activation=None,
                    kernel_initializer=initializer)
                if normalizer_fn is not None:
                    outputs = normalizer_fn(outputs, **normalizer_params)
                outputs = lrelu(outputs)
                features.append(outputs)
                dim *= 2

        outputs = tf.contrib.layers.flatten(outputs)

        with tf.variable_scope('fc'):
            fc_d = tf.contrib.layers.fully_connected(
                inputs=outputs,
                num_outputs=1,
                activation_fn=None,
                weights_initializer=initializer,
                biases_initializer=tf.zeros_initializer())
            act_d = activation_fn(fc_d) if activation_fn else fc_d

        if num_classes is None:
            if return_features:
                return act_d, fc_d, features
            else:
                return act_d, fc_d
        else:
            with tf.variable_scope('fc_c'):
                fc_c = tf.contrib.layers.fully_connected(
                    inputs=outputs,
                    num_outputs=num_classes,
                    activation_fn=None,
                    weights_initializer=initializer,
                    biases_initializer=tf.zeros_initializer())
                act_c = class_activation_fn(
                    fc_c) if class_activation_fn else fc_c
            if return_features:
                return act_d, fc_d, act_c, fc_c, features
            else:
                return act_d, fc_d, act_c, fc_c


class GAN(Model):
    def __init__(self,
                 sess,
                 X_real,
                 num_examples,
                 output_shape,
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
                 generator_fn=build_basic_generator,
                 discriminator_fn=build_basic_discriminator,
                 image_summary=False,
                 name='GAN'):
        with tf.variable_scope(name) as scope:
            super().__init__(sess=sess, name=name)

            self.output_shape = output_shape
            self.batch_size = batch_size
            self.num_examples = num_examples

            self.z_stddev = z_stddev
            self.z_dim = z_dim

            self.g_dim = g_dim
            self.g_learning_rate = g_learning_rate
            self.g_beta1 = g_beta1

            self.d_dim = d_dim
            self.d_learning_rate = d_learning_rate
            self.d_beta1 = d_beta1
            self.d_label_smooth = d_label_smooth

            self.image_summary = image_summary

            self.X = X_real
            self.z = tf.random_normal(
                (batch_size, z_dim),
                mean=0.0,
                stddev=z_stddev,
                name='z',
                dtype=tf.float32)
            self.is_training = tf.placeholder(tf.bool, [], name='is_training')
            self.updates_collections_noop = 'updates_collections_noop'

            self._build_GAN(generator_fn, discriminator_fn)
            self._build_summary()
            self._build_optimizer(scope)

            self.saver = tf.train.Saver()
            tf.global_variables_initializer().run()

    def _build_GAN(self, generator_fn, discriminator_fn):
        self.g = generator_fn(
            self.z,
            self.is_training,
            tf.GraphKeys.UPDATE_OPS,
            self.output_shape,
            dim=self.g_dim,
            name='generator')

        self.d_real, self.d_logits_real = discriminator_fn(
            self.X,
            self.is_training,
            tf.GraphKeys.UPDATE_OPS,
            input_shape=self.output_shape,
            dim=self.d_dim,
            name='discriminator')
        self.d_fake, self.d_logits_fake = discriminator_fn(
            self.g,
            self.is_training,
            tf.GraphKeys.UPDATE_OPS,
            input_shape=self.output_shape,
            dim=self.d_dim,
            reuse=True,
            name='discriminator')

        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.d_logits_fake,
                labels=tf.ones_like(self.d_logits_fake)))

        if self.d_label_smooth > 0.0:
            labels_real = tf.ones_like(self.d_real) - self.d_label_smooth
        else:
            labels_real = tf.ones_like(self.d_real)

        self.d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.d_logits_real, labels=labels_real))
        self.d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.d_logits_fake, labels=tf.zeros_like(self.d_fake)))
        self.d_loss = self.d_loss_real + self.d_loss_fake

        with tf.variable_scope('generator') as scope:
            self.g_vars = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)
        with tf.variable_scope('discriminator') as scope:
            self.d_vars = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)

        self.z_sampler = tf.placeholder(
            tf.float32, [None, self.z_dim], name='z_sampler')
        self.sampler = generator_fn(
            self.z_sampler,
            self.is_training,
            self.updates_collections_noop,
            self.output_shape,
            dim=self.g_dim,
            name='generator',
            reuse=True)

    def _build_summary(self):
        self.z_sum = tf.summary.histogram('z', self.z)
        if self.image_summary:
            self.g_sum = tf.summary.image(
                'g', tf.reshape(self.g, (-1, ) + self.output_shape))
        else:
            self.g_sum = tf.summary.histogram('g', self.g)
        self.d_real_sum = tf.summary.histogram('d_real', self.d_real)
        self.d_fake_sum = tf.summary.histogram('d_fake', self.d_fake)

        self.g_loss_sum = tf.summary.scalar('g_loss', self.g_loss)
        self.d_loss_sum = tf.summary.scalar('d_loss', self.d_loss)
        self.d_loss_real_sum = tf.summary.scalar('d_loss_real',
                                                 self.d_loss_real)
        self.d_loss_fake_sum = tf.summary.scalar('d_loss_fake',
                                                 self.d_loss_fake)

        self.summary = tf.summary.merge([
            self.z_sum, self.d_real_sum, self.d_fake_sum, self.d_loss_real_sum,
            self.d_loss_fake_sum, self.d_loss_sum, self.g_sum, self.g_loss_sum
        ])

    def _build_optimizer(self, scope):
        g_total_loss = self.g_loss
        d_total_loss = self.d_loss

        with tf.variable_scope('generator') as g_scope:
            update_ops_g = tf.get_collection(
                tf.GraphKeys.UPDATE_OPS, scope=g_scope.name)
        with tf.control_dependencies(update_ops_g):
            self.g_optim = tf.train.AdamOptimizer(
                self.g_learning_rate, beta1=self.g_beta1).minimize(
                    g_total_loss, var_list=self.g_vars)

        with tf.variable_scope('discriminator') as d_scope:
            update_ops_d = tf.get_collection(
                tf.GraphKeys.UPDATE_OPS, scope=d_scope.name)
        with tf.control_dependencies(update_ops_d + update_ops_g):
            self.d_optim = tf.train.AdamOptimizer(
                self.d_learning_rate, beta1=self.d_beta1).minimize(
                    d_total_loss, var_list=self.d_vars)

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
                run_name = '{}_{}'.format(self.name, time.time())
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
                epoch_g_loss = IncrementalAverage()
                epoch_d_loss_fake = IncrementalAverage()
                epoch_d_loss_real = IncrementalAverage()
                t = self._trange(
                    start_idx, num_batches, desc='Epoch #{}'.format(epoch + 1))
                for idx in t:
                    (_, _, d_loss_fake, d_loss_real, g_loss,
                     summary_str) = self.sess.run(
                         [
                             self.d_optim, self.g_optim, self.d_loss_fake,
                             self.d_loss_real, self.g_loss, self.summary
                         ],
                         feed_dict={self.is_training: True})
                    epoch_d_loss_fake.add(d_loss_fake)
                    epoch_d_loss_real.add(d_loss_real)
                    epoch_g_loss.add(g_loss)

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
                        g_loss=epoch_g_loss.average,
                        d_loss_real=epoch_d_loss_real.average,
                        d_loss_fake=epoch_d_loss_fake.average)

            # Save final checkpoint
            if checkpoint_dir:
                self.save(checkpoint_dir, step)

    def sample(self, num_samples=None, z=None):
        if z is not None:
            return self.sess.run(
                self.sampler,
                feed_dict={self.is_training: False,
                           self.z_sampler: z})
        elif num_samples is not None:
            return self.sess.run(
                self.sampler,
                feed_dict={
                    self.is_training: False,
                    self.z_sampler: self.sample_z(num_samples)
                })
        else:
            return self.sess.run(self.g, feed_dict={self.is_training: False})

    def sample_z(self, num_samples):
        return np.random.normal(0.0, self.z_stddev, (num_samples, self.z_dim))
