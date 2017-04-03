import functools
import operator
import os
import time

import numpy as np
import tensorflow as tf

from .base import Model
from .gan import build_basic_discriminator, build_basic_generator


class DiscoGAN(Model):
    def __init__(self,
                 sess,
                 X_real,
                 Y_real,
                 num_examples,
                 x_output_shape,
                 y_output_shape,
                 g_dim=32,
                 d_dim=32,
                 batch_size=128,
                 g_learning_rate=0.0002,
                 g_beta1=0.5,
                 d_learning_rate=0.0002,
                 d_beta1=0.5,
                 d_label_smooth=0.25,
                 generator_fn=build_basic_generator,
                 discriminator_fn=build_basic_discriminator,
                 image_summary=False,
                 name='DiscoGAN'):
        with tf.variable_scope(name) as scope:
            super().__init__(sess=sess, name=name)

            self.x_output_shape = x_output_shape
            self.y_output_shape = y_output_shape
            self.batch_size = batch_size
            self.num_examples = num_examples

            self.g_dim = g_dim
            self.g_learning_rate = g_learning_rate
            self.g_beta1 = g_beta1

            self.d_dim = d_dim
            self.d_learning_rate = d_learning_rate
            self.d_beta1 = d_beta1
            self.d_label_smooth = d_label_smooth

            self.image_summary = image_summary

            self.X = X_real
            self.Y = Y_real
            self.is_training = tf.placeholder(tf.bool, [], name='is_training')
            self.updates_collections_noop = 'updates_collections_noop'
            self.updates_collections_d = 'updates_collections_d'
            self.updates_collections_g = 'updates_collections_g'

            self._build_GAN(generator_fn, discriminator_fn)
            self._build_summary()
            self._build_optimizer(scope)

            self.saver = tf.train.Saver()
            tf.global_variables_initializer().run()

    def _build_GAN(self, generator_fn, discriminator_fn):
        self.x_g = generator_fn(
            self.Y,
            self.is_training,
            self.updates_collections_g,
            self.x_output_shape,
            dim=self.g_dim,
            skip_first_batch=True,
            name='x_generator')
        self.y_g = generator_fn(
            self.X,
            self.is_training,
            self.updates_collections_g,
            self.y_output_shape,
            dim=self.g_dim,
            skip_first_batch=True,
            name='y_generator')

        self.x_g_recon = generator_fn(
            self.y_g,
            self.is_training,
            self.updates_collections_noop,
            self.x_output_shape,
            dim=self.g_dim,
            reuse=True,
            skip_first_batch=True,
            name='x_generator')
        self.y_g_recon = generator_fn(
            self.x_g,
            self.is_training,
            self.updates_collections_noop,
            self.y_output_shape,
            dim=self.g_dim,
            reuse=True,
            skip_first_batch=True,
            name='y_generator')

        self.x_d_real, self.x_d_logits_real = discriminator_fn(
            self.X,
            self.is_training,
            self.updates_collections_d,
            input_shape=self.x_output_shape,
            dim=self.d_dim,
            name='x_discriminator')
        self.y_d_real, self.y_d_logits_real = discriminator_fn(
            self.Y,
            self.is_training,
            self.updates_collections_d,
            input_shape=self.y_output_shape,
            dim=self.d_dim,
            name='y_discriminator')

        self.x_d_fake, self.x_d_logits_fake = discriminator_fn(
            self.x_g,
            self.is_training,
            self.updates_collections_d,
            input_shape=self.x_output_shape,
            dim=self.d_dim,
            reuse=True,
            name='x_discriminator')
        self.y_d_fake, self.y_d_logits_fake = discriminator_fn(
            self.y_g,
            self.is_training,
            self.updates_collections_d,
            input_shape=self.y_output_shape,
            dim=self.d_dim,
            reuse=True,
            name='y_discriminator')

        self.x_recon_loss = tf.reduce_sum(
            tf.losses.mean_squared_error(self.X, self.x_g_recon))
        self.y_recon_loss = tf.reduce_sum(
            tf.losses.mean_squared_error(self.Y, self.y_g_recon))

        self.x_g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.x_d_logits_fake,
                labels=tf.ones_like(self.x_d_logits_fake)))
        self.y_g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.y_d_logits_fake,
                labels=tf.ones_like(self.y_d_logits_fake)))
        self.g_loss = self.x_g_loss + self.y_g_loss + self.x_recon_loss + self.y_recon_loss

        if self.d_label_smooth > 0.0:
            labels_real = tf.ones_like(self.x_d_real) - self.d_label_smooth
        else:
            labels_real = tf.ones_like(self.x_d_real)

        self.x_d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.x_d_logits_real, labels=labels_real))
        self.y_d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.y_d_logits_real, labels=labels_real))
        self.x_d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.x_d_logits_fake,
                labels=tf.zeros_like(self.x_d_fake)))
        self.y_d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.y_d_logits_fake,
                labels=tf.zeros_like(self.y_d_fake)))
        self.d_loss = self.x_d_loss_real + self.x_d_loss_fake + self.y_d_loss_real + self.y_d_loss_fake

        with tf.variable_scope('x_generator') as scope:
            self.x_g_vars = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)
        with tf.variable_scope('y_generator') as scope:
            self.y_g_vars = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)
        with tf.variable_scope('x_discriminator') as scope:
            self.x_d_vars = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)
        with tf.variable_scope('y_discriminator') as scope:
            self.y_d_vars = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)

        x_output_size = functools.reduce(operator.mul, self.x_output_shape)
        y_output_size = functools.reduce(operator.mul, self.y_output_shape)
        self.x_sampler = tf.placeholder(
            tf.float32, [None, x_output_size], name='x_sampler')
        self.y_sampler = tf.placeholder(
            tf.float32, [None, y_output_size], name='y_sampler')
        self.sampler_for_x = generator_fn(
            self.y_sampler,
            self.is_training,
            self.updates_collections_noop,
            self.x_output_shape,
            dim=self.g_dim,
            skip_first_batch=True,
            name='x_generator',
            reuse=True)
        self.sampler_for_y = generator_fn(
            self.x_sampler,
            self.is_training,
            self.updates_collections_noop,
            self.y_output_shape,
            dim=self.g_dim,
            skip_first_batch=True,
            name='y_generator',
            reuse=True)
        self.sampler_recon_for_x = generator_fn(
            self.sampler_for_y,
            self.is_training,
            self.updates_collections_noop,
            self.x_output_shape,
            dim=self.g_dim,
            skip_first_batch=True,
            name='x_generator',
            reuse=True)
        self.sampler_recon_for_y = generator_fn(
            self.sampler_for_x,
            self.is_training,
            self.updates_collections_noop,
            self.y_output_shape,
            dim=self.g_dim,
            skip_first_batch=True,
            name='y_generator',
            reuse=True)

    def _build_summary(self):
        if self.image_summary:
            self.x_sum = tf.summary.image(
                'x', tf.reshape(self.X, (-1, ) + self.x_ouput_shape))
            self.y_sum = tf.summary.image(
                'y', tf.reshape(self.Y, (-1, ) + self.y_ouput_shape))
            self.x_g_sum = tf.summary.image(
                'x_g', tf.reshape(self.x_g, (-1, ) + self.x_ouput_shape))
            self.y_g_sum = tf.summary.image(
                'y_g', tf.reshape(self.y_g, (-1, ) + self.y_ouput_shape))
            self.x_g_recon_sum = tf.summary.image(
                'x_g_recon',
                tf.reshape(self.x_g_recon, (-1, ) + self.x_ouput_shape))
            self.y_g_recon_sum = tf.summary.image(
                'y_g_recon',
                tf.reshape(self.y_g_recon, (-1, ) + self.y_ouput_shape))
        else:
            self.x_sum = tf.summary.histogram('x', self.X)
            self.y_sum = tf.summary.histogram('y', self.Y)
            self.x_g_sum = tf.summary.histogram('x_g', self.x_g)
            self.y_g_sum = tf.summary.histogram('y_g', self.y_g)
            self.x_g_recon_sum = tf.summary.histogram('x_g_recon',
                                                      self.x_g_recon)
            self.y_g_recon_sum = tf.summary.histogram('y_g_recon',
                                                      self.y_g_recon)
        self.x_d_real_sum = tf.summary.histogram('x_d_real', self.x_d_real)
        self.y_d_real_sum = tf.summary.histogram('y_d_real', self.y_d_real)
        self.x_d_fake_sum = tf.summary.histogram('x_d_fake', self.x_d_fake)
        self.y_d_fake_sum = tf.summary.histogram('y_d_fake', self.y_d_fake)

        self.g_loss_sum = tf.summary.scalar('g_loss', self.g_loss)
        self.d_loss_sum = tf.summary.scalar('d_loss', self.d_loss)

        self.summary = tf.summary.merge([
            self.x_sum, self.y_sum, self.x_g_sum, self.y_g_sum,
            self.x_g_recon_sum, self.y_g_recon_sum, self.x_d_real_sum,
            self.x_d_fake_sum, self.y_d_real_sum, self.y_d_fake_sum,
            self.d_loss_sum, self.g_loss_sum
        ])

    def _build_optimizer(self, scope):
        g_total_loss = self.g_loss
        d_total_loss = self.d_loss

        update_ops_g = tf.get_collection(
            self.updates_collections_g, scope=scope.name)
        with tf.control_dependencies(update_ops_g):
            self.g_optim = tf.train.AdamOptimizer(
                self.g_learning_rate, beta1=self.g_beta1).minimize(
                    g_total_loss, var_list=self.x_g_vars + self.y_g_vars)

        update_ops_d = tf.get_collection(
            self.updates_collections_d, scope=scope.name)
        with tf.control_dependencies(update_ops_d + update_ops_g):
            self.d_optim = tf.train.AdamOptimizer(
                self.d_learning_rate, beta1=self.d_beta1).minimize(
                    d_total_loss, var_list=self.x_d_vars + self.y_d_vars)

    def train(self,
              num_epochs,
              resume=True,
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

            success, step = False, 0
            if resume and checkpoint_dir:
                success, saved_step = self.load(checkpoint_dir)

            if success:
                step = saved_step + 1
                start_epoch = (step * self.batch_size) // self.num_examples
            else:
                start_epoch = 0

            num_batches = self.num_examples // self.batch_size
            t = self._trange(start_epoch, num_epochs)
            for epoch in t:
                start_idx = step % num_batches
                epoch_g_loss = []
                epoch_d_loss = []
                for idx in range(start_idx, num_batches):
                    _, _, d_loss, g_loss, summary_str = self.sess.run(
                        [
                            self.d_optim, self.g_optim, self.d_loss,
                            self.g_loss, self.summary
                        ],
                        feed_dict={self.is_training: True})
                    epoch_d_loss.append(d_loss)
                    epoch_g_loss.append(g_loss)

                    if self.writer:
                        self.writer.add_summary(summary_str, step)
                    step += 1

                    # Save checkpoint
                    if checkpoint_dir and step % save_step == 0:
                        self.save(checkpoint_dir, step)

                    # Sample
                    if sample_fn and sample_step and (
                        (isinstance(sample_step, int) and
                         step % sample_step == 0) or (not isinstance(
                             sample_step, int) and step in sample_step)):
                        sample_fn(self, step)

                t.set_postfix(
                    g_loss=np.mean(epoch_g_loss), d_loss=np.mean(epoch_d_loss))

    def sample_x(self, y=None):
        if y is not None:
            return [y] + self.sess.run(
                [self.sampler_for_x, self.sampler_recon_for_y],
                feed_dict={self.is_training: False,
                           self.y_sampler: y})
        else:
            return self.sess.run(
                [self.Y, self.x_g, self.y_g_recon],
                feed_dict={self.is_training: False})

    def sample_y(self, x=None):
        if x is not None:
            return [x] + self.sess.run(
                [self.sampler_for_y, self.sampler_recon_for_x],
                feed_dict={self.is_training: False,
                           self.x_sampler: x})
        else:
            return self.sess.run(
                [self.X, self.y_g, self.x_g_recon],
                feed_dict={self.is_training: False})
