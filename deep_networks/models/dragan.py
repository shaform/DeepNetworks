"""
Deep Regret Analytic Generative Adversarial Networks
"""

import datetime
import os

import numpy as np
import tensorflow as tf

from .base import GANModel
from .blocks import BasicGenerator, BasicDiscriminator
from ..ops import dragan_perturb
from ..train import IncrementalAverage


class DRAGAN(GANModel):
    def __init__(self,
                 sess,
                 X_real,
                 num_examples,
                 output_shape,
                 reg_const=5e-5,
                 stddev=None,
                 z_dim=10,
                 z_stddev=1.,
                 batch_size=128,
                 g_learning_rate=0.0002,
                 g_beta1=0.5,
                 d_learning_rate=0.0002,
                 d_beta1=0.5,
                 d_label_smooth=0.25,
                 lambda_gp=0.5,
                 lambda_dra=0.5,
                 generator_cls=BasicGenerator,
                 discriminator_cls=BasicDiscriminator,
                 image_summary=False,
                 name='DRAGAN'):
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

            self.g_learning_rate = g_learning_rate
            self.g_beta1 = g_beta1

            self.d_learning_rate = d_learning_rate
            self.d_beta1 = d_beta1
            self.d_label_smooth = d_label_smooth

            self.lambda_gp = lambda_gp
            self.lambda_dra = lambda_dra

            self.X = X_real
            self.z = tf.random_normal(
                (batch_size, z_dim),
                mean=0.0,
                stddev=z_stddev,
                name='z',
                dtype=tf.float32)
            self.z = tf.placeholder_with_default(self.z, [None, z_dim])

            self.eps_dra = tf.random_uniform(
                (self.batch_size, 1),
                minval=-1.0,
                maxval=1.0,
                dtype=tf.float32,
                name='eps_dra')

            self._build_GAN(generator_cls, discriminator_cls)
            self._build_losses()
            self._build_optimizer()
            self._build_summary()

            self.saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())

    def _build_GAN(self, generator_cls, discriminator_cls):
        self.g = generator_cls(
            inputs=self.z,
            output_shape=self.output_shape,
            initializer=self.initializer,
            name='generator')

        self.d_real = discriminator_cls(
            inputs=self.X,
            input_shape=self.output_shape,
            regularizer=self.regularizer,
            initializer=self.initializer,
            name='discriminator')
        self.d_fake = discriminator_cls(
            inputs=self.g.activations,
            input_shape=self.output_shape,
            regularizer=self.regularizer,
            initializer=self.initializer,
            reuse=True,
            name='discriminator')

        if self.lambda_gp:
            self.X_hat = dragan_perturb(self.X, self.eps_dra, self.lambda_dra)
            self.d_hat = discriminator_cls(
                inputs=self.X_hat,
                input_shape=self.output_shape,
                regularizer=self.regularizer,
                initializer=self.initializer,
                reuse=True,
                name='discriminator')

        self.g_vars = self.g.get_vars()
        self.d_vars = self.d_real.get_vars()

    def _build_losses(self):
        with tf.variable_scope('generator'):
            self.g_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.d_fake.disc_outputs,
                    labels=tf.ones_like(self.d_fake.disc_outputs)))
            self.g_total_loss = self.g_loss

        with tf.variable_scope('discriminator'):
            if self.d_label_smooth > 0.0:
                labels_real = tf.ones_like(
                    self.d_real.disc_outputs) - self.d_label_smooth
            else:
                labels_real = tf.ones_like(self.d_real.disc_outputs)

            self.d_loss_real = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.d_real.disc_outputs, labels=labels_real))
            self.d_loss_fake = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.d_fake.disc_outputs,
                    labels=tf.zeros_like(self.d_fake.disc_outputs)))
            self.d_loss = self.d_loss_real + self.d_loss_fake

            self.d_reg_loss = self.d_real.reg_loss()
            self.d_total_loss = self.d_loss + self.d_reg_loss

            if self.lambda_gp:
                self.d_grad_loss = self.d_hat.gp_loss(self.lambda_gp)
                self.d_total_loss += self.d_grad_loss

    def _build_summary(self):
        with tf.variable_scope('summary') as scope:
            self.z_sum = tf.summary.histogram('z', self.z)
            if self.image_summary:
                self.g_sum = tf.summary.image('g',
                                              tf.reshape(
                                                  self.g.activations,
                                                  (-1, ) + self.output_shape))
            else:
                self.g_sum = tf.summary.histogram('g', self.g.activations)
            self.d_real_sum = tf.summary.histogram(
                'd_real', self.d_real.disc_activations)
            self.d_fake_sum = tf.summary.histogram(
                'd_fake', self.d_fake.disc_activations)

            self.g_loss_sum = tf.summary.scalar('g_loss', self.g_loss)
            self.g_total_loss_sum = tf.summary.scalar('g_total_loss',
                                                      self.g_total_loss)
            self.d_loss_sum = tf.summary.scalar('d_loss', self.d_loss)
            self.d_loss_real_sum = tf.summary.scalar('d_loss_real',
                                                     self.d_loss_real)
            self.d_loss_fake_sum = tf.summary.scalar('d_loss_fake',
                                                     self.d_loss_fake)
            if self.regularizer is not None:
                self.d_reg_loss_sum = tf.summary.scalar(
                    'd_reg_loss', self.d_reg_loss)
            self.d_total_loss_sum = tf.summary.scalar('d_total_loss',
                                                      self.d_total_loss)

            self.summary = tf.summary.merge(
                tf.get_collection(tf.GraphKeys.SUMMARIES, scope=scope.name))

    def _build_optimizer(self):
        with tf.variable_scope('generator'):
            self.g_optim = tf.train.AdamOptimizer(
                self.g_learning_rate, beta1=self.g_beta1).minimize(
                    self.g_total_loss, var_list=self.g_vars)

        with tf.variable_scope('discriminator'):
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
                    start_idx,
                    num_batches,
                    desc='Epoch #{}'.format(epoch + 1),
                    leave=False)
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
                self.g.activations,
                feed_dict={self.is_training: False,
                           self.z: z})
        elif num_samples is not None:
            return self.sess.run(
                self.g.activations,
                feed_dict={
                    self.is_training: False,
                    self.z: self.sample_z(num_samples)
                })
        else:
            return self.sess.run(
                self.g.activations, feed_dict={self.is_training: False})

    def sample_z(self, num_samples):
        return np.random.normal(0.0, self.z_stddev, (num_samples, self.z_dim))
