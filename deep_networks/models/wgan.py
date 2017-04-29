"""
Wasserstein Generative Adversarial Networks
"""

import datetime
import os

import numpy as np
import tensorflow as tf

from .base import GANModel
from .gan import BasicGenerator, BasicDiscriminator
from ..train import IncrementalAverage


class WGAN(GANModel):
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
                 g_learning_rate=0.00005,
                 d_learning_rate=0.00005,
                 d_clamp_lower=-0.05,
                 d_clamp_upper=0.05,
                 d_iters=5,
                 d_high_iters=100,
                 d_intial_high_rounds=25,
                 d_step_high_rounds=500,
                 generator_cls=BasicGenerator,
                 discriminator_cls=BasicDiscriminator,
                 image_summary=False,
                 name='WGAN'):
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

            self.d_dim = d_dim
            self.d_learning_rate = d_learning_rate
            self.d_clamp_lower = d_clamp_lower
            self.d_clamp_upper = d_clamp_upper
            self.d_iters = d_iters
            self.d_high_iters = d_high_iters
            self.d_intial_high_rounds = d_intial_high_rounds
            self.d_step_high_rounds = d_step_high_rounds

            self.image_summary = image_summary

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
            activation_fn=None,
            skip_last_biases=True,
            name='discriminator')
        self.d_fake = discriminator_cls(
            X=self.g.outputs,
            is_training=self.is_training,
            input_shape=self.output_shape,
            regularizer=self.regularizer,
            initializer=self.initializer,
            dim=self.d_dim,
            activation_fn=None,
            skip_last_biases=True,
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
            self.g_loss = -tf.reduce_mean(self.d_fake.outputs_d)

            g_reg_ops = tf.get_collection(
                tf.GraphKeys.REGULARIZATION_LOSSES, scope=scope.name)
            self.g_reg_loss = tf.add_n(g_reg_ops) if g_reg_ops else 0.0

            self.g_total_loss = self.g_loss + self.g_reg_loss

        with tf.variable_scope('discriminator') as scope:
            self.d_loss_real = tf.reduce_mean(self.d_real.outputs_d)
            self.d_loss_fake = tf.reduce_mean(self.d_fake.outputs_d)
            self.d_loss = self.d_loss_fake - self.d_loss_real

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
                self.g_optim = tf.train.RMSPropOptimizer(
                    self.g_learning_rate).minimize(
                        self.g_total_loss, var_list=self.g_vars)

        with tf.variable_scope('discriminator') as scope:
            d_clip = [
                v.assign(
                    tf.clip_by_value(v, self.d_clamp_lower,
                                     self.d_clamp_upper)) for v in self.d_vars
            ]
            update_ops_d = tf.get_collection(
                tf.GraphKeys.UPDATE_OPS, scope=scope.name)
            with tf.control_dependencies(update_ops_d + update_ops_g + d_clip):
                self.d_optim = tf.train.RMSPropOptimizer(
                    self.d_learning_rate).minimize(
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

            initial_steps = self.d_high_iters * self.d_intial_high_rounds
            # steps to free from initial steps
            passing_steps = initial_steps + (
                (self.d_step_high_rounds -
                 (self.d_intial_high_rounds % self.d_step_high_rounds)
                 ) % self.d_step_high_rounds) * self.d_iters
            block_steps = self.d_high_iters + (
                self.d_step_high_rounds - 1) * self.d_iters
            for epoch in range(start_epoch, num_epochs):
                start_idx = step % num_batches
                epoch_g_total_loss = IncrementalAverage()
                epoch_d_total_loss = IncrementalAverage()

                def train_D():
                    _, d_total_loss = self.sess.run(
                        [self.d_optim, self.d_total_loss])
                    epoch_d_total_loss.add(d_total_loss)
                    return None

                def train_D_G():
                    # Update generator
                    _, _, d_total_loss, g_total_loss, summary_str = self.sess.run(
                        [
                            self.d_optim, self.g_optim, self.d_total_loss,
                            self.g_total_loss, self.summary
                        ])
                    epoch_d_total_loss.add(d_total_loss)
                    epoch_g_total_loss.add(g_total_loss)
                    return summary_str

                # the complicated loop is to achieve the following
                # with restore cabability
                #
                # gen_iterations = 0
                # while True:
                #    if (gen_iterations < self.d_intial_high_rounds or
                #        gen_iterations % self.d_step_high_rounds == 0):
                #        d_iters = self.d_high_iters
                #    else:
                #        d_iters = self.d_iters
                #    for _ in range(d_iters):
                #        train D
                #    train G
                t = self._trange(
                    start_idx, num_batches, desc='Epoch #{}'.format(epoch + 1))
                for idx in t:
                    # initially we train discriminator more
                    if step < initial_steps:
                        if (step + 1) % self.d_high_iters != 0:
                            summary_str = train_D()
                        else:
                            summary_str = train_D_G()
                    elif step < passing_steps:
                        passing_step = (step - initial_steps) % self.d_iters
                        if (passing_step + 1) % self.d_iters != 0:
                            summary_str = train_D()
                        else:
                            summary_str = train_D_G()
                    else:
                        block_step = (step - passing_steps) % block_steps
                        if (block_step + 1) < self.d_high_iters or (
                                block_step + 1 - self.d_high_iters
                        ) % self.d_iters != 0:
                            # train D
                            summary_str = train_D()
                        else:
                            # train G
                            summary_str = train_D_G()

                    if self.writer and summary_str:
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
