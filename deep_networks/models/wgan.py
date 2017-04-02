import os
import time

import numpy as np
import tensorflow as tf

from tqdm import trange

from .base import Model
from .gan import build_basic_discriminator, build_basic_generator


class WGAN(Model):
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
                 g_learning_rate=0.00005,
                 d_learning_rate=0.00005,
                 d_clamp_lower=-0.05,
                 d_clamp_upper=0.05,
                 d_iters=5,
                 d_high_iters=100,
                 d_intial_high_rounds=25,
                 d_step_high_rounds=500,
                 generator_fn=build_basic_generator,
                 discriminator_fn=build_basic_discriminator,
                 image_summary=False,
                 name='WGAN'):
        with tf.variable_scope(name) as scope:
            super().__init__(sess=sess, name=name)

            self.output_shape = output_shape
            self.batch_size = batch_size
            self.num_examples = num_examples

            self.z_stddev = z_stddev
            self.z_dim = z_dim

            self.g_dim = g_dim
            self.g_learning_rate = g_learning_rate

            self.d_dim = d_dim
            self.d_learning_rate = d_learning_rate
            self.d_clamp_lower = d_clamp_lower
            self.d_clamp_upper = d_clamp_upper
            self.d_iters = 5
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
            self.is_training = tf.placeholder(tf.bool, [], name='is_training')
            self.update_ops_noop = self.name + '/update_ops_noop'
            self.update_ops_d = self.name + '/update_ops_d'
            self.update_ops_g = self.name + '/update_ops_g'

            self._build_GAN(generator_fn, discriminator_fn)
            self._build_summary()
            self._build_optimizer(scope)

            self.saver = tf.train.Saver()
            tf.global_variables_initializer().run()

    def _build_GAN(self, generator_fn, discriminator_fn):
        self.g = generator_fn(
            self.z,
            self.is_training,
            self.update_ops_g,
            self.output_shape,
            dim=self.g_dim,
            name='generator')

        self.d_real, self.d_logits_real = discriminator_fn(
            self.X,
            self.is_training,
            self.update_ops_d,
            dim=self.d_dim,
            activation_fn=None,
            name='discriminator')
        self.d_fake, self.d_logits_fake = discriminator_fn(
            self.g,
            self.is_training,
            self.update_ops_d,
            dim=self.d_dim,
            activation_fn=None,
            reuse=True,
            name='discriminator')

        self.g_loss = -tf.reduce_mean(self.d_logits_fake)

        self.d_loss_real = tf.reduce_mean(self.d_logits_real)
        self.d_loss_fake = tf.reduce_mean(self.d_logits_fake)
        self.d_loss = self.d_loss_real - self.d_loss_fake

        self.g_vars = []
        self.d_vars = []
        for var in tf.trainable_variables():
            if self._is_component('generator', var.name):
                self.g_vars.append(var)
            elif self._is_component('discriminator', var.name):
                self.d_vars.append(var)

        self.z_sampler = tf.placeholder(
            tf.float32, [None, self.z_dim], name='z_sampler')
        self.sampler = generator_fn(
            self.z_sampler,
            self.is_training,
            self.update_ops_noop,
            self.output_shape,
            dim=self.g_dim,
            name='generator',
            reuse=True)

    def _build_summary(self):
        self.z_sum = tf.summary.histogram('z', self.z)
        if self.image_summary:
            self.g_sum = tf.summary.image('g',
                                          tf.reshape(self.g, self.ouput_shape))
        else:
            self.g_sum = tf.summary.histogram('g', self.g)
        self.d_real_sum = tf.summary.histogram('d_real', self.d_real)
        self.d_fake_sum = tf.summary.histogram('d_fake', self.d_fake)

        self.g_loss_sum = tf.summary.scalar('g_loss', self.g_loss)
        self.d_loss_sum = tf.summary.scalar('d_loss', self.d_loss)

        self.d_sum = tf.summary.merge(
            [self.z_sum, self.d_real_sum, self.d_fake_sum, self.d_loss_sum])
        self.summary = tf.summary.merge([
            self.z_sum, self.d_real_sum, self.d_fake_sum, self.d_loss_sum,
            self.g_sum, self.g_loss_sum
        ])

    def _build_optimizer(self, scope):
        g_total_loss = self.g_loss
        d_total_loss = -self.d_loss

        update_ops_g = tf.get_collection(self.update_ops_g, scope=scope.name)
        with tf.control_dependencies(update_ops_g):
            self.g_optim = tf.train.RMSPropOptimizer(
                self.g_learning_rate).minimize(
                    g_total_loss, var_list=self.g_vars)

        d_clip = [
            v.assign(
                tf.clip_by_value(v, self.d_clamp_lower, self.d_clamp_upper))
            for v in self.d_vars
        ]
        update_ops_d = tf.get_collection(self.update_ops_g, scope=scope.name)
        with tf.control_dependencies(update_ops_d + update_ops_g + d_clip):
            self.d_optim = tf.train.RMSPropOptimizer(
                self.d_learning_rate).minimize(
                    d_total_loss, var_list=self.d_vars)

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
            initial_steps = self.d_high_iters * self.d_intial_high_rounds
            # steps to free from initial steps
            passing_steps = initial_steps + (
                (self.d_step_high_rounds -
                 (self.d_intial_high_rounds % self.d_step_high_rounds
                  )) % self.d_step_high_rounds) * self.d_iters
            block_steps = self.d_high_iters + (self.d_step_high_rounds - 1
                                               ) * self.d_iters
            t = trange(start_epoch, num_epochs)
            for epoch in t:
                start_idx = step % num_batches
                epoch_g_loss = []
                epoch_d_loss = []

                def train_D():
                    _, d_loss, summary_str = self.sess.run(
                        [self.d_optim, self.d_loss, self.d_sum],
                        feed_dict={self.is_training: True})
                    epoch_d_loss.append(d_loss)
                    return summary_str

                def train_D_G():
                    # Update generator
                    _, _, d_loss, g_loss, summary_str = self.sess.run(
                        [
                            self.d_optim, self.g_optim, self.d_loss,
                            self.g_loss, self.summary
                        ],
                        feed_dict={self.is_training: True})
                    epoch_d_loss.append(d_loss)
                    epoch_g_loss.append(g_loss)
                    return summary_str

                # the complicated loop is to achieve the following
                # with restore cabability
                #
                # gen_iterations = 0
                # while True:
                #    if gen_iterations < self.d_intial_high_rounds or gen_iterations % self.d_step_high_rounds == 0:
                #        d_iters = self.d_high_iters
                #    else:
                #        d_iters = self.d_iters
                #    for _ in range(d_iters):
                #        train D
                #    train G
                for idx in range(start_idx, num_batches):
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

    def _is_component(self, component, name):
        prefix = self.name + '/' + component + '/'
        return prefix in name
