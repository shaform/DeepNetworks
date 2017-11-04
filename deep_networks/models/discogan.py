import datetime
import functools
import operator
import os

import tensorflow as tf

from .base import GANModel
from .blocks import BasicGenerator, BasicDiscriminator
from ..train import IncrementalAverage


class DiscoGAN(GANModel):
    def __init__(self,
                 sess,
                 X_real,
                 Y_real,
                 num_examples,
                 x_output_shape,
                 y_output_shape,
                 reg_const=5e-5,
                 stddev=None,
                 batch_size=128,
                 g_learning_rate=0.0001,
                 g_beta1=0.5,
                 g_beta2=0.9,
                 d_learning_rate=0.0001,
                 d_lambda=10.0,
                 d_beta1=0.5,
                 d_beta2=0.9,
                 d_iters=5,
                 d_high_iters=0,
                 d_intial_high_rounds=25,
                 d_step_high_rounds=500,
                 generator_cls=BasicGenerator,
                 discriminator_cls=BasicDiscriminator,
                 image_summary=False,
                 name='DiscoGAN'):
        with tf.variable_scope(name):
            super().__init__(
                sess=sess,
                name=name,
                num_examples=num_examples,
                output_shape=None,
                reg_const=reg_const,
                stddev=stddev,
                batch_size=batch_size,
                image_summary=image_summary)

            self.x_output_shape = x_output_shape
            self.y_output_shape = y_output_shape
            x_output_size = functools.reduce(operator.mul, x_output_shape)
            y_output_size = functools.reduce(operator.mul, y_output_shape)

            self.g_learning_rate = g_learning_rate
            self.g_beta1 = g_beta1
            self.g_beta2 = g_beta2

            self.d_learning_rate = d_learning_rate
            self.d_beta1 = d_beta1
            self.d_beta2 = d_beta2
            self.d_lambda = d_lambda
            self.d_iters = d_iters
            self.d_high_iters = d_high_iters
            self.d_intial_high_rounds = d_intial_high_rounds
            self.d_step_high_rounds = d_step_high_rounds

            self.X = X_real
            self.X = tf.placeholder_with_default(self.X, [None, x_output_size])
            self.Y = Y_real
            self.Y = tf.placeholder_with_default(self.Y, [None, y_output_size])
            self.epsilon = tf.random_uniform(
                (batch_size, 1),
                minval=0.0,
                maxval=1.0,
                dtype=tf.float32,
                name='epsilon')
            self.updates_collections_noop = 'updates_collections_noop'

            self._build_GAN(generator_cls, discriminator_cls)
            self._build_losses()
            self._build_optimizer()
            self._build_summary()

            self.saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())

    def _build_GAN(self, generator_cls, discriminator_cls):
        self.x_g = generator_cls(
            inputs=self.Y,
            output_shape=self.x_output_shape,
            initializer=self.initializer,
            name='x_generator')
        self.y_g = generator_cls(
            inputs=self.X,
            output_shape=self.y_output_shape,
            initializer=self.initializer,
            name='y_generator')

        self.x_g_recon = generator_cls(
            inputs=self.y_g.activations,
            output_shape=self.x_output_shape,
            initializer=self.initializer,
            reuse=True,
            name='x_generator')
        self.y_g_recon = generator_cls(
            inputs=self.x_g.activations,
            output_shape=self.y_output_shape,
            initializer=self.initializer,
            reuse=True,
            name='y_generator')

        self.x_d_real = discriminator_cls(
            inputs=self.X,
            input_shape=self.x_output_shape,
            regularizer=self.regularizer,
            initializer=self.initializer,
            disc_activation_fn=None,
            name='x_discriminator')

        self.y_d_real = discriminator_cls(
            inputs=self.Y,
            input_shape=self.y_output_shape,
            regularizer=self.regularizer,
            initializer=self.initializer,
            disc_activation_fn=None,
            name='y_discriminator')

        self.x_d_fake = discriminator_cls(
            inputs=self.x_g.activations,
            input_shape=self.x_output_shape,
            regularizer=self.regularizer,
            initializer=self.initializer,
            disc_activation_fn=None,
            reuse=True,
            name='x_discriminator')

        self.y_d_fake = discriminator_cls(
            inputs=self.y_g.activations,
            input_shape=self.y_output_shape,
            regularizer=self.regularizer,
            initializer=self.initializer,
            disc_activation_fn=None,
            reuse=True,
            name='y_discriminator')

        self.x_d_recon = discriminator_cls(
            inputs=self.x_g_recon.activations,
            input_shape=self.x_output_shape,
            regularizer=self.regularizer,
            initializer=self.initializer,
            disc_activation_fn=None,
            reuse=True,
            name='x_discriminator')

        self.y_d_recon = discriminator_cls(
            inputs=self.y_g_recon.activations,
            input_shape=self.y_output_shape,
            regularizer=self.regularizer,
            initializer=self.initializer,
            disc_activation_fn=None,
            reuse=True,
            name='y_discriminator')

        self.X_hat = self.X * self.epsilon + self.x_g.activations * (
            1. - self.epsilon)

        self.Y_hat = self.Y * self.epsilon + self.y_g.activations * (
            1. - self.epsilon)

        self.x_d_hat = discriminator_cls(
            inputs=self.X_hat,
            input_shape=self.x_output_shape,
            regularizer=self.regularizer,
            initializer=self.initializer,
            disc_activation_fn=None,
            reuse=True,
            name='x_discriminator')

        self.y_d_hat = discriminator_cls(
            inputs=self.Y_hat,
            input_shape=self.y_output_shape,
            regularizer=self.regularizer,
            initializer=self.initializer,
            disc_activation_fn=None,
            reuse=True,
            name='y_discriminator')

        self.x_g_vars = self.x_g.get_vars()
        self.y_g_vars = self.y_g.get_vars()
        self.x_d_vars = self.x_d_real.get_vars()
        self.y_d_vars = self.y_d_real.get_vars()

    def _build_losses(self):
        with tf.variable_scope('x_generator') as scope:
            self.x_recon_loss = tf.reduce_sum(
                tf.losses.mean_squared_error(
                    self.X, self.x_g_recon.activations)) + self.feats_loss(
                        self.x_d_real.features[1:],
                        self.x_d_recon.features[1:])

            self.x_g_loss = -tf.reduce_mean(
                self.x_d_fake.disc_outputs) + self.feats_loss(
                    self.x_d_real.features[1:], self.x_d_fake.features[1:])

        with tf.variable_scope('y_generator') as scope:
            self.y_recon_loss = tf.reduce_sum(
                tf.losses.mean_squared_error(
                    self.Y, self.y_g_recon.activations)) + self.feats_loss(
                        self.y_d_real.features[1:],
                        self.y_d_recon.features[1:])

            self.y_g_loss = -tf.reduce_mean(
                self.y_d_fake.disc_outputs) + self.feats_loss(
                    self.y_d_real.features[1:], self.y_d_fake.features[1:])

        with tf.variable_scope('x_discriminator') as scope:
            self.x_d_loss_real = tf.reduce_mean(self.x_d_real.disc_outputs)
            self.x_d_loss_fake = tf.reduce_mean(self.x_d_fake.disc_outputs)
            self.x_d_loss = self.x_d_loss_fake - self.x_d_loss_real
            self.x_d_grad = tf.gradients(self.x_d_hat.disc_outputs, [
                self.X_hat,
            ])[0]
            self.x_d_grad_loss = self.d_lambda * tf.reduce_mean(
                tf.square(
                    tf.sqrt(tf.reduce_sum(tf.square(self.x_d_grad), 1)) - 1.0))

            x_d_reg_ops = tf.get_collection(
                tf.GraphKeys.REGULARIZATION_LOSSES, scope=scope.name)
            self.x_d_reg_loss = tf.add_n(x_d_reg_ops) if x_d_reg_ops else 0.0
        with tf.variable_scope('y_discriminator') as scope:
            self.y_d_loss_real = tf.reduce_mean(self.y_d_real.disc_outputs)
            self.y_d_loss_fake = tf.reduce_mean(self.y_d_fake.disc_outputs)
            self.y_d_loss = self.y_d_loss_fake - self.y_d_loss_real
            self.y_d_grad = tf.gradients(self.y_d_hat.disc_outputs, [
                self.Y_hat,
            ])[0]
            self.y_d_grad_loss = self.d_lambda * tf.reduce_mean(
                tf.square(
                    tf.sqrt(tf.reduce_sum(tf.square(self.y_d_grad), 1)) - 1.0))

            y_d_reg_ops = tf.get_collection(
                tf.GraphKeys.REGULARIZATION_LOSSES, scope=scope.name)
            self.y_d_reg_loss = tf.add_n(y_d_reg_ops) if y_d_reg_ops else 0.0

        self.g_total_loss = (self.x_g_loss + self.y_g_loss + self.x_recon_loss
                             + self.y_recon_loss)
        self.d_total_loss = (
            self.x_d_loss + self.y_d_loss + self.x_d_grad_loss +
            self.y_d_grad_loss + self.x_d_reg_loss + self.y_d_reg_loss)

    def _build_summary(self):
        with tf.variable_scope('summary') as scope:
            if self.image_summary:
                self.x_sum = tf.summary.image(
                    'x', tf.reshape(self.X, (-1, ) + self.x_output_shape))
                self.y_sum = tf.summary.image(
                    'y', tf.reshape(self.Y, (-1, ) + self.y_output_shape))
                self.x_g_sum = tf.summary.image(
                    'x_g',
                    tf.reshape(self.x_g.activations,
                               (-1, ) + self.x_output_shape))
                self.y_g_sum = tf.summary.image(
                    'y_g',
                    tf.reshape(self.y_g.activations,
                               (-1, ) + self.y_output_shape))
                self.x_g_recon_sum = tf.summary.image(
                    'x_g_recon',
                    tf.reshape(self.x_g_recon.activations,
                               (-1, ) + self.x_output_shape))
                self.y_g_recon_sum = tf.summary.image(
                    'y_g_recon',
                    tf.reshape(self.y_g_recon.activations,
                               (-1, ) + self.y_output_shape))
            else:
                self.x_sum = tf.summary.histogram('x', self.X)
                self.y_sum = tf.summary.histogram('y', self.Y)
                self.x_g_sum = tf.summary.histogram('x_g',
                                                    self.x_g.activations)
                self.y_g_sum = tf.summary.histogram('y_g',
                                                    self.y_g.activations)
                self.x_g_recon_sum = tf.summary.histogram(
                    'x_g_recon', self.x_g_recon.activations)
                self.y_g_recon_sum = tf.summary.histogram(
                    'y_g_recon', self.y_g_recon.activations)
            self.x_d_loss_sum = tf.summary.histogram('x_d_loss', self.x_d_loss)
            self.y_d_loss_sum = tf.summary.histogram('y_d_loss', self.y_d_loss)

            self.g_total_loss_sum = tf.summary.scalar('g_total_loss',
                                                      self.g_total_loss)
            self.d_total_loss_sum = tf.summary.scalar('d_total_loss',
                                                      self.d_total_loss)

            self.summary = tf.summary.merge(
                tf.get_collection(tf.GraphKeys.SUMMARIES, scope=scope.name))

    def _build_optimizer(self):
        self.g_optim = tf.train.AdamOptimizer(
            self.g_learning_rate, beta1=self.g_beta1).minimize(
                self.g_total_loss, var_list=self.x_g_vars + self.y_g_vars)

        self.d_optim = tf.train.AdamOptimizer(
            self.d_learning_rate, beta1=self.d_beta1).minimize(
                self.d_total_loss, var_list=self.x_d_vars + self.y_d_vars)

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
                    (_, _, d_total_loss, g_total_loss,
                     summary_str) = self.sess.run([
                         self.d_optim, self.g_optim, self.d_total_loss,
                         self.g_total_loss, self.summary
                     ])
                    epoch_d_total_loss.add(d_total_loss)
                    epoch_g_total_loss.add(g_total_loss)
                    return summary_str

                # the complicated loop is to achieve the following
                # with restore capability
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

            # Save final checkpoint
            if checkpoint_dir:
                self.save(checkpoint_dir, step)

    def sample_x(self, y=None):
        if y is not None:
            return [y] + self.sess.run(
                [self.x_g.activations, self.y_g_recon.activations],
                feed_dict={self.is_training: False,
                           self.Y: y})
        else:
            return self.sess.run(
                [self.Y, self.x_g.activations, self.y_g_recon.activations],
                feed_dict={self.is_training: False})

    def sample_y(self, x=None):
        if x is not None:
            return [x] + self.sess.run(
                [self.y_g.activations, self.x_g_recon.activations],
                feed_dict={self.is_training: False,
                           self.X: x})
        else:
            return self.sess.run(
                [self.X, self.y_g.activations, self.x_g_recon.activations],
                feed_dict={self.is_training: False})

    def feats_loss(self, real_feats, fake_feats):
        losses = tf.constant(0.)

        for real_feat, fake_feat in zip(real_feats, fake_feats):
            losses += tf.reduce_mean(
                tf.losses.mean_squared_error(real_feat, fake_feat))
        return losses
