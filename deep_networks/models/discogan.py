import datetime
import functools
import operator
import os

import tensorflow as tf

from .base import GANModel
from .blocks import BasicGenerator, BasicDiscriminator
from ..ops import dragan_perturb
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

            self.d_learning_rate = d_learning_rate
            self.d_beta1 = d_beta1
            self.d_label_smooth = d_label_smooth

            self.lambda_gp = lambda_gp
            self.lambda_dra = lambda_dra

            self.X = X_real
            self.X = tf.placeholder_with_default(self.X, [None, x_output_size])
            self.Y = Y_real
            self.Y = tf.placeholder_with_default(self.Y, [None, y_output_size])
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

        self.X_hat = dragan_perturb(self.X, self.eps_dra, self.lambda_dra)

        self.Y_hat = dragan_perturb(self.Y, self.eps_dra, self.lambda_dra)

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

            self.x_g_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.x_d_fake.disc_outputs,
                    labels=tf.ones_like(self.x_d_fake.disc_outputs))
            ) + self.feats_loss(self.x_d_real.features[1:],
                                self.x_d_fake.features[1:])

        with tf.variable_scope('y_generator') as scope:
            self.y_recon_loss = tf.reduce_sum(
                tf.losses.mean_squared_error(
                    self.Y, self.y_g_recon.activations)) + self.feats_loss(
                        self.y_d_real.features[1:],
                        self.y_d_recon.features[1:])

            self.y_g_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.y_d_fake.disc_outputs,
                    labels=tf.ones_like(self.y_d_fake.disc_outputs))
            ) + self.feats_loss(self.y_d_real.features[1:],
                                self.y_d_fake.features[1:])

        if self.d_label_smooth > 0.0:
            labels_real = tf.ones_like(
                self.x_d_real.disc_outputs) - self.d_label_smooth
        else:
            labels_real = tf.ones_like(self.x_d_real.disc_outputs)

        with tf.variable_scope('x_discriminator') as scope:
            self.x_d_loss_real = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.x_d_real.disc_outputs, labels=labels_real))
            self.x_d_loss_fake = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.x_d_fake.disc_outputs,
                    labels=tf.zeros_like(self.x_d_fake.disc_outputs)))
            self.x_d_loss = self.x_d_loss_fake + self.x_d_loss_real
            self.x_d_grad_loss = self.x_d_hat.gp_loss(self.lambda_gp)
            self.x_d_reg_loss = self.x_d_real.reg_loss()

        with tf.variable_scope('y_discriminator') as scope:
            self.y_d_loss_real = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.y_d_real.disc_outputs, labels=labels_real))
            self.y_d_loss_fake = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.y_d_fake.disc_outputs,
                    labels=tf.zeros_like(self.y_d_fake.disc_outputs)))
            self.y_d_loss = self.y_d_loss_fake + self.y_d_loss_real
            self.y_d_grad_loss = self.y_d_hat.gp_loss(self.lambda_gp)
            self.y_d_reg_loss = self.y_d_real.reg_loss()

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
