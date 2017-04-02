import functools
import operator
import os
import time

import numpy as np
import tensorflow as tf

from tqdm import trange

from .base import Model
from ..ops import lrelu


def build_basic_generator(z,
                          c,
                          is_training,
                          update_ops,
                          output_shape,
                          num_classes,
                          name='generator',
                          reuse=False,
                          stddev=0.02,
                          dim=128,
                          num_layers=3,
                          activation_fn=None):
    assert num_layers > 0
    output_size = functools.reduce(operator.mul, output_shape)
    initializer = tf.truncated_normal_initializer(stddev=stddev)
    xavier_initializer = tf.contrib.layers.xavier_initializer()

    with tf.variable_scope(name, reuse=reuse):
        codes = tf.get_variable(
            'codes', [num_classes, z.get_shape()[1]],
            initializer=xavier_initializer,
            regularizer=tf.contrib.layers.l2_regularizer(0.8))
        z_c = tf.nn.embedding_lookup(codes, c)
        fc = tf.multiply(z, z_c)
        for i in range(num_layers - 1):
            fc = tf.contrib.layers.fully_connected(
                inputs=fc,
                num_outputs=dim,
                reuse=reuse,
                activation_fn=tf.nn.relu,
                normalizer_fn=tf.contrib.layers.batch_norm,
                normalizer_params={
                    'is_training': is_training,
                    'reuse': reuse,
                    'scope': 'g_fc{}_bn'.format(i),
                    'updates_collections': update_ops,
                },
                weights_initializer=initializer,
                biases_initializer=tf.zeros_initializer(),
                scope='g_fc{}'.format(i + 1))
        fc = tf.contrib.layers.fully_connected(
            inputs=fc,
            num_outputs=output_size,
            reuse=reuse,
            activation_fn=activation_fn,
            weights_initializer=initializer,
            biases_initializer=tf.zeros_initializer(),
            scope='g_fc{}'.format(num_layers))
        return fc, codes


def build_basic_discriminator(X,
                              is_training,
                              update_ops,
                              num_classes,
                              name='discriminator',
                              reuse=False,
                              stddev=0.02,
                              dim=128,
                              num_layers=3,
                              activation_fn=tf.nn.sigmoid,
                              class_activation_fn=tf.nn.softmax):
    assert num_layers > 0
    initializer = tf.truncated_normal_initializer(stddev=stddev)

    with tf.variable_scope(name, reuse=reuse):
        fc = X
        for i in range(num_layers - 1):
            normalizer_fn = tf.contrib.layers.batch_norm if i != 0 else None
            normalizer_params = {
                'scope': 'd_fc{}_bn'.format(i),
                'is_training': is_training,
                'updates_collections': update_ops,
                'reuse': reuse
            } if i != 0 else None
            fc = tf.contrib.layers.fully_connected(
                inputs=fc,
                num_outputs=dim,
                reuse=reuse,
                activation_fn=lrelu,
                normalizer_fn=normalizer_fn,
                normalizer_params=normalizer_params,
                weights_initializer=initializer,
                biases_initializer=tf.zeros_initializer(),
                scope='d_fc{}'.format(i))
        fc_d = tf.contrib.layers.fully_connected(
            inputs=fc,
            num_outputs=1,
            reuse=reuse,
            activation_fn=None,
            weights_initializer=initializer,
            biases_initializer=tf.zeros_initializer(),
            scope='d_fc{}'.format(num_layers))
        act_d = activation_fn(fc_d) if activation_fn else fc_d

        fc_c = tf.contrib.layers.fully_connected(
            inputs=fc,
            num_outputs=num_classes,
            reuse=reuse,
            activation_fn=None,
            weights_initializer=initializer,
            biases_initializer=tf.zeros_initializer(),
            scope='d_fc_c')
        act_c = class_activation_fn(fc_c) if class_activation_fn else fc_c
        return act_d, fc_d, act_c, fc_c


class ACGAN(Model):
    def __init__(self,
                 sess,
                 X_real,
                 y_real,
                 num_examples,
                 num_classes,
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
                 name='ACGAN'):
        with tf.variable_scope(name) as scope:
            super().__init__(sess=sess, name=name)

            self.output_shape = output_shape
            self.batch_size = batch_size
            self.num_examples = num_examples
            self.num_classes = num_classes

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
            self.y = y_real
            self.z = tf.random_normal(
                (batch_size, z_dim),
                mean=0.0,
                stddev=z_stddev,
                name='z',
                dtype=tf.float32)
            self.c = tf.random_uniform(
                (batch_size, ),
                minval=0,
                maxval=num_classes,
                dtype=tf.int64,
                name='c')
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
        self.g, self.codes = generator_fn(
            self.z,
            self.c,
            self.is_training,
            self.update_ops_g,
            self.output_shape,
            self.num_classes,
            dim=self.g_dim,
            name='generator')

        self.d_real, self.d_logits_real, self.d_c_real, self.d_c_logits_real = discriminator_fn(
            self.X,
            self.is_training,
            self.update_ops_d,
            self.num_classes,
            dim=self.d_dim,
            name='discriminator')
        self.d_fake, self.d_logits_fake, self.d_c_fake, self.d_c_logits_fake = discriminator_fn(
            self.g,
            self.is_training,
            self.update_ops_d,
            self.num_classes,
            dim=self.d_dim,
            reuse=True,
            name='discriminator')

        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.d_logits_fake,
                labels=tf.ones_like(self.d_logits_fake)))
        self.g_c_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=self.d_c_logits_fake,
                labels=tf.one_hot(self.c, depth=self.num_classes, axis=-1)))
        self.g_reg_loss = tf.add_n([
            var
            for var in tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            if self._is_component('generator', var.name)
        ])

        correct_prediction = tf.equal(self.c, tf.argmax(self.d_c_fake, 1))
        self.g_c_accuracy = tf.reduce_mean(
            tf.cast(correct_prediction, tf.float32))

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

        labels_c_real = tf.one_hot(self.y, depth=self.num_classes, axis=-1)
        if self.d_label_smooth > 0.0:
            smooth_positives = 1.0 - self.d_label_smooth
            smooth_negatives = self.d_label_smooth / self.num_classes
            labels_c_real = labels_c_real * smooth_positives + smooth_negatives

        self.d_c_loss_real = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=self.d_c_logits_real, labels=labels_c_real))
        correct_prediction_real = tf.equal(self.y,
                                           tf.argmax(self.d_c_logits_real, 1))
        self.d_c_accuracy_real = tf.reduce_mean(
            tf.cast(correct_prediction_real, tf.float32))
        self.d_c_loss = self.d_c_loss_real

        self.g_vars = []
        self.d_vars = []
        for var in tf.trainable_variables():
            if self._is_component('generator', var.name):
                self.g_vars.append(var)
            elif self._is_component('discriminator', var.name):
                self.d_vars.append(var)

        self.z_sampler = tf.placeholder(
            tf.float32, [None, self.z_dim], name='z_sampler')
        self.c_sampler = tf.placeholder(
            tf.int64, [
                None,
            ], name='c_sampler')
        self.sampler, _ = generator_fn(
            self.z_sampler,
            self.c_sampler,
            self.is_training,
            self.update_ops_noop,
            self.output_shape,
            self.num_classes,
            dim=self.g_dim,
            name='generator',
            reuse=True)

    def _build_summary(self):
        self.z_sum = tf.summary.histogram('z', self.z)
        self.c_sum = tf.summary.histogram('c', self.c)
        if self.image_summary:
            self.g_sum = tf.summary.image('g',
                                          tf.reshape(self.g, self.ouput_shape))
        else:
            self.g_sum = tf.summary.histogram('g', self.g)
        self.d_real_sum = tf.summary.histogram('d_real', self.d_real)
        self.d_fake_sum = tf.summary.histogram('d_fake', self.d_fake)

        self.d_c_real_sum = tf.summary.histogram('d_c_real', self.d_c_real)
        self.d_c_fake_sum = tf.summary.histogram('d_c_fake', self.d_c_fake)

        self.g_loss_sum = tf.summary.scalar('g_loss', self.g_loss)
        self.g_c_loss_sum = tf.summary.scalar('g_c_loss', self.g_c_loss)
        self.d_loss_sum = tf.summary.scalar('d_loss', self.d_loss)
        self.d_loss_real_sum = tf.summary.scalar('d_loss_real',
                                                 self.d_loss_real)
        self.d_loss_fake_sum = tf.summary.scalar('d_loss_fake',
                                                 self.d_loss_fake)
        self.d_c_loss_sum = tf.summary.scalar('d_c_loss', self.d_c_loss)
        self.d_c_loss_real_sum = tf.summary.scalar('d_c_loss_real',
                                                   self.d_c_loss_real)

        self.summary = tf.summary.merge([
            self.z_sum, self.c_sum, self.d_real_sum, self.d_fake_sum,
            self.d_c_real_sum, self.d_c_fake_sum, self.d_loss_real_sum,
            self.d_loss_fake_sum, self.d_loss_sum, self.d_c_loss_real_sum,
            self.d_c_loss_sum, self.g_sum, self.g_loss_sum, self.g_c_loss_sum
        ])

    def _build_optimizer(self, scope):
        g_total_loss = self.g_loss + self.g_c_loss + self.g_reg_loss
        d_total_loss = self.d_loss + self.d_c_loss

        update_ops_g = tf.get_collection(self.update_ops_g, scope=scope.name)
        with tf.control_dependencies(update_ops_g):
            self.g_optim = tf.train.AdamOptimizer(
                self.g_learning_rate, beta1=self.g_beta1).minimize(
                    g_total_loss, var_list=self.g_vars)

        update_ops_d = tf.get_collection(self.update_ops_d, scope=scope.name)
        with tf.control_dependencies(update_ops_d + update_ops_g):
            self.d_optim = tf.train.AdamOptimizer(
                self.d_learning_rate, beta1=self.d_beta1).minimize(
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
            t = trange(start_epoch, num_epochs)
            for epoch in t:
                start_idx = step % num_batches
                epoch_g_loss = []
                epoch_g_c_loss = []
                epoch_g_c_accuracy = []
                epoch_d_loss_fake = []
                epoch_d_loss_real = []
                epoch_d_c_loss_real = []
                epoch_d_c_accuracy_real = []
                for idx in range(start_idx, num_batches):
                    _, _, d_loss_fake, d_loss_real, d_c_loss_real, d_c_accuracy_real, g_loss, g_c_loss, g_c_accuracy, summary_str = self.sess.run(
                        [
                            self.d_optim, self.g_optim, self.d_loss_fake,
                            self.d_loss_real, self.d_c_loss_real,
                            self.d_c_accuracy_real, self.g_loss, self.g_c_loss,
                            self.g_c_accuracy, self.summary
                        ],
                        feed_dict={self.is_training: True})
                    epoch_d_loss_fake.append(d_loss_fake)
                    epoch_d_loss_real.append(d_loss_real)
                    epoch_d_c_loss_real.append(d_c_loss_real)
                    epoch_d_c_accuracy_real.append(d_c_accuracy_real)
                    epoch_g_loss.append(g_loss)
                    epoch_g_c_loss.append(g_c_loss)
                    epoch_g_c_accuracy.append(g_c_accuracy)

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
                    g_c_loss=np.mean(epoch_g_c_loss),
                    g_c_accuracy=np.mean(epoch_g_c_accuracy),
                    g_loss=np.mean(epoch_g_loss),
                    d_c_loss_real=np.mean(epoch_d_c_loss_real),
                    d_c_accuracy_real=np.mean(epoch_d_c_accuracy_real),
                    d_loss_real=np.mean(epoch_d_loss_real),
                    d_loss_fake=np.mean(epoch_d_loss_fake))

    def sample(self, num_samples=None, z=None, c=None):
        if z is not None and c is not None:
            return self.sess.run(
                self.sampler,
                feed_dict={
                    self.is_training: False,
                    self.z_sampler: z,
                    self.c_sampler: c
                })
        elif num_samples is not None:
            return self.sess.run(
                self.sampler,
                feed_dict={
                    self.is_training: False,
                    self.z_sampler: self.sample_z(num_samples),
                    self.c_sampler: self.sample_c(num_samples)
                })
        else:
            return self.sess.run(self.g, feed_dict={self.is_training: False})

    def sample_z(self, num_samples):
        return np.random.normal(0.0, self.z_stddev, (num_samples, self.z_dim))

    def sample_c(self, num_samples):
        return np.random.randint(self.num_classes, size=(num_samples, ))

    def _is_component(self, component, name):
        prefix = self.name + '/' + component + '/'
        return prefix in name
