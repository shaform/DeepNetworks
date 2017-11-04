"""Base for models"""
import logging
import os

import tensorflow as tf

from tqdm import trange

from ..layers import dense_with_weight_norm

logger = logging.getLogger(__name__)


class BaseBlock(object):
    def __init__(self, scope, reuse):
        self.scope = scope
        self.reuse = reuse

    def log_name(self):
        if not self.reuse:
            logger.info('==== %s (%s) ====', self.scope.name,
                        self.__class__.__name__)

    def log_msg(self, *args, **kwargs):
        if not self.reuse:
            logger.info(*args, **kwargs)

    def get_vars(self):
        return tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope.name)

    def reg_loss(self):
        reg_vars = tf.get_collection(
            tf.GraphKeys.REGULARIZATION_LOSSES, scope=self.scope.name)
        return tf.add_n(reg_vars) if reg_vars else tf.constant(0.0)

    def update_ops(self):
        return tf.get_collection(
            tf.GraphKeys.UPDATE_OPS, scope=self.scope.name)


class BaseGenerator(BaseBlock):
    def build_latents(self, *inputs):
        if len(inputs) == 1:
            return inputs
        else:
            return tf.concat(inputs, axis=-1)


class BaseImageGenerator(BaseGenerator):
    def compute_upsamples(self, output_shape, min_size, min_dim, max_dim):
        shape = list(output_shape)[:2]
        nb_upsamples = 0
        while all(size % 2 == 0 and size > min_size for size in shape):
            shape[0] //= 2
            shape[1] //= 2
            nb_upsamples += 1

        dim = max_dim
        upsamples = []
        for _ in range(nb_upsamples + 1):
            if dim >= min_dim:
                upsamples.append(dim)
                dim //= 2
        upsamples = [max_dim] * (len(upsamples) - nb_upsamples - 1) + upsamples

        return shape, upsamples


class BaseImageDiscriminator(BaseBlock):
    def compute_downsamples(self, input_shape, min_size, min_dim, max_dim):
        shape = list(input_shape)[:2]
        nb_downsamples = 0
        while all(size % 2 == 0 and size > min_size for size in shape):
            shape[0] //= 2
            shape[1] //= 2
            nb_downsamples += 1

        dim = min_dim
        downsamples = []
        for _ in range(nb_downsamples + 1):
            downsamples.append(dim)
            dim = min(dim * 2, max_dim)

        return shape, downsamples

    def build_disc_outputs(self, inputs, initializer, regularizer):
        return dense_with_weight_norm(
            inputs=inputs,
            units=1,
            activation=None,
            kernel_initializer=initializer,
            kernel_regularizer=regularizer,
            use_bias=True,
            bias_initializer=tf.zeros_initializer())

    def build_cls_outputs(self, inputs, num_classes, initializer, regularizer):
        return dense_with_weight_norm(
            inputs=inputs,
            units=num_classes,
            activation=None,
            kernel_initializer=initializer,
            kernel_regularizer=regularizer,
            use_bias=True,
            bias_initializer=tf.zeros_initializer())


class Model(object):
    """Model"""

    def __init__(self, sess, name):
        self.name = name
        self.sess = sess
        self.saver = None
        self._trange = trange

    def save(self, checkpoint_dir, step):
        """save model

        :param checkpoint_dir: the directory to save checkpoints
        :param step: step number for the checkpoint
        """
        checkpoint_dir = os.path.join(checkpoint_dir, self.name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.init_saver()

        ckpt_name = self.saver.save(
            self.sess,
            os.path.join(checkpoint_dir, self.name),
            global_step=step)
        tf.logging.debug('Saved checkpoint %s', ckpt_name)

    def load(self, checkpoint_dir, step=None):
        """Load saved model

        :param checkpoint_dir: the checkpoint directory
        :param step: which step to load
        """
        import re
        checkpoint_dir = os.path.join(checkpoint_dir, self.name)
        self.init_saver()

        checkpoint_path = None
        if os.path.exists(checkpoint_dir):
            if step is None:
                ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    checkpoint_path = ckpt.model_checkpoint_path
            else:
                checkpoint_path = os.path.join(checkpoint_dir, '{}-{}'.format(
                    self.name, step))

        if checkpoint_path is not None and tf.train.checkpoint_exists(
                checkpoint_path):
            tf.logging.info('Reading checkpoints...')
            self.saver.restore(self.sess, checkpoint_path)
            ckpt_name = os.path.basename(checkpoint_path)
            counter = int(
                re.search(r'(\d+)', ckpt_name.split('-')[-1]).group(1))
            tf.logging.info('Success to read %s', ckpt_name)
            return True, counter
        else:
            tf.logging.info('Failed to find a checkpoint')
            return False, 0

    def init_saver(self, saver=None):
        """Initialize saver object"""
        if self.saver is None or saver is not None:
            self.saver = saver or tf.train.Saver()


class GANModel(Model):
    """GANModel"""

    def __init__(self, sess, name, num_examples, output_shape, reg_const,
                 stddev, batch_size, image_summary):
        super().__init__(sess=sess, name=name)

        self.output_shape = output_shape
        self.batch_size = batch_size
        self.num_examples = num_examples

        self.image_summary = image_summary

        self.is_training = tf.placeholder_with_default(
            True, [], name='is_training')
        self.regularizer = tf.contrib.layers.l2_regularizer(
            scale=reg_const) if reg_const > 0.0 else None
        self.initializer = tf.truncated_normal_initializer(
            stddev=stddev
        ) if stddev is not None else tf.contrib.layers.xavier_initializer(
            uniform=False)
