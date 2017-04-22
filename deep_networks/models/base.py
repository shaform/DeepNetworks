"""Base for models"""
import os

import tensorflow as tf

from tqdm import trange


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
