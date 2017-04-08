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
