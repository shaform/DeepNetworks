import os

import tensorflow as tf

from tqdm import trange


class Model(object):
    def __init__(self, sess, name):
        self.name = name
        self.sess = sess
        self.saver = None
        self._trange = trange

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.name)
        os.makedirs(checkpoint_dir, exist_ok=True)

        ckpt_name = self.saver.save(
            self.sess,
            os.path.join(checkpoint_dir, self.name),
            global_step=step)
        tf.logging.debug('Saved checkpoint {}'.format(ckpt_name))

    def load(self, checkpoint_dir):
        import re
        checkpoint_dir = os.path.join(checkpoint_dir, self.name)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            tf.logging.info('Reading checkpoints...')
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess,
                               os.path.join(checkpoint_dir, ckpt_name))
            counter = int(re.search(r'-(\d+)', ckpt_name).group(1))
            tf.logging.info('Success to read {}'.format(ckpt_name))
            return True, counter
        else:
            tf.logging.info('Failed to find a checkpoint')
            return False, 0
