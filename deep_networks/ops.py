"""Tensor operations"""
import tensorflow as tf

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops


# will be available in TF 1.4
def leaky_relu(features, alpha=0.2, name=None):
    """Compute the Leaky ReLU activation function.

    "Rectifier Nonlinearities Improve Neural Network Acoustic Models"
    AL Maas, AY Hannun, AY Ng - Proc. ICML, 2013
    http://web.stanford.edu/~awni/papers/relu_hybrid_icml2013_final.pdf

    Args:
      features: A `Tensor` representing preactivation values.
      alpha: Slope of the activation function at x < 0.
      name: A name for the operation (optional).

    Returns:
      The activation value.
    """
    with ops.name_scope(name, 'LeakyRelu', [features, alpha]):
        features = ops.convert_to_tensor(features, name='features')
        alpha = ops.convert_to_tensor(alpha, name='alpha')
        return math_ops.maximum(alpha * features, features)


# will be available in TF 1.4
def selu(features, name=None):
    """Scaled Exponential Linear Unit. (Klambauer et al., 2017).

    Arguments:
        features: A tensor or variable to compute the activation function for.

    Returns:
      Tensor with the same shape and dtype as `features`.

    References:
        - [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)
    """
    with ops.name_scope(name, 'ScaledElu', [features]):
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale * tf.where(features >= 0.0, features,
                                alpha * tf.nn.elu(features))


def conv2d_subpixel(inputs, scale=2, data_format='NHWC', name=None):
    """Sub-pixel convolution operation. (Shi et al., 2017).

    Arguments:
        inputs: A tensor or variable to compute the sub-pixel function for.

    Returns:
      Tensor with scale*height, scale*weight, and 1/scale**2 channels as `inputs`.

    References:
        - [Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network](https://arxiv.org/abs/1609.05158)
    """

    if data_format not in ('NCHW', 'NHWC'):
        raise ValueError('data_format has to be either NCHW or NHWC.')

    if data_format == 'NCHW':
        c_axis, h_axis, w_axis = 1, 2, 3
    else:
        c_axis, h_axis, w_axis = 3, 1, 2

    batch_size = array_ops.shape(inputs)[0]
    with ops.name_scope(name, 'Conv2d_subpixel', [inputs]):
        inputs = ops.convert_to_tensor(inputs)

        inputs_shape = inputs.get_shape()
        if int(inputs_shape[c_axis]) / (scale**2) % 1 != 0:
            raise ValueError(
                'The number of input channels == (scale x scale) x The number of output channels'
            )

        num_outputs = int(inputs_shape[c_axis]) // (scale**2)

        outputs = tf.split(inputs, scale, c_axis)
        outputs = tf.concat(outputs, w_axis)
        outputs_shape = [batch_size, 0, 0, 0]
        outputs_shape[c_axis] = num_outputs
        outputs_shape[h_axis] = scale * int(inputs_shape[h_axis])
        outputs_shape[w_axis] = scale * int(inputs_shape[w_axis])
        outputs = tf.reshape(outputs, outputs_shape)
    return outputs
