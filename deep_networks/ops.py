# The ops are modified from TensorFlow ops.
#
# The following are the copyright notice from TensorFlow:
#
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""Tensor operations"""
import tensorflow as tf

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops


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
