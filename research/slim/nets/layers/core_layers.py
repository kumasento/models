""" Core classes for building TensorFlow layers.

This file is inspired by tensorflow/contrib/model_pruning/python/layers/core_layers.py


Author: Ruizhe Zhao <vincentzhaorz@gmail.com>
Date: 27/10/2018
"""

from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base
from tensorflow.python.layers import utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import standard_ops

MASK_COLLECTION = 'masks'
THRESHOLD_COLLECTION = 'thresholds'
MASKED_WEIGHT_COLLECTION = 'masked_weights'
WEIGHT_COLLECTION = 'kernel'
# The 'weights' part of the name is needed for the quantization library
# to recognize that the kernel should be quantized.
MASKED_WEIGHT_NAME = 'weights/masked_weight'


class MaskedConv2D(base.Layer):
  """ Masked conv2d layer """

  def __init__(self,
               filters,
               kernel_size,
               strides=1,
               padding='valid',
               data_format='channels_last',
               dilation_rate=1,
               activation=None,
               use_bias=True,
               kernel_initializer=None,
               bias_initializer=init_ops.zeros_initializer(),
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               trainable=True,
               name=None,
               **kwargs):
    super().__init__(
        trainable=trainable,
        name=name,
        activity_regularizer=activity_regularizer,
        **kwargs)
    self.ndim = 4
    self.filters = filters
    self.kernel_size = utils.normalize_tuple(kernel_size, rank, 'kernel_size')
    self.strides = utils.normalize_tuple(strides, rank, 'strides')
    self.padding = utils.normalize_padding(padding)
    self.data_format = utils.normalize_data_format(data_format)
    self.dilation_rate = utils.normalize_tuple(dilation_rate, rank,
                                               'dilation_rate')
    self.activation = activation
    self.use_bias = use_bias
    self.kernel_initializer = kernel_initializer
    self.bias_initializer = bias_initializer
    self.kernel_regularizer = kernel_regularizer
    self.bias_regularizer = bias_regularizer
    self.input_spec = base.InputSpec(ndim=self.ndim)

  def build(self, input_shape):
    """ Build the layer by a given input shape. """
    input_shape = tensor_shape.TensorShape(input_shape)
    channel_axis = 1 if self.data_format == 'channels_first' else -1
    if input_shape[channel_axis].value is None:
      raise ValueError('The channel dimension of the inputs '
                       'should be defined. Found `None`.')
      input_dim = input_shape[channel_axis].value
      kernel_shape = self.kernel_size + (input_dim, self.filters)
      self.mask = self.add_variable(
          name='mask',
          shape=kernel_shape,
          initializer=init_ops.ones_initializer(),
          trainable=False,
          dtype=self.dtype)

      self.kernel = self.add_variable(
          name='kernel',
          shape=kernel_shape,
          initializer=self.kernel_initializer,
          regularizer=self.kernel_regularizer,
          trainable=True,
          dtype=self.dtype)

      self.threshold = self.add_variable(
          name='threshold',
          shape=[],
          initializer=init_ops.zeros_initializer(),
          trainable=False,
          dtype=self.dtype)

      # Add masked_weights in the weights namescope so as to make it easier
      # for the quantization library to add quant ops.
      self.masked_kernel = math_ops.multiply(self.mask, self.kernel,
                                             MASKED_WEIGHT_NAME)

      ops.add_to_collection(MASK_COLLECTION, self.mask)
      ops.add_to_collection(MASKED_WEIGHT_COLLECTION, self.masked_kernel)
      ops.add_to_collection(THRESHOLD_COLLECTION, self.threshold)
      ops.add_to_collection(WEIGHT_COLLECTION, self.kernel)

      if self.use_bias:
        self.bias = self.add_variable(
            name='bias',
            shape=(self.filters,),
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            trainable=True,
            dtype=self.dtype)
      else:
        self.bias = None
        self.input_spec = base.InputSpec(
            ndim=self.rank + 2, axes={channel_axis: input_dim})
        self.built = True

  def call(self, inputs):
    """ Call the computation of this layer """
    outputs = nn.convolution(
        input=inputs,
        filter=self.masked_kernel,
        dilation_rate=self.dilation_rate,
        strides=self.strides,
        padding=self.padding.upper(),
        data_format=utils.convert_data_format(self.data_format, self.rank + 2))

    if self.bias is not None:
      if self.data_format == 'channels_first':
        outputs = nn.bias_add(
            outputs, self.bias, data_format='NHWC')
      if self.activation is not None:
        return self.activation(outputs)
      return outputs

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(
        input_shape).as_list()
    if self.data_format == 'channels_last':
      space = input_shape[1:-1]
      new_space = []
      for i in range(len(space)):
        new_dim = utils.conv_output_length(
            space[i],
            self.kernel_size[i],
            padding=self.padding,
            stride=self.strides[i],
            dilation=self.dilation_rate[i])
        new_space.append(new_dim)
      return tensor_shape.TensorShape([input_shape[0]] + new_space +
                                      [self.filters])
    else:
      space = input_shape[2:]
      new_space = []
      for i in range(len(space)):
        new_dim = utils.conv_output_length(
            space[i],
            self.kernel_size[i],
            padding=self.padding,
            stride=self.strides[i],
            dilation=self.dilation_rate[i])
        new_space.append(
            new_dim)
      return tensor_shape.TensorShape([input_shape[0], self.filters] +
                                      new_space)
