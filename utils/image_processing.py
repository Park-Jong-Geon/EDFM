from typing import List, Tuple, Any, Optional, Union
from abc import ABCMeta, abstractmethod
import inspect
import math

import jax
import jax.numpy as jnp
from jax import Array

import numpy as np
import tensorflow as tf

def split_channels(
        image: Any,
        channel_axis: int,
    ) -> Tuple[Array, Array, Array]:
    """Splits an image into its channels."""
    split_axes = jnp.split(image, 3, axis=channel_axis)
    return tuple(map(lambda e: jnp.squeeze(e, axis=channel_axis), split_axes))


def rgb_planes_to_hsv_planes(
        r: Any, g: Any, b: Any
    ) -> Tuple[Array, Array, Array]:
    """Converts RGB color planes to HSV planes."""
    v = jnp.maximum(jnp.maximum(r, g), b)
    c = v - jnp.minimum(jnp.minimum(r, g), b)
    safe_v = jnp.where(v > 0., v, 1.)
    safe_c = jnp.where(c > 0., c, 1.)

    s = jnp.where(v > 0., c / safe_v, 0.)

    rc = (v - r) / safe_c
    gc = (v - g) / safe_c
    bc = (v - b) / safe_c

    h = jnp.where(r == v, bc - gc, 0.)
    h = jnp.where(g == v, 2. + rc - bc, h)
    h = jnp.where(b == v, 4. + gc - rc, h)

    h = (h / 6.) % 1.
    h = jnp.where(c == 0., 0., h)

    return h, s, v


def hsv_planes_to_rgb_planes(
        h: Any, s: Any, v: Any
    ) -> Tuple[Array, Array, Array]:
    """Converts HSV color planes to RGB planes."""
    h = h % 1.0
    dh = h * 6.0

    dr = jnp.clip(jnp.abs(dh - 3.) - 1., 0., 1.)
    dg = jnp.clip(2. - jnp.abs(dh - 2.), 0., 1.)
    db = jnp.clip(2. - jnp.abs(dh - 4.), 0., 1.)

    ms = 1. - s
    r = v * (ms + s * dr)
    g = v * (ms + s * dg)
    b = v * (ms + s * db)

    return r, g, b


def rgb_to_hsv(
        image: Any,
        *,
        channel_axis: int = -1,
    ) -> Array:
    """Converts an image from RGB to HSV."""
    r, g, b = split_channels(image, channel_axis)
    return jnp.stack(rgb_planes_to_hsv_planes(r, g, b), axis=channel_axis)


def hsv_to_rgb(
        image: Any,
        *,
        channel_axis: int = -1,
    ) -> Array:
    """Converts an image from HSV to RGB."""
    h, s, v = split_channels(image, channel_axis)
    return jnp.stack(hsv_planes_to_rgb_planes(h, s, v), axis=channel_axis)


def rgb_to_grayscale(
        image: Any,
        *,
        keepdims: bool = True,
        luma_standard='rec601',
        channel_axis: int = -1,
    ) -> Array:
    """Converts an image to a grayscale image."""
    if luma_standard == 'rec601':
        weight = jnp.array([0.2989, 0.5870, 0.1140], dtype=image.dtype)
    elif luma_standard == 'rec709':
        weight = jnp.array([0.2126, 0.7152, 0.0722], dtype=image.dtype)
    elif luma_standard == 'bt2001':
        weight = jnp.array([0.2627, 0.6780, 0.0593], dtype=image.dtype)
    else:
        raise NotImplementedError(f'Unknown luma_standard={luma_standard}')
    grayscale = jnp.tensordot(image, weight, axes=(channel_axis, -1))
    grayscale = jnp.expand_dims(grayscale, axis=channel_axis)
    if keepdims:
        if channel_axis < 0:
            channel_axis += image.ndim
        return jnp.tile(grayscale, [
            (1 if axis != channel_axis else 3) for axis in range(image.ndim)])
    return grayscale


class Transform(metaclass=ABCMeta):
    # pylint: disable=too-few-public-methods
    """Base class for transformations."""

    @abstractmethod
    def __call__(self, rng, image):
        """Apply the transform on an image."""


class TransformChain(Transform):
    # pylint: disable=too-few-public-methods
    """Chain multiple transformations."""

    def __init__(self, transforms: List[Transform], prob: float = 1.0):
        """Apply transforms with the given probability."""
        self.transforms = transforms
        self.prob = prob

    def __call__(self, rng, image):
        jmage = image
        _rngs = jax.random.split(rng, len(self.transforms))
        for _transform, _rng in zip(self.transforms, _rngs):
            jmage = _transform(_rng, jmage)
        return jnp.where(jax.random.bernoulli(rng, self.prob), jmage, image)


class RandomHFlipTransform(Transform):
    # pylint: disable=too-few-public-methods
    """RandomHFlipTransform"""

    def __init__(self, prob=0.5):
        """Flips an image horizontally with the given probability."""
        self.prob = prob

    def __call__(self, rng, image):
        jmage = jnp.flip(image, axis=1)
        return jnp.where(
            jax.random.bernoulli(rng, self.prob), jmage, image)


class RandomCropTransform(Transform):
    # pylint: disable=too-few-public-methods
    """RandomCropTransform"""

    def __init__(self, size, padding):
        """Crops an image with the given size and padding."""
        self.size = size
        self.padding = padding

    def __call__(self, rng, image):
        rngs = jax.random.split(rng, 2)
        pad_width = (
            (self.padding, self.padding), (self.padding, self.padding), (0, 0))
        image = jnp.pad(
            image, pad_width=pad_width, mode='constant', constant_values=0)
        h0 = jax.random.randint(
            rngs[0], shape=(1,), minval=0, maxval=2*self.padding+1)[0]
        w0 = jax.random.randint(
            rngs[1], shape=(1,), minval=0, maxval=2*self.padding+1)[0]
        image = jax.lax.dynamic_slice(
            image, start_indices=(h0, w0, 0),
            slice_sizes=(self.size, self.size, image.shape[2]))
        return image


class RandomGrayscaleTransform(Transform):
    # pylint: disable=too-few-public-methods
    """RandomGrayscaleTransform"""

    def __init__(self, prob=0.5):
        """Grayscales an image with the given probability."""
        self.prob = prob

    def __call__(self, rng, image):
        jmage = rgb_to_grayscale(image, keepdims=True)
        jmage = jnp.clip(jmage, 0., 1.).astype(image.dtype)
        return jnp.where(
            jax.random.bernoulli(rng, self.prob), jmage, image)


class RandomBrightnessTransform(Transform):
    # pylint: disable=too-few-public-methods
    """RandomBrightnessTransform"""

    def __init__(self, lower=0.5, upper=1.5):
        """Changes a brightness of an image."""
        self.lower = lower
        self.upper = upper

    def __call__(self, rng, image):
        alpha = jax.random.uniform(
            rng, shape=(1,), minval=self.lower, maxval=self.upper)
        jmage = (image * alpha).astype(image.dtype)
        return jnp.clip(jmage, 0., 1.)


class RandomContrastTransform(Transform):
    # pylint: disable=too-few-public-methods
    """RandomContrastTransform"""

    def __init__(self, lower=0.5, upper=1.5):
        """Changes a contrast of an image."""
        self.lower = lower
        self.upper = upper

    def __call__(self, rng, image):
        alpha = jax.random.uniform(
            rng, shape=(1,), minval=self.lower, maxval=self.upper)
        _mean = jnp.mean(image, axis=(0, 1), keepdims=True)
        jmage = (_mean + (image - _mean) * alpha).astype(image.dtype)
        return jnp.clip(jmage, 0., 1.)


class RandomSaturationTransform(Transform):
    # pylint: disable=too-few-public-methods
    """RandomSaturationTransform"""

    def __init__(self, lower=0.5, upper=1.5):
        """Changes a saturation of an image."""
        self.lower = lower
        self.upper = upper

    def __call__(self, rng, image):
        alpha = jax.random.uniform(
            rng, shape=(1,), minval=self.lower, maxval=self.upper)
        r, g, b = split_channels(image, channel_axis=2)
        h, s, v = rgb_planes_to_hsv_planes(r, g, b)
        jmage = hsv_planes_to_rgb_planes(h, jnp.clip(s * alpha, 0., 1.), v)
        jmage = jnp.stack(jmage, axis=2).astype(image.dtype)
        return jnp.clip(jmage, 0., 1.)


class RandomHueTransform(Transform):
    # pylint: disable=too-few-public-methods
    """RandomHueTransform"""

    def __init__(self, delta=0.5):
        """Changes a hue of an image."""
        self.delta = delta

    def __call__(self, rng, image):
        alpha = jax.random.uniform(
            rng, shape=(1,), minval=-self.delta, maxval=self.delta)
        r, g, b = split_channels(image, channel_axis=2)
        h, s, v = rgb_planes_to_hsv_planes(r, g, b)
        jmage = hsv_planes_to_rgb_planes((h + alpha) % 1.0, s, v)
        jmage = jnp.stack(jmage, axis=2).astype(image.dtype)
        return jnp.clip(jmage, 0., 1.)
      
# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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

"""Augmentation policies for enhanced image/video preprocessing.

RandAugment Reference: https://arxiv.org/abs/1909.13719
"""

# This signifies the max integer that the controller RNN could predict for the
# augmentation scheme.
_MAX_LEVEL = 10.


def to_4d(image: tf.Tensor) -> tf.Tensor:
  """Converts an input Tensor to 4 dimensions.

  4D image => [N, H, W, C] or [N, C, H, W]
  3D image => [1, H, W, C] or [1, C, H, W]
  2D image => [1, H, W, 1]

  Args:
    image: The 2/3/4D input tensor.

  Returns:
    A 4D image tensor.

  Raises:
    `TypeError` if `image` is not a 2/3/4D tensor.

  """
  shape = tf.shape(image)
  original_rank = tf.rank(image)
  left_pad = tf.cast(tf.less_equal(original_rank, 3), dtype=tf.int32)
  right_pad = tf.cast(tf.equal(original_rank, 2), dtype=tf.int32)
  new_shape = tf.concat(
      [
          tf.ones(shape=left_pad, dtype=tf.int32),
          shape,
          tf.ones(shape=right_pad, dtype=tf.int32),
      ],
      axis=0,
  )
  return tf.reshape(image, new_shape)


def from_4d(image: tf.Tensor, ndims: tf.Tensor) -> tf.Tensor:
  """Converts a 4D image back to `ndims` rank."""
  shape = tf.shape(image)
  begin = tf.cast(tf.less_equal(ndims, 3), dtype=tf.int32)
  end = 4 - tf.cast(tf.equal(ndims, 2), dtype=tf.int32)
  new_shape = shape[begin:end]
  return tf.reshape(image, new_shape)


def _pad(
    image: tf.Tensor,
    filter_shape: Union[List[int], Tuple[int, ...]],
    mode: str = 'CONSTANT',
    constant_values: Union[int, tf.Tensor] = 0,
) -> tf.Tensor:
  """Explicitly pads a 4-D image.

  Equivalent to the implicit padding method offered in `tf.nn.conv2d` and
  `tf.nn.depthwise_conv2d`, but supports non-zero, reflect and symmetric
  padding mode. For the even-sized filter, it pads one more value to the
  right or the bottom side.

  Args:
    image: A 4-D `Tensor` of shape `[batch_size, height, width, channels]`.
    filter_shape: A `tuple`/`list` of 2 integers, specifying the height and
      width of the 2-D filter.
    mode: A `string`, one of "REFLECT", "CONSTANT", or "SYMMETRIC". The type of
      padding algorithm to use, which is compatible with `mode` argument in
      `tf.pad`. For more details, please refer to
      https://www.tensorflow.org/api_docs/python/tf/pad.
    constant_values: A `scalar`, the pad value to use in "CONSTANT" padding
      mode.

  Returns:
    A padded image.
  """
  if mode.upper() not in {'REFLECT', 'CONSTANT', 'SYMMETRIC'}:
    raise ValueError(
        'padding should be one of "REFLECT", "CONSTANT", or "SYMMETRIC".'
    )
  constant_values = tf.convert_to_tensor(constant_values, image.dtype)
  filter_height, filter_width = filter_shape
  pad_top = (filter_height - 1) // 2
  pad_bottom = filter_height - 1 - pad_top
  pad_left = (filter_width - 1) // 2
  pad_right = filter_width - 1 - pad_left
  paddings = [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]]
  return tf.pad(image, paddings, mode=mode, constant_values=constant_values)


def _get_gaussian_kernel(sigma, filter_shape):
  """Computes 1D Gaussian kernel."""
  sigma = tf.convert_to_tensor(sigma)
  x = tf.range(-filter_shape // 2 + 1, filter_shape // 2 + 1)
  x = tf.cast(x**2, sigma.dtype)
  x = tf.nn.softmax(-x / (2.0 * (sigma**2)))
  return x


def _get_gaussian_kernel_2d(gaussian_filter_x, gaussian_filter_y):
  """Computes 2D Gaussian kernel given 1D kernels."""
  gaussian_kernel = tf.matmul(gaussian_filter_x, gaussian_filter_y)
  return gaussian_kernel


def _normalize_tuple(value, n, name):
  """Transforms an integer or iterable of integers into an integer tuple.

  Args:
    value: The value to validate and convert. Could an int, or any iterable of
      ints.
    n: The size of the tuple to be returned.
    name: The name of the argument being validated, e.g. "strides" or
      "kernel_size". This is only used to format error messages.

  Returns:
    A tuple of n integers.

  Raises:
    ValueError: If something else than an int/long or iterable thereof was
      passed.
  """
  if isinstance(value, int):
    return (value,) * n
  else:
    try:
      value_tuple = tuple(value)
    except TypeError as exc:
      raise TypeError(
          f'The {name} argument must be a tuple of {n} integers. '
          f'Received: {value}'
      ) from exc
    if len(value_tuple) != n:
      raise ValueError(
          f'The {name} argument must be a tuple of {n} integers. '
          f'Received: {value}'
      )
    for single_value in value_tuple:
      try:
        int(single_value)
      except (ValueError, TypeError) as exc:
        raise ValueError(
            f'The {name} argument must be a tuple of {n} integers. Received:'
            f' {value} including element {single_value} of type'
            f' {type(single_value)}.'
        ) from exc
    return value_tuple


def gaussian_filter2d(
    image: tf.Tensor,
    filter_shape: Union[List[int], Tuple[int, ...], int],
    sigma: Union[List[float], Tuple[float, float], float] = 1.0,
    padding: str = 'REFLECT',
    constant_values: Union[int, tf.Tensor] = 0,
    name: Optional[str] = None,
) -> tf.Tensor:
  """Performs Gaussian blur on image(s).

  Args:
    image: Either a 2-D `Tensor` of shape `[height, width]`, a 3-D `Tensor` of
      shape `[height, width, channels]`, or a 4-D `Tensor` of shape
      `[batch_size, height, width, channels]`.
    filter_shape: An `integer` or `tuple`/`list` of 2 integers, specifying the
      height and width of the 2-D gaussian filter. Can be a single integer to
      specify the same value for all spatial dimensions.
    sigma: A `float` or `tuple`/`list` of 2 floats, specifying the standard
      deviation in x and y direction the 2-D gaussian filter. Can be a single
      float to specify the same value for all spatial dimensions.
    padding: A `string`, one of "REFLECT", "CONSTANT", or "SYMMETRIC". The type
      of padding algorithm to use, which is compatible with `mode` argument in
      `tf.pad`. For more details, please refer to
      https://www.tensorflow.org/api_docs/python/tf/pad.
    constant_values: A `scalar`, the pad value to use in "CONSTANT" padding
      mode.
    name: A name for this operation (optional).

  Returns:
    2-D, 3-D or 4-D `Tensor` of the same dtype as input.

  Raises:
    ValueError: If `image` is not 2, 3 or 4-dimensional,
      if `padding` is other than "REFLECT", "CONSTANT" or "SYMMETRIC",
      if `filter_shape` is invalid,
      or if `sigma` is invalid.
  """
  with tf.name_scope(name or 'gaussian_filter2d'):
    if isinstance(sigma, (list, tuple)):
      if len(sigma) != 2:
        raise ValueError('sigma should be a float or a tuple/list of 2 floats')
    else:
      sigma = (sigma,) * 2

    if any(s < 0 for s in sigma):
      raise ValueError('sigma should be greater than or equal to 0.')

    image = tf.convert_to_tensor(image, name='image')
    sigma = tf.convert_to_tensor(sigma, name='sigma')

    original_ndims = tf.rank(image)
    image = to_4d(image)

    # Keep the precision if it's float;
    # otherwise, convert to float32 for computing.
    orig_dtype = image.dtype
    if not image.dtype.is_floating:
      image = tf.cast(image, tf.float32)

    channels = tf.shape(image)[3]
    filter_shape = _normalize_tuple(filter_shape, 2, 'filter_shape')

    sigma = tf.cast(sigma, image.dtype)
    gaussian_kernel_x = _get_gaussian_kernel(sigma[1], filter_shape[1])
    gaussian_kernel_x = gaussian_kernel_x[tf.newaxis, :]

    gaussian_kernel_y = _get_gaussian_kernel(sigma[0], filter_shape[0])
    gaussian_kernel_y = gaussian_kernel_y[:, tf.newaxis]

    gaussian_kernel_2d = _get_gaussian_kernel_2d(
        gaussian_kernel_y, gaussian_kernel_x
    )
    gaussian_kernel_2d = gaussian_kernel_2d[:, :, tf.newaxis, tf.newaxis]
    gaussian_kernel_2d = tf.tile(gaussian_kernel_2d, [1, 1, channels, 1])

    image = _pad(
        image, filter_shape, mode=padding, constant_values=constant_values
    )

    output = tf.nn.depthwise_conv2d(
        input=image,
        filter=gaussian_kernel_2d,
        strides=(1, 1, 1, 1),
        padding='VALID',
    )
    output = from_4d(output, original_ndims)
    return tf.cast(output, orig_dtype)


def _convert_translation_to_transform(translations: tf.Tensor) -> tf.Tensor:
  """Converts translations to a projective transform.

  The translation matrix looks like this:
    [[1 0 -dx]
     [0 1 -dy]
     [0 0 1]]

  Args:
    translations: The 2-element list representing [dx, dy], or a matrix of
      2-element lists representing [dx dy] to translate for each image. The
      shape must be static.

  Returns:
    The transformation matrix of shape (num_images, 8).

  Raises:
    `TypeError` if
      - the shape of `translations` is not known or
      - the shape of `translations` is not rank 1 or 2.

  """
  translations = tf.convert_to_tensor(translations, dtype=tf.float32)
  if translations.get_shape().ndims is None:
    raise TypeError('translations rank must be statically known')
  elif len(translations.get_shape()) == 1:
    translations = translations[None]
  elif len(translations.get_shape()) != 2:
    raise TypeError('translations should have rank 1 or 2.')
  num_translations = tf.shape(translations)[0]

  return tf.concat(
      values=[
          tf.ones((num_translations, 1), tf.dtypes.float32),
          tf.zeros((num_translations, 1), tf.dtypes.float32),
          -translations[:, 0, None],
          tf.zeros((num_translations, 1), tf.dtypes.float32),
          tf.ones((num_translations, 1), tf.dtypes.float32),
          -translations[:, 1, None],
          tf.zeros((num_translations, 2), tf.dtypes.float32),
      ],
      axis=1,
  )


def _convert_angles_to_transform(angles: tf.Tensor, image_width: tf.Tensor,
                                 image_height: tf.Tensor) -> tf.Tensor:
  """Converts an angle or angles to a projective transform.

  Args:
    angles: A scalar to rotate all images, or a vector to rotate a batch of
      images. This must be a scalar.
    image_width: The width of the image(s) to be transformed.
    image_height: The height of the image(s) to be transformed.

  Returns:
    A tensor of shape (num_images, 8).

  Raises:
    `TypeError` if `angles` is not rank 0 or 1.

  """
  angles = tf.convert_to_tensor(angles, dtype=tf.float32)
  if len(angles.get_shape()) == 0:  # pylint:disable=g-explicit-length-test
    angles = angles[None]
  elif len(angles.get_shape()) != 1:
    raise TypeError('Angles should have a rank 0 or 1.')
  x_offset = ((image_width - 1) -
              (tf.math.cos(angles) * (image_width - 1) - tf.math.sin(angles) *
               (image_height - 1))) / 2.0
  y_offset = ((image_height - 1) -
              (tf.math.sin(angles) * (image_width - 1) + tf.math.cos(angles) *
               (image_height - 1))) / 2.0
  num_angles = tf.shape(angles)[0]
  return tf.concat(
      values=[
          tf.math.cos(angles)[:, None],
          -tf.math.sin(angles)[:, None],
          x_offset[:, None],
          tf.math.sin(angles)[:, None],
          tf.math.cos(angles)[:, None],
          y_offset[:, None],
          tf.zeros((num_angles, 2), tf.dtypes.float32),
      ],
      axis=1,
  )


def _apply_transform_to_images(
    images,
    transforms,
    fill_mode='reflect',
    fill_value=0.0,
    interpolation='bilinear',
    output_shape=None,
    name=None,
):
  """Applies the given transform(s) to the image(s).

  Args:
    images: A tensor of shape `(num_images, num_rows, num_columns,
      num_channels)` (NHWC). The rank must be statically known (the shape is
      not `TensorShape(None)`).
    transforms: Projective transform matrix/matrices. A vector of length 8 or
      tensor of size N x 8. If one row of transforms is [a0, a1, a2, b0, b1,
      b2, c0, c1], then it maps the *output* point `(x, y)` to a transformed
      *input* point `(x', y') = ((a0 x + a1 y + a2) / k, (b0 x + b1 y + b2) /
      k)`, where `k = c0 x + c1 y + 1`. The transforms are *inverted* compared
      to the transform mapping input points to output points. Note that
      gradients are not backpropagated into transformation parameters.
    fill_mode: Points outside the boundaries of the input are filled according
      to the given mode (one of `{"constant", "reflect", "wrap", "nearest"}`).
    fill_value: a float represents the value to be filled outside the
      boundaries when `fill_mode="constant"`.
    interpolation: Interpolation mode. Supported values: `"nearest"`,
      `"bilinear"`.
    output_shape: Output dimension after the transform, `[height, width]`. If
      `None`, output is the same size as input image.
    name: The name of the op.  Fill mode behavior for each valid value is as
      follows
      - `"reflect"`: `(d c b a | a b c d | d c b a)` The input is extended by
      reflecting about the edge of the last pixel.
      - `"constant"`: `(k k k k | a b c d | k k k k)` The input is extended by
      filling all values beyond the edge with the same constant value k = 0.
      - `"wrap"`: `(a b c d | a b c d | a b c d)` The input is extended by
      wrapping around to the opposite edge.
      - `"nearest"`: `(a a a a | a b c d | d d d d)` The input is extended by
      the nearest pixel.  Input shape: 4D tensor with shape:
      `(samples, height, width, channels)`, in `"channels_last"` format.
      Output shape: 4D tensor with shape: `(samples, height, width, channels)`,
      in `"channels_last"` format.

  Returns:
    Image(s) with the same type and shape as `images`, with the given
    transform(s) applied. Transformed coordinates outside of the input image
    will be filled with zeros.
  """
  with tf.name_scope(name or 'transform'):
    if output_shape is None:
      output_shape = tf.shape(images)[1:3]
      if not tf.executing_eagerly():
        output_shape_value = tf.get_static_value(output_shape)
        if output_shape_value is not None:
          output_shape = output_shape_value

    output_shape = tf.convert_to_tensor(
        output_shape, tf.int32, name='output_shape'
    )

    if not output_shape.get_shape().is_compatible_with([2]):
      raise ValueError(
          'output_shape must be a 1-D Tensor of 2 elements: '
          'new_height, new_width, instead got '
          f'output_shape={output_shape}'
      )

    fill_value = tf.convert_to_tensor(fill_value, tf.float32, name='fill_value')

    return tf.raw_ops.ImageProjectiveTransformV3(
        images=images,
        output_shape=output_shape,
        fill_value=fill_value,
        transforms=transforms,
        fill_mode=fill_mode.upper(),
        interpolation=interpolation.upper(),
    )


def transform(
    image: tf.Tensor,
    transforms: Any,
    interpolation: str = 'nearest',
    output_shape=None,
    fill_mode: str = 'reflect',
    fill_value: float = 0.0,
) -> tf.Tensor:
  """Transforms an image."""
  original_ndims = tf.rank(image)
  transforms = tf.convert_to_tensor(transforms, dtype=tf.float32)
  if transforms.shape.rank == 1:
    transforms = transforms[None]
  image = to_4d(image)
  image = _apply_transform_to_images(
      images=image,
      transforms=transforms,
      interpolation=interpolation,
      fill_mode=fill_mode,
      fill_value=fill_value,
      output_shape=output_shape,
  )
  return from_4d(image, original_ndims)


def translate(
    image: tf.Tensor,
    translations,
    fill_value: float = 0.0,
    fill_mode: str = 'reflect',
    interpolation: str = 'nearest',
) -> tf.Tensor:
  """Translates image(s) by provided vectors.

  Args:
    image: An image Tensor of type uint8.
    translations: A vector or matrix representing [dx dy].
    fill_value: a float represents the value to be filled outside the boundaries
      when `fill_mode="constant"`.
    fill_mode: Points outside the boundaries of the input are filled according
      to the given mode (one of `{"constant", "reflect", "wrap", "nearest"}`).
    interpolation: Interpolation mode. Supported values: `"nearest"`,
      `"bilinear"`.

  Returns:
    The translated version of the image.
  """
  transforms = _convert_translation_to_transform(translations)  # pytype: disable=wrong-arg-types  # always-use-return-annotations
  return transform(
      image,
      transforms=transforms,
      interpolation=interpolation,
      fill_value=fill_value,
      fill_mode=fill_mode,
  )


def rotate(image: tf.Tensor, degrees: float) -> tf.Tensor:
  """Rotates the image by degrees either clockwise or counterclockwise.

  Args:
    image: An image Tensor of type uint8.
    degrees: Float, a scalar angle in degrees to rotate all images by. If
      degrees is positive the image will be rotated clockwise otherwise it will
      be rotated counterclockwise.

  Returns:
    The rotated version of image.

  """
  # Convert from degrees to radians.
  degrees_to_radians = math.pi / 180.0
  radians = tf.cast(degrees * degrees_to_radians, tf.float32)

  original_ndims = tf.rank(image)
  image = to_4d(image)

  image_height = tf.cast(tf.shape(image)[1], tf.float32)
  image_width = tf.cast(tf.shape(image)[2], tf.float32)
  transforms = _convert_angles_to_transform(
      angles=radians, image_width=image_width, image_height=image_height)
  # In practice, we should randomize the rotation degrees by flipping
  # it negatively half the time, but that's done on 'degrees' outside
  # of the function.
  image = transform(image, transforms=transforms)
  return from_4d(image, original_ndims)


def blend(image1: tf.Tensor, image2: tf.Tensor, factor: float) -> tf.Tensor:
  """Blend image1 and image2 using 'factor'.

  Factor can be above 0.0.  A value of 0.0 means only image1 is used.
  A value of 1.0 means only image2 is used.  A value between 0.0 and
  1.0 means we linearly interpolate the pixel values between the two
  images.  A value greater than 1.0 "extrapolates" the difference
  between the two pixel values, and we clip the results to values
  between 0 and 255.

  Args:
    image1: An image Tensor of type uint8.
    image2: An image Tensor of type uint8.
    factor: A floating point value above 0.0.

  Returns:
    A blended image Tensor of type uint8.
  """
  if factor == 0.0:
    return tf.convert_to_tensor(image1)
  if factor == 1.0:
    return tf.convert_to_tensor(image2)

  image1 = tf.cast(image1, tf.float32)
  image2 = tf.cast(image2, tf.float32)

  difference = image2 - image1
  scaled = factor * difference

  # Do addition in float.
  temp = tf.cast(image1, tf.float32) + scaled

  # Interpolate
  if factor > 0.0 and factor < 1.0:
    # Interpolation means we always stay within 0 and 255.
    return tf.cast(temp, tf.uint8)

  # Extrapolate:
  #
  # We need to clip and then cast.
  return tf.cast(tf.clip_by_value(temp, 0.0, 255.0), tf.uint8)


def cutout(image: tf.Tensor, pad_size: int, replace: int = 0) -> tf.Tensor:
  """Apply cutout (https://arxiv.org/abs/1708.04552) to image.

  This operation applies a (2*pad_size x 2*pad_size) mask of zeros to
  a random location within `image`. The pixel values filled in will be of the
  value `replace`. The location where the mask will be applied is randomly
  chosen uniformly over the whole image.

  Args:
    image: An image Tensor of type uint8.
    pad_size: Specifies how big the zero mask that will be generated is that is
      applied to the image. The mask will be of size (2*pad_size x 2*pad_size).
    replace: What pixel value to fill in the image in the area that has the
      cutout mask applied to it.

  Returns:
    An image Tensor that is of type uint8.
  """
  if image.shape.rank not in [3, 4]:
    raise ValueError('Bad image rank: {}'.format(image.shape.rank))

  if image.shape.rank == 4:
    return cutout_video(image, replace=replace)

  image_height = tf.shape(image)[0]
  image_width = tf.shape(image)[1]

  # Sample the center location in the image where the zero mask will be applied.
  cutout_center_height = tf.random.uniform(
      shape=[], minval=0, maxval=image_height, dtype=tf.int32)

  cutout_center_width = tf.random.uniform(
      shape=[], minval=0, maxval=image_width, dtype=tf.int32)

  image = _fill_rectangle(image, cutout_center_width, cutout_center_height,
                          pad_size, pad_size, replace)

  return image


def _fill_rectangle(image,
                    center_width,
                    center_height,
                    half_width,
                    half_height,
                    replace=None):
  """Fills blank area."""
  image_height = tf.shape(image)[0]
  image_width = tf.shape(image)[1]

  lower_pad = tf.maximum(0, center_height - half_height)
  upper_pad = tf.maximum(0, image_height - center_height - half_height)
  left_pad = tf.maximum(0, center_width - half_width)
  right_pad = tf.maximum(0, image_width - center_width - half_width)

  cutout_shape = [
      image_height - (lower_pad + upper_pad),
      image_width - (left_pad + right_pad)
  ]
  padding_dims = [[lower_pad, upper_pad], [left_pad, right_pad]]
  mask = tf.pad(
      tf.zeros(cutout_shape, dtype=image.dtype),
      padding_dims,
      constant_values=1)
  mask = tf.expand_dims(mask, -1)
  mask = tf.tile(mask, [1, 1, 3])

  if replace is None:
    fill = tf.random.normal(tf.shape(image), dtype=image.dtype)
  elif isinstance(replace, tf.Tensor):
    fill = replace
  else:
    fill = tf.ones_like(image, dtype=image.dtype) * replace
  image = tf.where(tf.equal(mask, 0), fill, image)

  return image


def _fill_rectangle_video(image,
                          center_width,
                          center_height,
                          half_width,
                          half_height,
                          replace=None):
  """Fills blank area for video."""
  image_time = tf.shape(image)[0]
  image_height = tf.shape(image)[1]
  image_width = tf.shape(image)[2]
  image_channels = tf.shape(image)[3]

  lower_pad = tf.maximum(0, center_height - half_height)
  upper_pad = tf.maximum(0, image_height - center_height - half_height)
  left_pad = tf.maximum(0, center_width - half_width)
  right_pad = tf.maximum(0, image_width - center_width - half_width)

  cutout_shape = [
      image_time, image_height - (lower_pad + upper_pad),
      image_width - (left_pad + right_pad)
  ]
  padding_dims = [[0, 0], [lower_pad, upper_pad], [left_pad, right_pad]]
  mask = tf.pad(
      tf.zeros(cutout_shape, dtype=image.dtype),
      padding_dims,
      constant_values=1)
  mask = tf.expand_dims(mask, -1)
  mask = tf.tile(mask, [1, 1, 1, image_channels])

  if replace is None:
    fill = tf.random.normal(tf.shape(image), dtype=image.dtype)
  elif isinstance(replace, tf.Tensor):
    fill = replace
  else:
    fill = tf.ones_like(image, dtype=image.dtype) * replace
  image = tf.where(tf.equal(mask, 0), fill, image)

  return image


def cutout_video(
    video: tf.Tensor,
    mask_shape: Optional[tf.Tensor] = None,
    replace: int = 0,
) -> tf.Tensor:
  """Apply cutout (https://arxiv.org/abs/1708.04552) to a video.

  This operation applies a random size 3D mask of zeros to a random location
  within `video`. The mask is padded The pixel values filled in will be of the
  value `replace`. The location where the mask will be applied is randomly
  chosen uniformly over the whole video. If the size of the mask is not set,
  then, it is randomly sampled uniformly from [0.25*height, 0.5*height],
  [0.25*width, 0.5*width], and [1, 0.25*depth], which represent the height,
  width, and number of frames of the input video tensor respectively.

  Args:
    video: A video Tensor of shape [T, H, W, C].
    mask_shape: An optional integer tensor that specifies the depth, height and
      width of the mask to cut. If it is not set, the shape is randomly sampled
      as described above. The shape dimensions should be divisible by 2
      otherwise they will rounded down.
    replace: What pixel value to fill in the image in the area that has the
      cutout mask applied to it.

  Returns:
    A video Tensor with cutout applied.
  """
  tf.debugging.assert_shapes([
      (video, ('T', 'H', 'W', 'C')),
  ])

  video_depth = tf.shape(video)[0]
  video_height = tf.shape(video)[1]
  video_width = tf.shape(video)[2]

  # Sample the center location in the image where the zero mask will be applied.
  cutout_center_height = tf.random.uniform(
      shape=[], minval=0, maxval=video_height, dtype=tf.int32
  )

  cutout_center_width = tf.random.uniform(
      shape=[], minval=0, maxval=video_width, dtype=tf.int32
  )

  cutout_center_depth = tf.random.uniform(
      shape=[], minval=0, maxval=video_depth, dtype=tf.int32
  )

  if mask_shape is not None:
    pad_shape = tf.maximum(1, mask_shape // 2)
    pad_size_depth, pad_size_height, pad_size_width = (
        pad_shape[0],
        pad_shape[1],
        pad_shape[2],
    )
  else:
    pad_size_height = tf.random.uniform(
        shape=[],
        minval=tf.maximum(1, tf.cast(video_height / 4, tf.int32)),
        maxval=tf.maximum(2, tf.cast(video_height / 2, tf.int32)),
        dtype=tf.int32,
    )
    pad_size_width = tf.random.uniform(
        shape=[],
        minval=tf.maximum(1, tf.cast(video_width / 4, tf.int32)),
        maxval=tf.maximum(2, tf.cast(video_width / 2, tf.int32)),
        dtype=tf.int32,
    )
    pad_size_depth = tf.random.uniform(
        shape=[],
        minval=1,
        maxval=tf.maximum(2, tf.cast(video_depth / 4, tf.int32)),
        dtype=tf.int32,
    )

  lower_pad = tf.maximum(0, cutout_center_height - pad_size_height)
  upper_pad = tf.maximum(
      0, video_height - cutout_center_height - pad_size_height
  )
  left_pad = tf.maximum(0, cutout_center_width - pad_size_width)
  right_pad = tf.maximum(0, video_width - cutout_center_width - pad_size_width)
  back_pad = tf.maximum(0, cutout_center_depth - pad_size_depth)
  forward_pad = tf.maximum(
      0, video_depth - cutout_center_depth - pad_size_depth
  )

  cutout_shape = [
      video_depth - (back_pad + forward_pad),
      video_height - (lower_pad + upper_pad),
      video_width - (left_pad + right_pad),
  ]
  padding_dims = [[back_pad, forward_pad],
                  [lower_pad, upper_pad],
                  [left_pad, right_pad]]
  mask = tf.pad(
      tf.zeros(cutout_shape, dtype=video.dtype), padding_dims, constant_values=1
  )
  mask = tf.expand_dims(mask, -1)
  num_channels = tf.shape(video)[-1]
  mask = tf.tile(mask, [1, 1, 1, num_channels])
  video = tf.where(
      tf.equal(mask, 0), tf.ones_like(video, dtype=video.dtype) * replace, video
  )
  return video


def gaussian_noise(
    image: tf.Tensor, low: float = 0.1, high: float = 2.0) -> tf.Tensor:
  """Add Gaussian noise to image(s)."""
  augmented_image = gaussian_filter2d(  # pylint: disable=g-long-lambda
      image, filter_shape=[3, 3], sigma=np.random.uniform(low=low, high=high)
  )
  return augmented_image


def solarize(image: tf.Tensor, threshold: int = 128) -> tf.Tensor:
  """Solarize the input image(s)."""
  # For each pixel in the image, select the pixel
  # if the value is less than the threshold.
  # Otherwise, subtract 255 from the pixel.
  return tf.where(image < threshold, image, 255 - image)


def solarize_add(image: tf.Tensor,
                 addition: int = 0,
                 threshold: int = 128) -> tf.Tensor:
  """Additive solarize the input image(s)."""
  # For each pixel in the image less than threshold
  # we add 'addition' amount to it and then clip the
  # pixel value to be between 0 and 255. The value
  # of 'addition' is between -128 and 128.
  added_image = tf.cast(image, tf.int64) + addition
  added_image = tf.cast(tf.clip_by_value(added_image, 0, 255), tf.uint8)
  return tf.where(image < threshold, added_image, image)


def grayscale(image: tf.Tensor) -> tf.Tensor:
  """Convert image to grayscale."""
  return tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(image))


def color(image: tf.Tensor, factor: float) -> tf.Tensor:
  """Equivalent of PIL Color."""
  degenerate = grayscale(image)
  return blend(degenerate, image, factor)


def contrast(image: tf.Tensor, factor: float) -> tf.Tensor:
  """Equivalent of PIL Contrast."""
  degenerate = tf.image.rgb_to_grayscale(image)
  # Cast before calling tf.histogram.
  degenerate = tf.cast(degenerate, tf.int32)

  # Compute the grayscale histogram, then compute the mean pixel value,
  # and create a constant image size of that value.  Use that as the
  # blending degenerate target of the original image.
  hist = tf.histogram_fixed_width(degenerate, [0, 255], nbins=256)
  mean = tf.reduce_sum(tf.cast(hist, tf.float32)) / 256.0
  degenerate = tf.ones_like(degenerate, dtype=tf.float32) * mean
  degenerate = tf.clip_by_value(degenerate, 0.0, 255.0)
  degenerate = tf.image.grayscale_to_rgb(tf.cast(degenerate, tf.uint8))
  return blend(degenerate, image, factor)


def brightness(image: tf.Tensor, factor: float) -> tf.Tensor:
  """Equivalent of PIL Brightness."""
  degenerate = tf.zeros_like(image)
  return blend(degenerate, image, factor)


def posterize(image: tf.Tensor, bits: int) -> tf.Tensor:
  """Equivalent of PIL Posterize."""
  shift = 8 - bits
  return tf.bitwise.left_shift(tf.bitwise.right_shift(image, shift), shift)


def wrapped_rotate(image: tf.Tensor, degrees: float, replace: int) -> tf.Tensor:
  """Applies rotation with wrap/unwrap."""
  image = rotate(wrap(image), degrees=degrees)
  return unwrap(image, replace)


def translate_x(image: tf.Tensor, pixels: int, replace: int) -> tf.Tensor:
  """Equivalent of PIL Translate in X dimension."""
  image = translate(wrap(image), [-pixels, 0])
  return unwrap(image, replace)


def translate_y(image: tf.Tensor, pixels: int, replace: int) -> tf.Tensor:
  """Equivalent of PIL Translate in Y dimension."""
  image = translate(wrap(image), [0, -pixels])
  return unwrap(image, replace)


def shear_x(image: tf.Tensor, level: float, replace: int) -> tf.Tensor:
  """Equivalent of PIL Shearing in X dimension."""
  # Shear parallel to x axis is a projective transform
  # with a matrix form of:
  # [1  level
  #  0  1].
  image = transform(
      image=wrap(image), transforms=[1., level, 0., 0., 1., 0., 0., 0.])
  return unwrap(image, replace)


def shear_y(image: tf.Tensor, level: float, replace: int) -> tf.Tensor:
  """Equivalent of PIL Shearing in Y dimension."""
  # Shear parallel to y axis is a projective transform
  # with a matrix form of:
  # [1  0
  #  level  1].
  image = transform(
      image=wrap(image), transforms=[1., 0., 0., level, 1., 0., 0., 0.])
  return unwrap(image, replace)


def autocontrast(image: tf.Tensor) -> tf.Tensor:
  """Implements Autocontrast function from PIL using TF ops.

  Args:
    image: A 3D uint8 tensor.

  Returns:
    The image after it has had autocontrast applied to it and will be of type
    uint8.
  """

  def scale_channel(image: tf.Tensor) -> tf.Tensor:
    """Scale the 2D image using the autocontrast rule."""
    # A possibly cheaper version can be done using cumsum/unique_with_counts
    # over the histogram values, rather than iterating over the entire image.
    # to compute mins and maxes.
    lo = tf.cast(tf.reduce_min(image), tf.float32)
    hi = tf.cast(tf.reduce_max(image), tf.float32)

    # Scale the image, making the lowest value 0 and the highest value 255.
    def scale_values(im):
      scale = 255.0 / (hi - lo)
      offset = -lo * scale
      im = tf.cast(im, tf.float32) * scale + offset
      im = tf.clip_by_value(im, 0.0, 255.0)
      return tf.cast(im, tf.uint8)

    result = tf.cond(hi > lo, lambda: scale_values(image), lambda: image)
    return result

  # Assumes RGB for now.  Scales each channel independently
  # and then stacks the result.
  s1 = scale_channel(image[..., 0])
  s2 = scale_channel(image[..., 1])
  s3 = scale_channel(image[..., 2])
  image = tf.stack([s1, s2, s3], -1)

  return image


def sharpness(image: tf.Tensor, factor: float) -> tf.Tensor:
  """Implements Sharpness function from PIL using TF ops."""
  orig_image = image
  image = tf.cast(image, tf.float32)
  # Make image 4D for conv operation.
  image = tf.expand_dims(image, 0)
  # SMOOTH PIL Kernel.
  if orig_image.shape.rank == 3:
    kernel = tf.constant([[1, 1, 1], [1, 5, 1], [1, 1, 1]],
                         dtype=tf.float32,
                         shape=[3, 3, 1, 1]) / 13.
    # Tile across channel dimension.
    kernel = tf.tile(kernel, [1, 1, 3, 1])
    strides = [1, 1, 1, 1]
    degenerate = tf.nn.depthwise_conv2d(
        image, kernel, strides, padding='VALID', dilations=[1, 1])
  elif orig_image.shape.rank == 4:
    kernel = tf.constant([[1, 1, 1], [1, 5, 1], [1, 1, 1]],
                         dtype=tf.float32,
                         shape=[1, 3, 3, 1, 1]) / 13.
    strides = [1, 1, 1, 1, 1]
    # Run the kernel across each channel
    channels = tf.split(image, 3, axis=-1)
    degenerates = [
        tf.nn.conv3d(channel, kernel, strides, padding='VALID',
                     dilations=[1, 1, 1, 1, 1])
        for channel in channels
    ]
    degenerate = tf.concat(degenerates, -1)
  else:
    raise ValueError('Bad image rank: {}'.format(image.shape.rank))
  degenerate = tf.clip_by_value(degenerate, 0.0, 255.0)
  degenerate = tf.squeeze(tf.cast(degenerate, tf.uint8), [0])

  # For the borders of the resulting image, fill in the values of the
  # original image.
  mask = tf.ones_like(degenerate)
  paddings = [[0, 0]] * (orig_image.shape.rank - 3)
  padded_mask = tf.pad(mask, paddings + [[1, 1], [1, 1], [0, 0]])
  padded_degenerate = tf.pad(degenerate, paddings + [[1, 1], [1, 1], [0, 0]])
  result = tf.where(tf.equal(padded_mask, 1), padded_degenerate, orig_image)

  # Blend the final result.
  return blend(result, orig_image, factor)


def equalize(image: tf.Tensor) -> tf.Tensor:
  """Implements Equalize function from PIL using TF ops."""

  def scale_channel(im, c):
    """Scale the data in the channel to implement equalize."""
    im = tf.cast(im[..., c], tf.int32)
    # Compute the histogram of the image channel.
    histo = tf.histogram_fixed_width(im, [0, 255], nbins=256)

    # For the purposes of computing the step, filter out the nonzeros.
    nonzero = tf.where(tf.not_equal(histo, 0))
    nonzero_histo = tf.reshape(tf.gather(histo, nonzero), [-1])
    step = (tf.reduce_sum(nonzero_histo) - nonzero_histo[-1]) // 255

    def build_lut(histo, step):
      # Compute the cumulative sum, shifting by step // 2
      # and then normalization by step.
      lut = (tf.cumsum(histo) + (step // 2)) // step
      # Shift lut, prepending with 0.
      lut = tf.concat([[0], lut[:-1]], 0)
      # Clip the counts to be in range.  This is done
      # in the C code for image.point.
      return tf.clip_by_value(lut, 0, 255)

    # If step is zero, return the original image.  Otherwise, build
    # lut from the full histogram and step and then index from it.
    result = tf.cond(
        tf.equal(step, 0), lambda: im,
        lambda: tf.gather(build_lut(histo, step), im))

    return tf.cast(result, tf.uint8)

  # Assumes RGB for now.  Scales each channel independently
  # and then stacks the result.
  s1 = scale_channel(image, 0)
  s2 = scale_channel(image, 1)
  s3 = scale_channel(image, 2)
  image = tf.stack([s1, s2, s3], -1)
  return image


def invert(image: tf.Tensor) -> tf.Tensor:
  """Inverts the image pixels."""
  image = tf.convert_to_tensor(image)
  return 255 - image


def wrap(image: tf.Tensor) -> tf.Tensor:
  """Returns 'image' with an extra channel set to all 1s."""
  shape = tf.shape(image)
  extended_channel = tf.expand_dims(tf.ones(shape[:-1], image.dtype), -1)
  extended = tf.concat([image, extended_channel], axis=-1)
  return extended


def unwrap(image: tf.Tensor, replace: int) -> tf.Tensor:
  """Unwraps an image produced by wrap.

  Where there is a 0 in the last channel for every spatial position,
  the rest of the three channels in that spatial dimension are grayed
  (set to 128).  Operations like translate and shear on a wrapped
  Tensor will leave 0s in empty locations.  Some transformations look
  at the intensity of values to do preprocessing, and we want these
  empty pixels to assume the 'average' value, rather than pure black.


  Args:
    image: A 3D Image Tensor with 4 channels.
    replace: A one or three value 1D tensor to fill empty pixels.

  Returns:
    image: A 3D image Tensor with 3 channels.
  """
  image_shape = tf.shape(image)
  # Flatten the spatial dimensions.
  flattened_image = tf.reshape(image, [-1, image_shape[-1]])

  # Find all pixels where the last channel is zero.
  alpha_channel = tf.expand_dims(flattened_image[..., 3], axis=-1)

  replace = tf.concat([replace, tf.ones([1], image.dtype)], 0)

  # Where they are zero, fill them in with 'replace'.
  flattened_image = tf.where(
      tf.equal(alpha_channel, 0),
      tf.ones_like(flattened_image, dtype=image.dtype) * replace,
      flattened_image)

  image = tf.reshape(flattened_image, image_shape)
  image = tf.slice(
      image,
      [0] * image.shape.rank,
      tf.concat([image_shape[:-1], [3]], -1))
  return image


def _scale_bbox_only_op_probability(prob):
  """Reduce the probability of the bbox-only operation.

  Probability is reduced so that we do not distort the content of too many
  bounding boxes that are close to each other. The value of 3.0 was a chosen
  hyper parameter when designing the autoaugment algorithm that we found
  empirically to work well.

  Args:
    prob: Float that is the probability of applying the bbox-only operation.

  Returns:
    Reduced probability.
  """
  return prob / 3.0


def _apply_bbox_augmentation(image, bbox, augmentation_func, *args):
  """Applies augmentation_func to the subsection of image indicated by bbox.

  Args:
    image: 3D uint8 Tensor.
    bbox: 1D Tensor that has 4 elements (min_y, min_x, max_y, max_x)
      of type float that represents the normalized coordinates between 0 and 1.
    augmentation_func: Augmentation function that will be applied to the
      subsection of image.
    *args: Additional parameters that will be passed into augmentation_func
      when it is called.

  Returns:
    A modified version of image, where the bbox location in the image will
    have `ugmentation_func applied to it.
  """
  image_height = tf.cast(tf.shape(image)[0], tf.float32)
  image_width = tf.cast(tf.shape(image)[1], tf.float32)
  min_y = tf.cast(image_height * bbox[0], tf.int32)
  min_x = tf.cast(image_width * bbox[1], tf.int32)
  max_y = tf.cast(image_height * bbox[2], tf.int32)
  max_x = tf.cast(image_width * bbox[3], tf.int32)
  image_height = tf.cast(image_height, tf.int32)
  image_width = tf.cast(image_width, tf.int32)

  # Clip to be sure the max values do not fall out of range.
  max_y = tf.minimum(max_y, image_height - 1)
  max_x = tf.minimum(max_x, image_width - 1)

  # Get the sub-tensor that is the image within the bounding box region.
  bbox_content = image[min_y:max_y + 1, min_x:max_x + 1, :]

  # Apply the augmentation function to the bbox portion of the image.
  augmented_bbox_content = augmentation_func(bbox_content, *args)

  # Pad the augmented_bbox_content and the mask to match the shape of original
  # image.
  augmented_bbox_content = tf.pad(augmented_bbox_content,
                                  [[min_y, (image_height - 1) - max_y],
                                   [min_x, (image_width - 1) - max_x],
                                   [0, 0]])

  # Create a mask that will be used to zero out a part of the original image.
  mask_tensor = tf.zeros_like(bbox_content)

  mask_tensor = tf.pad(mask_tensor,
                       [[min_y, (image_height - 1) - max_y],
                        [min_x, (image_width - 1) - max_x],
                        [0, 0]],
                       constant_values=1)
  # Replace the old bbox content with the new augmented content.
  image = image * mask_tensor + augmented_bbox_content
  return image


def _concat_bbox(bbox, bboxes):
  """Helper function that concates bbox to bboxes along the first dimension."""

  # Note if all elements in bboxes are -1 (_INVALID_BOX), then this means
  # we discard bboxes and start the bboxes Tensor with the current bbox.
  bboxes_sum_check = tf.reduce_sum(bboxes)
  bbox = tf.expand_dims(bbox, 0)
  # This check will be true when it is an _INVALID_BOX
  bboxes = tf.cond(tf.equal(bboxes_sum_check, -4.0),
                   lambda: bbox,
                   lambda: tf.concat([bboxes, bbox], 0))
  return bboxes


def _apply_bbox_augmentation_wrapper(image, bbox, new_bboxes, prob,
                                     augmentation_func, func_changes_bbox,
                                     *args):
  """Applies _apply_bbox_augmentation with probability prob.

  Args:
    image: 3D uint8 Tensor.
    bbox: 1D Tensor that has 4 elements (min_y, min_x, max_y, max_x)
      of type float that represents the normalized coordinates between 0 and 1.
    new_bboxes: 2D Tensor that is a list of the bboxes in the image after they
      have been altered by aug_func. These will only be changed when
      func_changes_bbox is set to true. Each bbox has 4 elements
      (min_y, min_x, max_y, max_x) of type float that are the normalized
      bbox coordinates between 0 and 1.
    prob: Float that is the probability of applying _apply_bbox_augmentation.
    augmentation_func: Augmentation function that will be applied to the
      subsection of image.
    func_changes_bbox: Boolean. Does augmentation_func return bbox in addition
      to image.
    *args: Additional parameters that will be passed into augmentation_func
      when it is called.

  Returns:
    A tuple. Fist element is a modified version of image, where the bbox
    location in the image will have augmentation_func applied to it if it is
    chosen to be called with probability `prob`. The second element is a
    Tensor of Tensors of length 4 that will contain the altered bbox after
    applying augmentation_func.
  """
  should_apply_op = tf.cast(
      tf.floor(tf.random.uniform([], dtype=tf.float32) + prob), tf.bool)
  if func_changes_bbox:
    augmented_image, bbox = tf.cond(
        should_apply_op,
        lambda: augmentation_func(image, bbox, *args),
        lambda: (image, bbox))
  else:
    augmented_image = tf.cond(
        should_apply_op,
        lambda: _apply_bbox_augmentation(image, bbox, augmentation_func, *args),
        lambda: image)
  new_bboxes = _concat_bbox(bbox, new_bboxes)
  return augmented_image, new_bboxes


def _apply_multi_bbox_augmentation_wrapper(image, bboxes, prob, aug_func,
                                           func_changes_bbox, *args):
  """Checks to be sure num bboxes > 0 before calling inner function."""
  num_bboxes = tf.shape(bboxes)[0]
  image, bboxes = tf.cond(
      tf.equal(num_bboxes, 0),
      lambda: (image, bboxes),
      # pylint:disable=g-long-lambda
      lambda: _apply_multi_bbox_augmentation(
          image, bboxes, prob, aug_func, func_changes_bbox, *args))
  # pylint:enable=g-long-lambda
  return image, bboxes


# Represents an invalid bounding box that is used for checking for padding
# lists of bounding box coordinates for a few augmentation operations
_INVALID_BOX = [[-1.0, -1.0, -1.0, -1.0]]


def _apply_multi_bbox_augmentation(image, bboxes, prob, aug_func,
                                   func_changes_bbox, *args):
  """Applies aug_func to the image for each bbox in bboxes.

  Args:
    image: 3D uint8 Tensor.
    bboxes: 2D Tensor that is a list of the bboxes in the image. Each bbox
      has 4 elements (min_y, min_x, max_y, max_x) of type float.
    prob: Float that is the probability of applying aug_func to a specific
      bounding box within the image.
    aug_func: Augmentation function that will be applied to the
      subsections of image indicated by the bbox values in bboxes.
    func_changes_bbox: Boolean. Does augmentation_func return bbox in addition
      to image.
    *args: Additional parameters that will be passed into augmentation_func
      when it is called.

  Returns:
    A modified version of image, where each bbox location in the image will
    have augmentation_func applied to it if it is chosen to be called with
    probability prob independently across all bboxes. Also the final
    bboxes are returned that will be unchanged if func_changes_bbox is set to
    false and if true, the new altered ones will be returned.

  Raises:
    ValueError if applied to video.
  """
  if image.shape.rank == 4:
    raise ValueError('Image rank 4 is not supported')

  # Will keep track of the new altered bboxes after aug_func is repeatedly
  # applied. The -1 values are a dummy value and this first Tensor will be
  # removed upon appending the first real bbox.
  new_bboxes = tf.constant(_INVALID_BOX)

  # If the bboxes are empty, then just give it _INVALID_BOX. The result
  # will be thrown away.
  bboxes = tf.cond(tf.equal(tf.size(bboxes), 0),
                   lambda: tf.constant(_INVALID_BOX),
                   lambda: bboxes)

  bboxes = tf.ensure_shape(bboxes, (None, 4))

  # pylint:disable=g-long-lambda
  wrapped_aug_func = (
      lambda _image, bbox, _new_bboxes: _apply_bbox_augmentation_wrapper(
          _image, bbox, _new_bboxes, prob, aug_func, func_changes_bbox, *args))
  # pylint:enable=g-long-lambda

  # Setup the while_loop.
  num_bboxes = tf.shape(bboxes)[0]  # We loop until we go over all bboxes.
  idx = tf.constant(0)  # Counter for the while loop.

  # Conditional function when to end the loop once we go over all bboxes
  # images_and_bboxes contain (_image, _new_bboxes)
  cond = lambda _idx, _images_and_bboxes: tf.less(_idx, num_bboxes)

  # Shuffle the bboxes so that the augmentation order is not deterministic if
  # we are not changing the bboxes with aug_func.
  if not func_changes_bbox:
    loop_bboxes = tf.random.shuffle(bboxes)
  else:
    loop_bboxes = bboxes

  # Main function of while_loop where we repeatedly apply augmentation on the
  # bboxes in the image.
  # pylint:disable=g-long-lambda
  body = lambda _idx, _images_and_bboxes: [
      _idx + 1, wrapped_aug_func(_images_and_bboxes[0],
                                 loop_bboxes[_idx],
                                 _images_and_bboxes[1])]
  # pylint:enable=g-long-lambda

  _, (image, new_bboxes) = tf.while_loop(
      cond, body, [idx, (image, new_bboxes)],
      shape_invariants=[idx.get_shape(),
                        (image.get_shape(), tf.TensorShape([None, 4]))])

  # Either return the altered bboxes or the original ones depending on if
  # we altered them in anyway.
  if func_changes_bbox:
    final_bboxes = new_bboxes
  else:
    final_bboxes = bboxes
  return image, final_bboxes


def _clip_bbox(min_y, min_x, max_y, max_x):
  """Clip bounding box coordinates between 0 and 1.

  Args:
    min_y: Normalized bbox coordinate of type float between 0 and 1.
    min_x: Normalized bbox coordinate of type float between 0 and 1.
    max_y: Normalized bbox coordinate of type float between 0 and 1.
    max_x: Normalized bbox coordinate of type float between 0 and 1.

  Returns:
    Clipped coordinate values between 0 and 1.
  """
  min_y = tf.clip_by_value(min_y, 0.0, 1.0)
  min_x = tf.clip_by_value(min_x, 0.0, 1.0)
  max_y = tf.clip_by_value(max_y, 0.0, 1.0)
  max_x = tf.clip_by_value(max_x, 0.0, 1.0)
  return min_y, min_x, max_y, max_x


def _check_bbox_area(min_y, min_x, max_y, max_x, delta=0.05):
  """Adjusts bbox coordinates to make sure the area is > 0.

  Args:
    min_y: Normalized bbox coordinate of type float between 0 and 1.
    min_x: Normalized bbox coordinate of type float between 0 and 1.
    max_y: Normalized bbox coordinate of type float between 0 and 1.
    max_x: Normalized bbox coordinate of type float between 0 and 1.
    delta: Float, this is used to create a gap of size 2 * delta between
      bbox min/max coordinates that are the same on the boundary.
      This prevents the bbox from having an area of zero.

  Returns:
    Tuple of new bbox coordinates between 0 and 1 that will now have a
    guaranteed area > 0.
  """
  height = max_y - min_y
  width = max_x - min_x
  def _adjust_bbox_boundaries(min_coord, max_coord):
    # Make sure max is never 0 and min is never 1.
    max_coord = tf.maximum(max_coord, 0.0 + delta)
    min_coord = tf.minimum(min_coord, 1.0 - delta)
    return min_coord, max_coord
  min_y, max_y = tf.cond(tf.equal(height, 0.0),
                         lambda: _adjust_bbox_boundaries(min_y, max_y),
                         lambda: (min_y, max_y))
  min_x, max_x = tf.cond(tf.equal(width, 0.0),
                         lambda: _adjust_bbox_boundaries(min_x, max_x),
                         lambda: (min_x, max_x))
  return min_y, min_x, max_y, max_x


def _rotate_bbox(bbox, image_height, image_width, degrees):
  """Rotates the bbox coordinated by degrees.

  Args:
    bbox: 1D Tensor that has 4 elements (min_y, min_x, max_y, max_x)
      of type float that represents the normalized coordinates between 0 and 1.
    image_height: Int, height of the image.
    image_width: Int, height of the image.
    degrees: Float, a scalar angle in degrees to rotate all images by. If
      degrees is positive the image will be rotated clockwise otherwise it will
      be rotated counterclockwise.

  Returns:
    A tensor of the same shape as bbox, but now with the rotated coordinates.
  """
  image_height, image_width = (
      tf.cast(image_height, tf.float32), tf.cast(image_width, tf.float32))

  # Convert from degrees to radians.
  degrees_to_radians = math.pi / 180.0
  radians = degrees * degrees_to_radians

  # Translate the bbox to the center of the image and turn the normalized 0-1
  # coordinates to absolute pixel locations.
  # Y coordinates are made negative as the y axis of images goes down with
  # increasing pixel values, so we negate to make sure x axis and y axis points
  # are in the traditionally positive direction.
  min_y = -tf.cast(image_height * (bbox[0] - 0.5), tf.int32)
  min_x = tf.cast(image_width * (bbox[1] - 0.5), tf.int32)
  max_y = -tf.cast(image_height * (bbox[2] - 0.5), tf.int32)
  max_x = tf.cast(image_width * (bbox[3] - 0.5), tf.int32)
  coordinates = tf.stack(
      [[min_y, min_x], [min_y, max_x], [max_y, min_x], [max_y, max_x]])
  coordinates = tf.cast(coordinates, tf.float32)
  # Rotate the coordinates according to the rotation matrix clockwise if
  # radians is positive, else negative
  rotation_matrix = tf.stack(
      [[tf.cos(radians), tf.sin(radians)],
       [-tf.sin(radians), tf.cos(radians)]])
  new_coords = tf.cast(
      tf.matmul(rotation_matrix, tf.transpose(coordinates)), tf.int32)
  # Find min/max values and convert them back to normalized 0-1 floats.
  min_y = -(
      tf.cast(tf.reduce_max(new_coords[0, :]), tf.float32) / image_height - 0.5)
  min_x = tf.cast(tf.reduce_min(new_coords[1, :]),
                  tf.float32) / image_width + 0.5
  max_y = -(
      tf.cast(tf.reduce_min(new_coords[0, :]), tf.float32) / image_height - 0.5)
  max_x = tf.cast(tf.reduce_max(new_coords[1, :]),
                  tf.float32) / image_width + 0.5

  # Clip the bboxes to be sure the fall between [0, 1].
  min_y, min_x, max_y, max_x = _clip_bbox(min_y, min_x, max_y, max_x)
  min_y, min_x, max_y, max_x = _check_bbox_area(min_y, min_x, max_y, max_x)
  return tf.stack([min_y, min_x, max_y, max_x])


def rotate_with_bboxes(image, bboxes, degrees, replace):
  """Equivalent of PIL Rotate that rotates the image and bbox.

  Args:
    image: 3D uint8 Tensor.
    bboxes: 2D Tensor that is a list of the bboxes in the image. Each bbox
      has 4 elements (min_y, min_x, max_y, max_x) of type float.
    degrees: Float, a scalar angle in degrees to rotate all images by. If
      degrees is positive the image will be rotated clockwise otherwise it will
      be rotated counterclockwise.
    replace: A one or three value 1D tensor to fill empty pixels.

  Returns:
    A tuple containing a 3D uint8 Tensor that will be the result of rotating
    image by degrees. The second element of the tuple is bboxes, where now
    the coordinates will be shifted to reflect the rotated image.

  Raises:
    ValueError: If applied to video.
  """
  if image.shape.rank == 4:
    raise ValueError('Image rank 4 is not supported')

  # Rotate the image.
  image = wrapped_rotate(image, degrees, replace)

  # Convert bbox coordinates to pixel values.
  image_height = tf.shape(image)[0]
  image_width = tf.shape(image)[1]
  # pylint:disable=g-long-lambda
  wrapped_rotate_bbox = lambda bbox: _rotate_bbox(
      bbox, image_height, image_width, degrees)
  # pylint:enable=g-long-lambda
  bboxes = tf.map_fn(wrapped_rotate_bbox, bboxes)
  return image, bboxes


def _shear_bbox(bbox, image_height, image_width, level, shear_horizontal):
  """Shifts the bbox according to how the image was sheared.

  Args:
    bbox: 1D Tensor that has 4 elements (min_y, min_x, max_y, max_x)
      of type float that represents the normalized coordinates between 0 and 1.
    image_height: Int, height of the image.
    image_width: Int, height of the image.
    level: Float. How much to shear the image.
    shear_horizontal: If true then shear in X dimension else shear in
      the Y dimension.

  Returns:
    A tensor of the same shape as bbox, but now with the shifted coordinates.
  """
  image_height, image_width = (
      tf.cast(image_height, tf.float32), tf.cast(image_width, tf.float32))

  # Change bbox coordinates to be pixels.
  min_y = tf.cast(image_height * bbox[0], tf.int32)
  min_x = tf.cast(image_width * bbox[1], tf.int32)
  max_y = tf.cast(image_height * bbox[2], tf.int32)
  max_x = tf.cast(image_width * bbox[3], tf.int32)
  coordinates = tf.stack(
      [[min_y, min_x], [min_y, max_x], [max_y, min_x], [max_y, max_x]])
  coordinates = tf.cast(coordinates, tf.float32)

  # Shear the coordinates according to the translation matrix.
  if shear_horizontal:
    translation_matrix = tf.stack(
        [[1, 0], [-level, 1]])
  else:
    translation_matrix = tf.stack(
        [[1, -level], [0, 1]])
  translation_matrix = tf.cast(translation_matrix, tf.float32)
  new_coords = tf.cast(
      tf.matmul(translation_matrix, tf.transpose(coordinates)), tf.int32)

  # Find min/max values and convert them back to floats.
  min_y = tf.cast(tf.reduce_min(new_coords[0, :]), tf.float32) / image_height
  min_x = tf.cast(tf.reduce_min(new_coords[1, :]), tf.float32) / image_width
  max_y = tf.cast(tf.reduce_max(new_coords[0, :]), tf.float32) / image_height
  max_x = tf.cast(tf.reduce_max(new_coords[1, :]), tf.float32) / image_width

  # Clip the bboxes to be sure the fall between [0, 1].
  min_y, min_x, max_y, max_x = _clip_bbox(min_y, min_x, max_y, max_x)
  min_y, min_x, max_y, max_x = _check_bbox_area(min_y, min_x, max_y, max_x)
  return tf.stack([min_y, min_x, max_y, max_x])


def shear_with_bboxes(image, bboxes, level, replace, shear_horizontal):
  """Applies Shear Transformation to the image and shifts the bboxes.

  Args:
    image: 3D uint8 Tensor.
    bboxes: 2D Tensor that is a list of the bboxes in the image. Each bbox
      has 4 elements (min_y, min_x, max_y, max_x) of type float with values
      between [0, 1].
    level: Float. How much to shear the image. This value will be between
      -0.3 to 0.3.
    replace: A one or three value 1D tensor to fill empty pixels.
    shear_horizontal: Boolean. If true then shear in X dimension else shear in
      the Y dimension.

  Returns:
    A tuple containing a 3D uint8 Tensor that will be the result of shearing
    image by level. The second element of the tuple is bboxes, where now
    the coordinates will be shifted to reflect the sheared image.

  Raises:
    ValueError: If applied to video.
  """
  if image.shape.rank == 4:
    raise ValueError('Image rank 4 is not supported')

  if shear_horizontal:
    image = shear_x(image, level, replace)
  else:
    image = shear_y(image, level, replace)

  # Convert bbox coordinates to pixel values.
  image_height = tf.shape(image)[0]
  image_width = tf.shape(image)[1]
  # pylint:disable=g-long-lambda
  wrapped_shear_bbox = lambda bbox: _shear_bbox(
      bbox, image_height, image_width, level, shear_horizontal)
  # pylint:enable=g-long-lambda
  bboxes = tf.map_fn(wrapped_shear_bbox, bboxes)
  return image, bboxes


def _shift_bbox(bbox, image_height, image_width, pixels, shift_horizontal):
  """Shifts the bbox coordinates by pixels.

  Args:
    bbox: 1D Tensor that has 4 elements (min_y, min_x, max_y, max_x)
      of type float that represents the normalized coordinates between 0 and 1.
    image_height: Int, height of the image.
    image_width: Int, width of the image.
    pixels: An int. How many pixels to shift the bbox.
    shift_horizontal: Boolean. If true then shift in X dimension else shift in
      Y dimension.

  Returns:
    A tensor of the same shape as bbox, but now with the shifted coordinates.
  """
  pixels = tf.cast(pixels, tf.int32)
  # Convert bbox to integer pixel locations.
  min_y = tf.cast(tf.cast(image_height, tf.float32) * bbox[0], tf.int32)
  min_x = tf.cast(tf.cast(image_width, tf.float32) * bbox[1], tf.int32)
  max_y = tf.cast(tf.cast(image_height, tf.float32) * bbox[2], tf.int32)
  max_x = tf.cast(tf.cast(image_width, tf.float32) * bbox[3], tf.int32)

  if shift_horizontal:
    min_x = tf.maximum(0, min_x - pixels)
    max_x = tf.minimum(image_width, max_x - pixels)
  else:
    min_y = tf.maximum(0, min_y - pixels)
    max_y = tf.minimum(image_height, max_y - pixels)

  # Convert bbox back to floats.
  min_y = tf.cast(min_y, tf.float32) / tf.cast(image_height, tf.float32)
  min_x = tf.cast(min_x, tf.float32) / tf.cast(image_width, tf.float32)
  max_y = tf.cast(max_y, tf.float32) / tf.cast(image_height, tf.float32)
  max_x = tf.cast(max_x, tf.float32) / tf.cast(image_width, tf.float32)

  # Clip the bboxes to be sure the fall between [0, 1].
  min_y, min_x, max_y, max_x = _clip_bbox(min_y, min_x, max_y, max_x)
  min_y, min_x, max_y, max_x = _check_bbox_area(min_y, min_x, max_y, max_x)
  return tf.stack([min_y, min_x, max_y, max_x])


def translate_bbox(image, bboxes, pixels, replace, shift_horizontal):
  """Equivalent of PIL Translate in X/Y dimension that shifts image and bbox.

  Args:
    image: 3D uint8 Tensor.
    bboxes: 2D Tensor that is a list of the bboxes in the image. Each bbox
      has 4 elements (min_y, min_x, max_y, max_x) of type float with values
      between [0, 1].
    pixels: An int. How many pixels to shift the image and bboxes
    replace: A one or three value 1D tensor to fill empty pixels.
    shift_horizontal: Boolean. If true then shift in X dimension else shift in
      Y dimension.

  Returns:
    A tuple containing a 3D uint8 Tensor that will be the result of translating
    image by pixels. The second element of the tuple is bboxes, where now
    the coordinates will be shifted to reflect the shifted image.

  Raises:
    ValueError if applied to video.
  """
  if image.shape.rank == 4:
    raise ValueError('Image rank 4 is not supported')

  if shift_horizontal:
    image = translate_x(image, pixels, replace)
  else:
    image = translate_y(image, pixels, replace)

  # Convert bbox coordinates to pixel values.
  image_height = tf.shape(image)[0]
  image_width = tf.shape(image)[1]
  # pylint:disable=g-long-lambda
  wrapped_shift_bbox = lambda bbox: _shift_bbox(
      bbox, image_height, image_width, pixels, shift_horizontal)
  # pylint:enable=g-long-lambda
  bboxes = tf.map_fn(wrapped_shift_bbox, bboxes)
  return image, bboxes


def translate_y_only_bboxes(
    image: tf.Tensor, bboxes: tf.Tensor, prob: float, pixels: int, replace):
  """Apply translate_y to each bbox in the image with probability prob."""
  if bboxes.shape.rank == 4:
    raise ValueError('translate_y_only_bboxes does not support rank 4 boxes')

  func_changes_bbox = False
  prob = _scale_bbox_only_op_probability(prob)
  return _apply_multi_bbox_augmentation_wrapper(
      image, bboxes, prob, translate_y, func_changes_bbox, pixels, replace)


def _randomly_negate_tensor(tensor):
  """With 50% prob turn the tensor negative."""
  should_flip = tf.cast(tf.floor(tf.random.uniform([]) + 0.5), tf.bool)
  final_tensor = tf.cond(should_flip, lambda: tensor, lambda: -tensor)
  return final_tensor


def _rotate_level_to_arg(level: float):
  level = (level / _MAX_LEVEL) * 30.
  level = _randomly_negate_tensor(level)
  return (level,)


def _shrink_level_to_arg(level: float):
  """Converts level to ratio by which we shrink the image content."""
  if level == 0:
    return (1.0,)  # if level is zero, do not shrink the image
  # Maximum shrinking ratio is 2.9.
  level = 2. / (_MAX_LEVEL / level) + 0.9
  return (level,)


def _enhance_level_to_arg(level: float):
  return ((level / _MAX_LEVEL) * 1.8 + 0.1,)


def _shear_level_to_arg(level: float):
  level = (level / _MAX_LEVEL) * 0.3
  # Flip level to negative with 50% chance.
  level = _randomly_negate_tensor(level)
  return (level,)


def _translate_level_to_arg(level: float, translate_const: float):
  level = (level / _MAX_LEVEL) * float(translate_const)
  # Flip level to negative with 50% chance.
  level = _randomly_negate_tensor(level)
  return (level,)


def _gaussian_noise_level_to_arg(level: float, translate_const: float):
  low_std = (level / _MAX_LEVEL)
  high_std = translate_const * low_std
  return low_std, high_std


def _mult_to_arg(level: float, multiplier: float = 1.):
  return (int((level / _MAX_LEVEL) * multiplier),)


def _apply_func_with_prob(func: Any, image: tf.Tensor,
                          bboxes: Optional[tf.Tensor], args: Any, prob: float):
  """Apply `func` to image w/ `args` as input with probability `prob`."""
  assert isinstance(args, tuple)
  assert inspect.getfullargspec(func)[0][1] == 'bboxes'

  # Apply the function with probability `prob`.
  should_apply_op = tf.cast(
      tf.floor(tf.random.uniform([], dtype=tf.float32) + prob), tf.bool)
  augmented_image, augmented_bboxes = tf.cond(
      should_apply_op,
      lambda: func(image, bboxes, *args),
      lambda: (image, bboxes))
  return augmented_image, augmented_bboxes


def select_and_apply_random_policy(
    policies: Any, image: tf.Tensor, bboxes: Optional[tf.Tensor] = None
) -> Tuple[tf.Tensor, Optional[tf.Tensor]]:
  """Select a random policy from `policies` and apply it to `image`."""
  policy_to_select = tf.random.uniform([], maxval=len(policies), dtype=tf.int32)
  # Note that using tf.case instead of tf.conds would result in significantly
  # larger graphs and would even break export for some larger policies.
  for (i, policy) in enumerate(policies):
    image, bboxes = tf.cond(
        tf.equal(i, policy_to_select),
        lambda selected_policy=policy: selected_policy(image, bboxes),
        lambda: (image, bboxes))
  return image, bboxes


NAME_TO_FUNC = {
    'AutoContrast': autocontrast,
    'Equalize': equalize,
    'Invert': invert,
    'Rotate': wrapped_rotate,
    'Posterize': posterize,
    'Solarize': solarize,
    'SolarizeAdd': solarize_add,
    'Color': color,
    'Contrast': contrast,
    'Brightness': brightness,
    'Sharpness': sharpness,
    'ShearX': shear_x,
    'ShearY': shear_y,
    'TranslateX': translate_x,
    'TranslateY': translate_y,
    'Cutout': cutout,
    'Rotate_BBox': rotate_with_bboxes,
    'Grayscale': grayscale,
    'Gaussian_Noise': gaussian_noise,
    # pylint:disable=g-long-lambda
    'ShearX_BBox': lambda image, bboxes, level, replace: shear_with_bboxes(
        image, bboxes, level, replace, shear_horizontal=True),
    'ShearY_BBox': lambda image, bboxes, level, replace: shear_with_bboxes(
        image, bboxes, level, replace, shear_horizontal=False),
    'TranslateX_BBox': lambda image, bboxes, pixels, replace: translate_bbox(
        image, bboxes, pixels, replace, shift_horizontal=True),
    'TranslateY_BBox': lambda image, bboxes, pixels, replace: translate_bbox(
        image, bboxes, pixels, replace, shift_horizontal=False),
    # pylint:enable=g-long-lambda
    'TranslateY_Only_BBoxes': translate_y_only_bboxes,
}

# Functions that require a `bboxes` parameter.
REQUIRE_BOXES_FUNCS = frozenset({
    'Rotate_BBox',
    'ShearX_BBox',
    'ShearY_BBox',
    'TranslateX_BBox',
    'TranslateY_BBox',
    'TranslateY_Only_BBoxes',
})

# Functions that have a 'prob' parameter
PROB_FUNCS = frozenset({
    'TranslateY_Only_BBoxes',
})

# Functions that have a 'replace' parameter
REPLACE_FUNCS = frozenset({
    'Rotate',
    'TranslateX',
    'ShearX',
    'ShearY',
    'TranslateY',
    'Cutout',
    'Rotate_BBox',
    'ShearX_BBox',
    'ShearY_BBox',
    'TranslateX_BBox',
    'TranslateY_BBox',
    'TranslateY_Only_BBoxes',
})


def level_to_arg(cutout_const: float, translate_const: float):
  """Creates a dict mapping image operation names to their arguments."""

  no_arg = lambda level: ()
  posterize_arg = lambda level: _mult_to_arg(level, 4)
  solarize_arg = lambda level: _mult_to_arg(level, 256)
  solarize_add_arg = lambda level: _mult_to_arg(level, 110)
  cutout_arg = lambda level: _mult_to_arg(level, cutout_const)
  translate_arg = lambda level: _translate_level_to_arg(level, translate_const)
  translate_bbox_arg = lambda level: _translate_level_to_arg(level, 120)

  args = {
      'AutoContrast': no_arg,
      'Equalize': no_arg,
      'Invert': no_arg,
      'Rotate': _rotate_level_to_arg,
      'Posterize': posterize_arg,
      'Solarize': solarize_arg,
      'SolarizeAdd': solarize_add_arg,
      'Color': _enhance_level_to_arg,
      'Contrast': _enhance_level_to_arg,
      'Brightness': _enhance_level_to_arg,
      'Sharpness': _enhance_level_to_arg,
      'ShearX': _shear_level_to_arg,
      'ShearY': _shear_level_to_arg,
      'Cutout': cutout_arg,
      'TranslateX': translate_arg,
      'TranslateY': translate_arg,
      'Rotate_BBox': _rotate_level_to_arg,
      'ShearX_BBox': _shear_level_to_arg,
      'ShearY_BBox': _shear_level_to_arg,
      'Grayscale': no_arg,
      # pylint:disable=g-long-lambda
      'Gaussian_Noise': lambda level: _gaussian_noise_level_to_arg(
          level, translate_const),
      # pylint:disable=g-long-lambda
      'TranslateX_BBox': lambda level: _translate_level_to_arg(
          level, translate_const),
      'TranslateY_BBox': lambda level: _translate_level_to_arg(
          level, translate_const),
      # pylint:enable=g-long-lambda
      'TranslateY_Only_BBoxes': translate_bbox_arg,
  }
  return args


def bbox_wrapper(func):
  """Adds a bboxes function argument to func and returns unchanged bboxes."""
  def wrapper(images, bboxes, *args, **kwargs):
    return (func(images, *args, **kwargs), bboxes)
  return wrapper


def _parse_policy_info(name: str,
                       prob: float,
                       level: float,
                       replace_value: List[int],
                       cutout_const: float,
                       translate_const: float,
                       level_std: float = 0.) -> Tuple[Any, float, Any]:
  """Return the function that corresponds to `name` and update `level` param."""
  func = NAME_TO_FUNC[name]

  if level_std > 0:
    level += tf.random.normal([], dtype=tf.float32)
    level = tf.clip_by_value(level, 0., _MAX_LEVEL)

  args = level_to_arg(cutout_const, translate_const)[name](level)

  if name in PROB_FUNCS:
    # Add in the prob arg if it is required for the function that is called.
    args = tuple([prob] + list(args))

  if name in REPLACE_FUNCS:
    # Add in replace arg if it is required for the function that is called.
    args = tuple(list(args) + [replace_value])

  # Add bboxes as the second positional argument for the function if it does
  # not already exist.
  if 'bboxes' not in inspect.getfullargspec(func)[0]:
    func = bbox_wrapper(func)

  return func, prob, args


class ImageAugment(object):
  """Image augmentation class for applying image distortions."""

  def distort(
      self,
      image: tf.Tensor
  ) -> tf.Tensor:
    """Given an image tensor, returns a distorted image with the same shape.

    Expect the image tensor values are in the range [0, 255].

    Args:
      image: `Tensor` of shape [height, width, 3] or
        [num_frames, height, width, 3] representing an image or image sequence.

    Returns:
      The augmented version of `image`.
    """
    raise NotImplementedError()

  def distort_with_boxes(
      self,
      image: tf.Tensor,
      bboxes: tf.Tensor
  ) -> Tuple[tf.Tensor, tf.Tensor]:
    """Distorts the image and bounding boxes.

    Expect the image tensor values are in the range [0, 255].

    Args:
      image: `Tensor` of shape [height, width, 3] or
        [num_frames, height, width, 3] representing an image or image sequence.
      bboxes: `Tensor` of shape [num_boxes, 4] or [num_frames, num_boxes, 4]
        representing bounding boxes for an image or image sequence.

    Returns:
      The augmented version of `image` and `bboxes`.
    """
    raise NotImplementedError


def _maybe_identity(x: Optional[tf.Tensor]) -> Optional[tf.Tensor]:
  return tf.identity(x) if x is not None else None


class RandAugment(ImageAugment):
  """Applies the RandAugment policy to images.

  RandAugment is from the paper https://arxiv.org/abs/1909.13719.
  """

  def __init__(self,
               num_layers: int = 2,
               magnitude: float = 10.,
               cutout_const: float = 40.,
               translate_const: float = 100.,
               magnitude_std: float = 0.0,
               prob_to_apply: Optional[float] = None,
               exclude_ops: Optional[List[str]] = None):
    """Applies the RandAugment policy to images.

    Args:
      num_layers: Integer, the number of augmentation transformations to apply
        sequentially to an image. Represented as (N) in the paper. Usually best
        values will be in the range [1, 3].
      magnitude: Integer, shared magnitude across all augmentation operations.
        Represented as (M) in the paper. Usually best values are in the range
        [5, 10].
      cutout_const: multiplier for applying cutout.
      translate_const: multiplier for applying translation.
      magnitude_std: randomness of the severity as proposed by the authors of
        the timm library.
      prob_to_apply: The probability to apply the selected augmentation at each
        layer.
      exclude_ops: exclude selected operations.
    """
    super(RandAugment, self).__init__()

    self.num_layers = num_layers
    self.magnitude = float(magnitude)
    self.cutout_const = float(cutout_const)
    self.translate_const = float(translate_const)
    self.prob_to_apply = (
        float(prob_to_apply) if prob_to_apply is not None else None)
    self.available_ops = [
        'AutoContrast', 'Equalize', 'Invert', 'Rotate', 'Posterize', 'Solarize',
        'Color', 'Contrast', 'Brightness', 'Sharpness', 'ShearX', 'ShearY',
        'TranslateX', 'TranslateY', 'Cutout', 'SolarizeAdd'
    ]
    self.magnitude_std = magnitude_std
    if exclude_ops:
      self.available_ops = [
          op for op in self.available_ops if op not in exclude_ops
      ]

  @classmethod
  def build_for_detection(cls,
                          num_layers: int = 2,
                          magnitude: float = 10.,
                          cutout_const: float = 40.,
                          translate_const: float = 100.,
                          magnitude_std: float = 0.0,
                          prob_to_apply: Optional[float] = None,
                          exclude_ops: Optional[List[str]] = None):
    """Builds a RandAugment that modifies bboxes for geometric transforms."""
    augmenter = cls(
        num_layers=num_layers,
        magnitude=magnitude,
        cutout_const=cutout_const,
        translate_const=translate_const,
        magnitude_std=magnitude_std,
        prob_to_apply=prob_to_apply,
        exclude_ops=exclude_ops)
    box_aware_ops_by_base_name = {
        'Rotate': 'Rotate_BBox',
        'ShearX': 'ShearX_BBox',
        'ShearY': 'ShearY_BBox',
        'TranslateX': 'TranslateX_BBox',
        'TranslateY': 'TranslateY_BBox',
    }
    augmenter.available_ops = [
        box_aware_ops_by_base_name.get(op_name) or op_name
        for op_name in augmenter.available_ops
    ]
    return augmenter

  def _distort_common(
      self,
      image: tf.Tensor,
      bboxes: Optional[tf.Tensor] = None
  ) -> Tuple[tf.Tensor, Optional[tf.Tensor]]:
    """Distorts the image and optionally bounding boxes."""
    input_image_type = image.dtype

    if input_image_type != tf.uint8:
      image = tf.clip_by_value(image, 0.0, 255.0)
      image = tf.cast(image, dtype=tf.uint8)

    replace_value = [128] * 3
    min_prob, max_prob = 0.2, 0.8

    aug_image = image
    aug_bboxes = bboxes

    for _ in range(self.num_layers):
      op_to_select = tf.random.uniform([],
                                       maxval=len(self.available_ops) + 1,
                                       dtype=tf.int32)

      branch_fns = []
      for (i, op_name) in enumerate(self.available_ops):
        prob = tf.random.uniform([],
                                 minval=min_prob,
                                 maxval=max_prob,
                                 dtype=tf.float32)
        func, _, args = _parse_policy_info(op_name, prob, self.magnitude,
                                           replace_value, self.cutout_const,
                                           self.translate_const,
                                           self.magnitude_std)
        branch_fns.append((
            i,
            # pylint:disable=g-long-lambda
            lambda selected_func=func, selected_args=args: selected_func(
                image, bboxes, *selected_args)))
        # pylint:enable=g-long-lambda

      aug_image, aug_bboxes = tf.switch_case(
          branch_index=op_to_select,
          branch_fns=branch_fns,
          default=lambda: (tf.identity(image), _maybe_identity(bboxes)))  # pylint: disable=cell-var-from-loop

      if self.prob_to_apply is not None:
        aug_image, aug_bboxes = tf.cond(
            tf.random.uniform(shape=[], dtype=tf.float32) < self.prob_to_apply,
            lambda: (tf.identity(aug_image), _maybe_identity(aug_bboxes)),
            lambda: (tf.identity(image), _maybe_identity(bboxes)))
      image = aug_image
      bboxes = aug_bboxes

    image = tf.cast(image, dtype=input_image_type)
    return image, bboxes

  def distort(self, image: tf.Tensor) -> tf.Tensor:
    """See base class."""
    image, _ = self._distort_common(image)
    return image

  def distort_with_boxes(self, image: tf.Tensor,
                         bboxes: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """See base class."""
    image, bboxes = self._distort_common(image, bboxes)
    assert bboxes is not None
    return image, bboxes