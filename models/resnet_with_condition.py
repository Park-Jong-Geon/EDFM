from typing import Any, Callable, Sequence, Tuple
import functools
import inspect
import math
from einops import rearrange
import jax
import jax.numpy as jnp
import flax.linen as nn

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = jnp.exp(
        -math.log(max_period) * jnp.arange(0, half, dtype=jnp.float32) / half
    )
    args = timesteps[..., None] * freqs[None]
    embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
    if len(timesteps.shape) == 2:
        embedding = rearrange(embedding, "b n d -> b (n d)")
    if dim % 2:
        embedding = jnp.concatenate(
            [embedding, jnp.zeros_like(embedding[:, :1])], axis=-1)
    return embedding

class FlaxResNetwithCondition(nn.Module):
    depth:        int = 20
    widen_factor: float = 1.
    dtype:        Any = jnp.float32
    pixel_mean:   Tuple[int] = (0.0, 0.0, 0.0)
    pixel_std:    Tuple[int] = (1.0, 1.0, 1.0)
    num_classes:  int = None
    conv:         nn.Module = functools.partial(nn.Conv, use_bias=False,
                                                kernel_init=jax.nn.initializers.he_normal(),
                                                bias_init=jax.nn.initializers.zeros)
    norm:         nn.Module = functools.partial(nn.BatchNorm, momentum=0.9, epsilon=1e-5, use_bias=True, use_scale=True,
                                                scale_init=jax.nn.initializers.ones,
                                                bias_init=jax.nn.initializers.zeros)
    relu:         Callable = nn.relu
    fc:           nn.Module = functools.partial(nn.Dense, use_bias=True,
                                                kernel_init=jax.nn.initializers.he_normal(),
                                                bias_init=jax.nn.initializers.zeros)
    maxpool: nn.Module = nn.max_pool
    num_planes: int = 16
    num_blocks: Tuple[int] = None
    first_conv: Tuple[int] = None
    first_pool: Tuple[int] = None
    time_emb_dim: int = 32
    hidden_dim: int = 512

    @nn.compact
    def __call__(self, p, x, t, **kwargs):

        # NOTE: it should be False during training, if we use batch normalization...
        use_running_average = kwargs.pop('use_running_average', True)
        if 'use_running_average' in inspect.signature(self.norm).parameters:
            self.norm.keywords['use_running_average'] = use_running_average

        # NOTE: it should be False during training, if we use batch normalization...
        deterministic = kwargs.pop('deterministic', True)
        if 'deterministic' in inspect.signature(self.conv).parameters:
            self.conv.keywords['deterministic'] = deterministic
        if 'deterministic' in inspect.signature(self.fc).parameters:
            self.fc.keywords['deterministic'] = deterministic

        # standardize input images...
        x = x - jnp.reshape(jnp.array(self.pixel_mean, dtype=jnp.float32), (1, 1, 1, -1))
        x = x / jnp.reshape(jnp.array(self.pixel_std, dtype=jnp.float32), (1, 1, 1, -1))

        # specify block structure and widen factor...
        num_planes = self.num_planes
        num_blocks = (
            [(self.depth - 2) // 6,] * 3
        ) if self.num_blocks is None else self.num_blocks
        widen_factor = self.widen_factor

        # define the first layer...
        if not self.first_conv:
            y = self.conv(
                features=num_planes,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding='SAME',
                dtype=self.dtype,
            )(x)
        else:
            K,S,P = self.first_conv
            y = self.conv(
                features=num_planes,
                kernel_size=(K, K),
                strides=(S, S),
                padding=[(P,P),(P,P)],
                dtype=self.dtype,
            )(x)
        # print(f"{y.shape[1]*y.shape[2]*3**2*_y.shape[-1]*y.shape[-1]}")
        y = self.norm(dtype=self.dtype)(y)
        y = self.relu(y)
        if self.first_pool:
            K,S,P=self.first_pool
            y = self.maxpool(
                y,
                window_shape=(K,K),
                strides=(S,S),
                padding=[(P,P),(P,P)]
            )
        self.sow('intermediates', 'feature.layer0', y)

        # add conditions...
        t = timestep_embedding(t, self.time_emb_dim)
        t = self.fc(features=4*self.time_emb_dim)(t)
        t = nn.silu(t)
        
        p = self.fc(features=self.hidden_dim)(p)
        p = self.relu(p)

        # define intermediate layers...
        for layer_idx, num_block in enumerate(num_blocks):
            # add conditions...
            _t = self.fc(features=y.shape[-1])(t)
            _t = nn.silu(_t)
            
            _p = self.fc(features=y.shape[-3]*y.shape[-2])(p)
            _p = self.relu(_p)
            _p = _p[..., None]
            _p = rearrange(_p, 'b (h w) c -> b h w c', h=y.shape[-3], w=y.shape[-2])
            
            y = jnp.concatenate([y, _p], axis=-1) + _t[:, None, None, :]

            _strides = (1,) if layer_idx == 0 else (2,)
            _strides = _strides + (1,) * (num_block - 1)
            for _stride_idx, _stride in enumerate(_strides, start=1):
                _channel = num_planes * (2 ** layer_idx)
                residual = y
                y = self.conv(
                    features=int(_channel * widen_factor),
                    kernel_size=(3, 3),
                    strides=(_stride, _stride),
                    padding='SAME',
                    dtype=self.dtype,
                )(y)
                # print(f"{y.shape[1]*y.shape[2]*3**2*_y.shape[-1]*y.shape[-1]}")
                y = self.norm(dtype=self.dtype)(y)
                y = self.relu(y)
                y = self.conv(
                    features=int(_channel * widen_factor),
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding='SAME',
                    dtype=self.dtype,
                )(y)
                # print(f"{y.shape[1]*y.shape[2]*3**2*_y.shape[-1]*y.shape[-1]}")
                y = self.norm(dtype=self.dtype)(y)
                if residual.shape != y.shape:
                    # NOTE : we use the projection shortcut regardless of the input size,
                    #        which can make a difference compared to He et al. (2016).
                    residual = self.conv(
                        features=int(_channel * widen_factor),
                        kernel_size=(1, 1),
                        strides=(_stride, _stride),
                        padding='SAME',
                        dtype=self.dtype,
                    )(residual)
                    # print(f"{y.shape[1]*y.shape[2]*_y.shape[-1]*y.shape[-1]}")
                    residual = self.norm(dtype=self.dtype)(residual)

                if _stride_idx == len(_strides):
                    self.sow('intermediates',
                             f'pre_relu_feature.layer{layer_idx + 1}', y)
                else:
                    self.sow('intermediates',
                             f'pre_relu_feature.layer{layer_idx + 1}stride{_stride_idx}', y)

                y = self.relu(y + residual)

                if _stride_idx == len(_strides):
                    self.sow('intermediates',
                             f'feature.layer{layer_idx + 1}', y)
                else:
                    self.sow('intermediates',
                             f'feature.layer{layer_idx + 1}stride{_stride_idx}', y)

        y = jnp.mean(y, axis=(1, 2))
        self.sow('intermediates', 'feature.vector', y)

        # return logits if possible
        if self.num_classes:
            y = self.fc(features=self.num_classes, dtype=self.dtype)(y)
            self.sow('intermediates', 'cls.logit', y)

        return y