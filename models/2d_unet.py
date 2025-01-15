# 2D-convolution based U-Net architecture
# Code was largely adopted from "https://github.com/kim-hyunsu/dbn/blob/main/models/i2sb.py"

import math
import jax
import jax.numpy as jnp
import flax.linen as nn
import functools
from einops import rearrange
from typing import Callable, Sequence, Tuple

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


def to_norm_kwargs(norm, kwargs):
    return dict(
        use_running_average=not kwargs["training"]
    ) if getattr(norm, "func", norm) is nn.BatchNorm else dict()


class InvertedResidual(nn.Module):
    ch: int
    st: int
    expand: int
    conv: nn.Module = nn.Conv
    norm: nn.Module = nn.BatchNorm
    relu6: Callable = nn.activation.relu6
    fc: nn.Module = nn.Dense

    @nn.compact
    def __call__(self, x, **kwargs):
        residual = x
        norm_kwargs = to_norm_kwargs(self.norm, kwargs)
        in_ch = x.shape[-1]
        hidden_dim = in_ch*self.expand
        identity = self.st == 1 and in_ch == self.ch
        if self.expand == 1:
            x = self.conv(
                features=hidden_dim,
                kernel_size=(3, 3),
                strides=(self.st, self.st),
                padding=1,
                feature_group_count=hidden_dim,
                use_bias=False
            )(x)
            x = self.norm(**norm_kwargs)(x)
            x = self.relu6(x)
            x = self.conv(
                features=self.ch,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding=0,
                use_bias=False
            )(x)
            x = self.norm(**norm_kwargs)(x)
        else:
            x = self.conv(
                features=hidden_dim,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding=0,
                use_bias=False
            )(x)
            x = self.norm(**norm_kwargs)(x)
            x = self.relu6(x)
            x = self.conv(
                features=hidden_dim,
                kernel_size=(3, 3),
                strides=(self.st, self.st),
                padding=1,
                feature_group_count=hidden_dim,
                use_bias=False
            )(x)
            x = self.norm(**norm_kwargs)(x)
            x = self.relu6(x)
            x = self.conv(
                features=self.ch,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding=0,
                use_bias=False
            )(x)
            x = self.norm(**norm_kwargs)(x)
        if identity:
            return residual + x
        else:
            return x


class ClsUnet(nn.Module):
    num_classes: int = 10
    ch: int = 128
    conv:         nn.Module = functools.partial(nn.Conv, use_bias=False,
                                      kernel_init=jax.nn.initializers.he_normal(),
                                      bias_init=jax.nn.initializers.zeros)
    norm:         nn.Module = functools.partial(nn.BatchNorm, momentum=0.9, epsilon=1e-5, use_bias=True, use_scale=True,
                                      scale_init=jax.nn.initializers.ones,
                                      bias_init=jax.nn.initializers.zeros)                
    silu:         Callable = nn.silu  # activation for time embedding
    relu6:        Callable = nn.activation.relu6
    fc:           nn.Module = functools.partial(nn.Dense, use_bias=True,
                                      kernel_init=jax.nn.initializers.he_normal(),
                                      bias_init=jax.nn.initializers.zeros)
    droprate: float = 0
    dropout: nn.Module = nn.Dropout
    cfgs: Sequence[Tuple[int, int, int, int]] = (
        # e, c, n, s
        (1, 32, 1, 1),
        (3, 48, 2, 2),
        (3, 64, 3, 2),
        (3, 96, 2, 1),
        (3, 128, 2, 1),
    )
    new_dim: int = 16

    @nn.compact
    def __call__(self, p, x, t, **kwargs):
        norm_kwargs = to_norm_kwargs(self.norm, kwargs)
        
        # Encode t
        t = timestep_embedding(t, self.ch//4)
        for _ in range(2):
            t = self.fc(features=self.ch)(t)
            t = self.silu(t)

        # Encode p; upsample
        for _ in range(2):
            p = self.fc(features=x.shape[1]*x.shape[2])(p)
            p = self.relu6(p)
        p = p.reshape((p.shape[0], x.shape[1], x.shape[2], 1))
        p = self.conv(
            features=self.ch//2,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME"
        )(p)
        p = self.relu6(p)
        p = self.norm(**norm_kwargs)(p)

        # Encode x
        x = self.conv(
            features=self.ch//2,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME"
        )(x)
        x = self.relu6(x)
        x = self.norm(**norm_kwargs)(x)

        # Concatenate p and x
        z = jnp.concatenate([p, x], axis=-1)

        # U-Net
        for e, c, n, s in self.cfgs:
            _t = self.fc(features=z.shape[-1])(t)
            _t = self.silu(_t)
            _t = _t[:, None, None, :]
            z += _t
            for i in range(n):
                z = InvertedResidual(
                    ch=c,
                    st=s if i == 0 else 1,
                    expand=e
                )(z, **kwargs)
        
        z = self.dropout(rate=self.droprate)(z, deterministic=not kwargs["training"])
                
        # Post process
        z = self.conv(
            features=self.ch,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding=0,
            use_bias=False
        )(z)
        z = self.norm(**norm_kwargs)(z)
        z = self.relu6(z)

        z = jnp.mean(z, axis=(1, 2))
        z = self.fc(features=self.num_classes)(z)
        return z