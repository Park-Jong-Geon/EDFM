# MLP based model architecture

from typing import Any
import math
from einops import rearrange
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


class Mlp(nn.Module):
    hidden_size: int
    time_embed_dim: int
    num_blocks: int
    num_classes: int
    droprate: float
    time_scale: float
    dtype: Any = jnp.float32
    
    @nn.compact
    def __call__(self, x, z, t, **kwargs):        
        x = nn.Dense(
                features=self.hidden_size,
                dtype=self.dtype,
                kernel_init=nn.initializers.xavier_uniform(),
                bias_init=nn.initializers.normal(stddev=1e-6)
            )(x)
        x = nn.gelu(x)
        x = x[:, None, :]
        
        z = z.reshape((z.shape[0], -1, z.shape[-1]))
        z = nn.Dense(
                features=self.hidden_size,
                dtype=self.dtype,
                kernel_init=nn.initializers.xavier_uniform(),
                bias_init=nn.initializers.normal(stddev=1e-6)
            )(z)
        z = nn.gelu(z)
        
        x = jnp.concatenate([x, z], axis=1)

        t_skip = timestep_embedding(self.time_scale * t, self.time_embed_dim)

        # MLP Residual.
        for _ in range(self.num_blocks):
            x_skip = x
            
            t = nn.Dense(3 * self.hidden_size, kernel_init=nn.initializers.constant(0.))(t_skip)
            t = nn.silu(t)
            shift_mlp, scale_mlp, gate_mlp = jnp.split(t, 3, axis=-1)
            
            x = nn.LayerNorm(use_bias=False, use_scale=False)(x)
            x = x * (1 + scale_mlp[:, None, :]) + shift_mlp[:, None, :]
            x = nn.Dense(
                    features=self.hidden_size,
                    dtype=self.dtype,
                    kernel_init=nn.initializers.xavier_uniform(),
                    bias_init=nn.initializers.normal(stddev=1e-6)
                )(x)
            x = nn.gelu(x)
            x = jnp.swapaxes(x, -2, -1)
            
            x = nn.Dense(
                    features=x.shape[-1],
                    dtype=self.dtype,
                    kernel_init=nn.initializers.xavier_uniform(),
                    bias_init=nn.initializers.normal(stddev=1e-6)
                )(x)
            x = nn.gelu(x)
            x = jnp.swapaxes(x, -2, -1)
            
            x = x_skip + (gate_mlp[:, None, :] * x)
            x = nn.Dropout(rate=self.droprate)(x, deterministic=not kwargs["training"])
            
        x = x.mean(axis=1)
        x = nn.Dense(
                features=self.num_classes,
                dtype=self.dtype,
                kernel_init=nn.initializers.xavier_uniform(),
                bias_init=nn.initializers.normal(stddev=1e-6)
            )(x)
        return x