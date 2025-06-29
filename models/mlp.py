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
    s_data: float = 1.
    s_noise: float = 4.
    
    @nn.compact
    def __call__(self, x, z, t, **kwargs):
        x_copy = x
        
        c_in = 1 / jnp.sqrt(t**2 * self.s_data**2 + (1-t)**2 * self.s_noise**2)
        c_skip = (t * self.s_data**2 - (1-t) * self.s_noise**2) * c_in**2
        c_out = self.s_data * self.s_noise * c_in

        x *= c_in[..., None]
        t = jnp.log(self.time_scale * (1-t) + 1e-12) / 4
        
        z = jnp.mean(z, axis=(1, 2))
        x = jnp.concatenate([x, z], axis=-1)

        t_skip = timestep_embedding(t, self.time_embed_dim)

        # MLP Residual.
        for i in range(self.num_blocks):
            x_skip = x
            
            t = nn.Dense(3 * x.shape[-1], kernel_init=nn.initializers.constant(0.))(t_skip)
            t = nn.silu(t)
            shift_mlp, scale_mlp, gate_mlp = jnp.split(t, 3, axis=-1)
            
            x = nn.LayerNorm(use_bias=False, use_scale=False)(x)
            x = x * (1 + scale_mlp) + shift_mlp
            x = nn.Dense(
                    features=self.hidden_size,
                    dtype=self.dtype,
                    kernel_init=nn.initializers.xavier_uniform(),
                    bias_init=nn.initializers.normal(stddev=1e-6)
                )(x)
            x = nn.gelu(x)
            x = x_skip + (gate_mlp * x) if i > 0 else x
            x = nn.Dropout(rate=self.droprate)(x, deterministic=not kwargs["training"])
            
        x = nn.Dense(
                features=self.num_classes,
                dtype=self.dtype,
                kernel_init=nn.initializers.xavier_uniform(),
                bias_init=nn.initializers.normal(stddev=1e-6)
            )(x)
        
        return c_skip[..., None] * x_copy + c_out[..., None] * x