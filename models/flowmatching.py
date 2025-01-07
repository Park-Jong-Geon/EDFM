from typing import Any, Callable, Sequence, Tuple
import functools
import inspect
import math
from einops import rearrange
import jax
import jax.numpy as jnp
import flax.linen as nn

from .mlp_mixer import MlpMixer

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
        z = jnp.mean(z, axis=(1, 2))
        x = jnp.concatenate([x, z], axis=-1)

        t_skip = timestep_embedding(self.time_scale * t, self.time_embed_dim)

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
        return x
    

class FlowMatching(nn.Module):
    res_net: Sequence[nn.Module]
    score_net: Sequence[nn.Module]
    steps: float = 100.
    var: float = 0.2
    num_classes: int = 10
    eps: float = 1e-4
    alpha: float = 1.2

    def setup(self):
        self.resnet = self.res_net()
        self.score = self.score_net()

    def __call__(self, *args, **kwargs):
        return self.conditional_dbn(*args, **kwargs)

    def conditional_dbn(self, rng, l0, x, **kwargs):
        z, c = self.resnet(x, **kwargs)        
        l_t, t, u_t, l_1 = self.forward(rng, l0, c)
        eps = self.score(l_t, z, t, **kwargs)
        x_t = l_t + (1-t[:, None]) * eps
        p_1 = jax.nn.softmax(l_1)
        return eps, u_t, p_1, x_t

    def forward(self, rng, l_label, c):
        # Sample t
        t_rng, n_rng = jax.random.split(rng, 2)
        u = jax.random.uniform(t_rng, (l_label.shape[0],),)
        t = jnp.log(1 + (self.alpha**(1-self.eps) - 1) * u) / jnp.log(self.alpha)

        # Sample noise
        z = jax.random.normal(n_rng, l_label.shape)
        _t = t[:, None]
        x_t = _t * l_label + (1-_t) * self.var * z

        # Compute diff
        u_t = (l_label - x_t) / (1-_t)
        
        return x_t, t, u_t, l_label

    def sample(self, *args, **kwargs):
        return self.conditional_sample(*args, **kwargs)

    def conditional_sample(self, rng, sampler, x, num_models):
        z, c = self.resnet(x, training=False)
        z = z.repeat(num_models, axis=0)
        c = c.repeat(num_models, axis=0)
        init_logit = self.var * jax.random.normal(rng, (z.shape[0], self.num_classes))
        logit, val = sampler(
            functools.partial(self.score, training=False), init_logit, z, c)
        return logit, val
