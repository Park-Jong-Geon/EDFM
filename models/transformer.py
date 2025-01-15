# Transformer based model architecture
# The code was largely adopted from "https://github.com/kvfrans/jax-diffusion-transformer"

import math
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Tuple, Callable, Optional

Array = Any
PRNGKey = Any
Shape = Tuple[int]
Dtype = Any

# Port of https://github.com/facebookresearch/DiT/blob/main/models.py into jax.

#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    hidden_size: int
    frequency_embedding_size: int = 256

    @nn.compact
    def __call__(self, t):
        x = self.timestep_embedding(t)
        x = nn.Dense(self.hidden_size, kernel_init=nn.initializers.normal(0.02))(x)
        x = nn.silu(x)
        x = nn.Dense(self.hidden_size, kernel_init=nn.initializers.normal(0.02))(x)
        return x

    # t is between [0, max_period]. It's the INTEGER timestep, not the fractional (0,1).;
    def timestep_embedding(self, t, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        t = jax.lax.convert_element_type(t, jnp.float32)
        dim = self.frequency_embedding_size
        half = dim // 2
        freqs = jnp.exp( -math.log(max_period) * jnp.arange(start=0, stop=half, dtype=jnp.float32) / half)
        args = t[:, None] * freqs[None]
        embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
        return embedding

class MlpBlock(nn.Module):
    """Transformer MLP / feed-forward block."""

    mlp_dim: int
    dtype: Dtype = jnp.float32
    out_dim: Optional[int] = None
    dropout_rate: float = None
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.xavier_uniform()
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.normal(stddev=1e-6)

    @nn.compact
    def __call__(self, inputs):
        """It's just an MLP, so the input shape is (batch, len, emb)."""
        actual_out_dim = inputs.shape[-1] if self.out_dim is None else self.out_dim
        x = nn.Dense(
                features=self.mlp_dim,
                dtype=self.dtype,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init)(inputs)
        x = nn.gelu(x)
        # x = nn.Dropout(rate=self.dropout_rate)(x)
        output = nn.Dense(
                features=actual_out_dim,
                dtype=self.dtype,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init)(x)
        # output = nn.Dropout(rate=self.dropout_rate)(output)
        return output
    
def modulate(x, shift, scale):
    return x * (1 + scale[:, None]) + shift[:, None]
    
################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    hidden_size: int
    num_heads: int
    mlp_ratio: float = 4.0

    @nn.compact
    def __call__(self, x, z, t):
        # Calculate adaLn modulation parameters.
        t = nn.silu(t)
        t = nn.Dense(9 * self.hidden_size, kernel_init=nn.initializers.constant(0.))(t)
        shift_ca, scale_ca, gate_ca, shift_sa, scale_sa, gate_sa, shift_mlp, scale_mlp, gate_mlp = jnp.split(t, 9, axis=-1)

        # Cross Attention Residual.
        x_norm = nn.LayerNorm(use_bias=False, use_scale=False)(x)
        z_norm = nn.LayerNorm(use_bias=False, use_scale=False)(z)
        x_modulated = modulate(x_norm, shift_ca, scale_ca)
        z_modulated = modulate(z_norm, shift_ca, scale_ca)
        attn_x = nn.MultiHeadDotProductAttention(kernel_init=nn.initializers.xavier_uniform(),
            num_heads=self.num_heads)(x_modulated, z_modulated)
        x = x + (gate_ca[:, None] * attn_x)

        # Self Attention Residual.
        x_norm = nn.LayerNorm(use_bias=False, use_scale=False)(x)
        x_modulated = modulate(x_norm, shift_sa, scale_sa)
        attn_x = nn.MultiHeadDotProductAttention(kernel_init=nn.initializers.xavier_uniform(),
            num_heads=self.num_heads)(x_modulated, x_modulated)
        x = x + (gate_sa[:, None] * attn_x)

        # MLP Residual.
        x_norm = nn.LayerNorm(use_bias=False, use_scale=False)(x)
        x_modulated = modulate(x_norm, shift_mlp, scale_mlp)
        mlp_x = MlpBlock(mlp_dim=int(self.hidden_size * self.mlp_ratio))(x_modulated)
        x = x + (gate_mlp[:, None] * mlp_x)
        return x
    
class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    out_channels: int
    hidden_size: int

    @nn.compact
    def __call__(self, x, c):
        c = nn.silu(c)
        c = nn.Dense(2 * self.hidden_size, kernel_init=nn.initializers.constant(0))(c)
        shift, scale = jnp.split(c, 2, axis=-1)
        x = modulate(nn.LayerNorm(use_bias=False, use_scale=False)(x), shift, scale)
        x = nn.Dense(self.out_channels, kernel_init=nn.initializers.constant(0))(x)
        return x

class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    hidden_size: int
    depth: int
    num_heads: int
    mlp_ratio: float
    droprate: float = 0.

    @nn.compact
    def __call__(self, x, z, t, training=False):
        batch_size = x.shape[0]
        num_classes = x.shape[1]
        x = nn.gelu(MlpBlock(mlp_dim=self.hidden_size, out_dim=self.hidden_size)(x))
        z = jnp.mean(z, axis=(1, 2))
        z = nn.gelu(MlpBlock(mlp_dim=self.hidden_size, out_dim=self.hidden_size)(z))
        x = x[:, None, :] # Embedding length is 1.
        z = z[:, None, :]
        t = TimestepEmbedder(self.hidden_size)(t) # (B, hidden_size)
        for _ in range(self.depth):
            x = DiTBlock(self.hidden_size, self.num_heads, self.mlp_ratio)(x, z, t)
        x = FinalLayer(num_classes, self.hidden_size)(x, t)
        x = x.squeeze(1)
        assert x.shape == (batch_size, num_classes)
        return x