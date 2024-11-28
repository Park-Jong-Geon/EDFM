# Copyright 2024 Google LLC.
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

from typing import Any, Optional

import einops
import flax.linen as nn
import jax.numpy as jnp


class MlpBlock(nn.Module):
  mlp_dim: int

  @nn.compact
  def __call__(self, x):
    y = nn.Dense(self.mlp_dim)(x)
    y = nn.gelu(y)
    return nn.Dense(x.shape[-1])(y)


class MixerBlock(nn.Module):
  """Mixer block layer."""
  mlp_dim: int

  @nn.compact
  def __call__(self, x):
    y = nn.LayerNorm()(x)
    y = MlpBlock(self.mlp_dim)(y)
    x = x + y
    y = nn.LayerNorm()(x)
    return x + MlpBlock(self.mlp_dim)(y)


class MlpMixer(nn.Module):
  """Mixer architecture."""
  out_dim: int
  num_blocks: int
  mlp_dim: int

  @nn.compact
  def __call__(self, x, c=None):
    for _ in range(self.num_blocks):
      if c is not None:
        x = jnp.concatenate([x, c], axis=-1)
      x = MixerBlock(self.mlp_dim)(x)
    x = nn.LayerNorm()(x)
    x = nn.Dense(self.out_dim, kernel_init=nn.initializers.zeros)(x)
    return x