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

from typing import Callable, Optional
import functools
import flax.linen as nn
import jax.numpy as jnp


class MlpBlock(nn.Module):
  mlp_dim: int

  act: Callable = nn.gelu
  fc:  nn.Module = functools.partial(nn.Dense,
                                     kernel_init=nn.initializers.he_normal(),
                                     bias_init=nn.initializers.zeros)

  @nn.compact
  def __call__(self, x):
    y = self.fc(self.mlp_dim)(x)
    y = self.act(y)
    return self.fc(x.shape[-1])(y)


class MixerBlock(nn.Module):
  """Mixer block layer."""
  tokens_mlp_dim: int
  channels_mlp_dim: int
  
  norm:         nn.Module = nn.LayerNorm
  
  @nn.compact
  def __call__(self, x):
    y = self.norm()(x)
    y = jnp.swapaxes(y, 1, 2)
    y = MlpBlock(self.tokens_mlp_dim, name='token_mixing')(y)
    y = jnp.swapaxes(y, 1, 2)
    x = x + y
    y = self.norm()(x)
    return x + MlpBlock(self.channels_mlp_dim, name='channel_mixing')(y)


class MlpMixer(nn.Module):
  """Mixer architecture."""
  num_classes: int
  num_blocks: int
  tokens_mlp_dim: int
  channels_mlp_dim: int
  model_name: Optional[str] = None
  droprate: float = 0.1
  
  act: Callable = nn.gelu
  fc:  nn.Module = functools.partial(nn.Dense,
                                     kernel_init=nn.initializers.he_normal(),
                                     bias_init=nn.initializers.zeros)
  norm:         nn.Module = nn.LayerNorm
  dropout:      nn.Module = nn.Dropout

  @nn.compact
  def __call__(self, x, t, train):
    for _ in range(self.num_blocks):
      x += self.act(self.fc(x.shape[-1])(t))[:, None, :]
      x = MixerBlock(self.tokens_mlp_dim, self.channels_mlp_dim)(x)
      x = self.dropout(self.droprate)(x, deterministic=not train)
      
    x = self.norm(name='pre_head_layer_norm')(x)
    x = jnp.mean(x, axis=1)
    
    if self.num_classes:
      x = nn.Dense(self.num_classes,
                   kernel_init=nn.initializers.zeros,
                   name='head')(x)
    return x