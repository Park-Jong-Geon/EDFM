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
    emb_dim: int = 32

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
        # m = self.variable('image_stats', 'm', lambda _: jnp.array(
        #     self.pixel_mean, dtype=jnp.float32), (x.shape[-1],))
        # s = self.variable('image_stats', 's', lambda _: jnp.array(
        #     self.pixel_std, dtype=jnp.float32), (x.shape[-1],))
        # x = x - jnp.reshape(m.value, (1, 1, 1, -1))
        # x = x / jnp.reshape(s.value, (1, 1, 1, -1))

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
        t = timestep_embedding(t, self.emb_dim)
        t = self.fc(features=2*self.emb_dim)(t)
        t = self.relu(t)

        p = self.fc(features=self.emb_dim)(p)
        p = self.relu(p)
        p = self.fc(features=1)(p)
        p = self.relu(p)
        p = timestep_embedding(p, self.emb_dim)
        p = self.fc(features=2*self.emb_dim)(p)
        p = self.relu(p)

        # define intermediate layers...
        for layer_idx, num_block in enumerate(num_blocks):
            # add conditions...
            _t = self.fc(features=y.shape[-1])(t)
            _t = self.relu(_t)
            
            _p = self.fc(features=y.shape[-1])(p)
            _p = self.relu(_p)
            
            y += _p[:, None, None, :] + _t[:, None, None, :]

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


class GaussianFourierProjection(nn.Module):
    """Gaussian Fourier embeddings for noise levels."""
    embedding_size: int = 256
    scale: float = 1.0

    @nn.compact
    def __call__(self, x):
        W = self.param(
            "W", jax.nn.initializers.normal(stddev=self.scale), (self.embedding_size,)
        )
        W = jax.lax.stop_gradient(W)
        x_proj = x[:, None] * W[None, :] * 2 * jnp.pi
        return jnp.concatenate([jnp.sin(x_proj), jnp.cos(x_proj)], axis=-1)
    

class MlpBridge(nn.Module):
    emb_dim: int = 256
    num_blocks: int = 32
    fourier_scale: float = 1.
    act:          Callable = nn.gelu
    fc:           nn.Module = functools.partial(nn.Dense, use_bias=True,
                                      kernel_init=jax.nn.initializers.he_normal(),
                                      bias_init=jax.nn.initializers.zeros)
    
    @nn.compact
    def __call__(self, p, c, t, **kwargs):
        # Probability embedding
        p_emb = self.fc(self.emb_dim)(p)
        p_emb = self.act(p_emb)
        p_emb = self.fc(self.emb_dim)(p_emb)
        p_emb = self.act(p_emb)
        
        # Timestep embedding; Gaussian Fourier features embeddings
        t_emb = GaussianFourierProjection(embedding_size=self.emb_dim, scale=self.fourier_scale)(t)
        t_emb = self.fc(self.emb_dim)(t_emb)
        t_emb = self.act(t_emb)
        t_emb = self.fc(self.emb_dim)(t_emb)
        t_emb = self.act(t_emb)
        
        # Image features embedding
        z_emb = self.fc(self.emb_dim)(c)
        z_emb = self.act(z_emb)
        z_emb = self.fc(self.emb_dim)(z_emb)
        z_emb = self.act(z_emb)

        emb = jnp.concatenate([z_emb, t_emb], axis=-1)
        h = MlpMixer(out_dim=p.shape[-1], 
                     num_blocks=self.num_blocks, 
                     mlp_dim=self.emb_dim)(p_emb, emb)
        return h


def to_norm_kwargs(norm, kwargs):
    return dict(
        use_running_average=not kwargs["training"]
    ) if getattr(norm, "func", norm) is nn.BatchNorm else dict()


class InvertedResidual(nn.Module):
    ch: int
    st: int
    expand: int
    conv: nn.Module = nn.Conv
    # norm: nn.Module = functools.partial(nn.GroupNorm, num_groups=16)
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
    # norm:        nn.Module = functools.partial(nn.GroupNorm, num_groups=16, epsilon=1e-5, use_bias=True, use_scale=True,
    #                                   scale_init=jax.nn.initializers.ones,
    #                                   bias_init=jax.nn.initializers.zeros)
    norm:         nn.Module = functools.partial(nn.BatchNorm, momentum=0.9, epsilon=1e-5, use_bias=True, use_scale=True,
                                      scale_init=jax.nn.initializers.ones,
                                      bias_init=jax.nn.initializers.zeros)                
    silu:         Callable = nn.silu  # activation for time embedding
    relu6:        Callable = nn.activation.relu6
    fc:           nn.Module = functools.partial(nn.Dense, use_bias=True,
                                      kernel_init=jax.nn.initializers.he_normal(),
                                      bias_init=jax.nn.initializers.zeros)
    droprate: float = 0
    dropout: nn.Module = functools.partial(nn.Dropout, deterministic=False)
    cfgs: Sequence[Tuple[int, int, int, int]] = (
        # e, c, n, s
        (1, 32, 1, 1),
        (3, 48, 2, 2),
        (3, 64, 3, 2),
        (3, 96, 2, 1),
        (3, 128, 2, 1),
    )

    @nn.compact
    def __call__(self, p, x, t, **kwargs):
        norm_kwargs = to_norm_kwargs(self.norm, kwargs)
        
        # Encode t
        t = timestep_embedding(t, self.ch//4)
        t = self.fc(features=self.ch)(t)
        t = self.silu(t)
        t = self.fc(features=self.ch)(t)
        t = self.silu(t)

        # Encode p; upsample
        p = self.fc(features=x.shape[1]*x.shape[2])(p)
        p = self.relu6(p)
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
        l_t, t, next_l_t = self.forward(rng, l0, c)
        # next_l_t = jax.nn.softmax(next_l_t)
        eps = self.score(l_t, z, t, **kwargs)
        return eps, next_l_t

    def forward(self, rng, l_label, c):
        # Sample t
        t_rng, n_rng = jax.random.split(rng, 2)
        t = jax.random.uniform(t_rng, (l_label.shape[0],), maxval=1-self.eps)  # (B,)
        # u = jax.random.uniform(t_rng, (l_label.shape[0],),)
        # t = jnp.log(1 + (self.alpha**(1-self.eps) - 1) * u) / jnp.log(self.alpha)

        # Sample noise
        z = jax.random.normal(n_rng, l_label.shape)
        _t = t[:, None]
        x_t = _t * l_label + (1-_t) * self.var * z

        # Compute diff
        u_t = (l_label - x_t) / (1-_t)
        
        # next_x_t = x_t + (1 / self.steps) * u_t
        return x_t, t, u_t #next_x_t

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
