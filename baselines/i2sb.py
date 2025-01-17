import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence
import functools

class FlowMatching(nn.Module):
    res_net: Sequence[nn.Module]
    score_net: Sequence[nn.Module]
    var: float
    num_classes: int
    eps: float
    train_timestep_alpha: float

    def setup(self):
        self.resnet = self.res_net()
        self.score = self.score_net()

    def __call__(self, *args, **kwargs):
        return self.conditional_dbn(*args, **kwargs)

    def conditional_dbn(self, rng, l0, x, **kwargs):
        z, c = self.resnet(x, **kwargs)        
        l_t, t, u_t = self.forward(rng, l0, c)
        eps = self.score(l_t, z, t, **kwargs)
        x_t = l_t + (1-t[:, None]) * eps
        
        return eps, u_t, x_t

    def forward(self, rng, l_label, c):
        # Sample t
        t_rng, n_rng = jax.random.split(rng, 2)
        u = jax.random.uniform(t_rng, (l_label.shape[0],),)
        t = jnp.log(1 + (self.train_timestep_alpha**(1-self.eps) - 1) * u) / jnp.log(self.train_timestep_alpha)

        # Sample noise
        z = jax.random.normal(n_rng, l_label.shape)
        _t = t[:, None]
        x_t = _t * l_label + (1-_t) * self.var * z

        # Compute diff
        u_t = (l_label - x_t) / (1-_t)
        
        return x_t, t, u_t

    def forward(self, rng, l_label, c):
        n_T = dsb_stats["n_T"]
        _sigma_weight_t = dsb_stats["sigma_weight_t"]
        _sigma_t = dsb_stats["sigma_t"]
        _bigsigma_t = dsb_stats["bigsigma_t"]

        t_rng, n_rng = jax.random.split(rng, 2)

        if _ts is None:
            _ts = jax.random.randint(t_rng, (x0.shape[0],), 1, n_T)  # (B,)
        sigma_weight_t = _sigma_weight_t[_ts]  # (B,)
        sigma_weight_t = expand_to_broadcast(sigma_weight_t, x0, axis=1)
        sigma_t = _sigma_t[_ts]
        mu_t = (sigma_weight_t*x0+(1-sigma_weight_t)*x1)
        bigsigma_t = _bigsigma_t[_ts]  # (B,)
        bigsigma_t = expand_to_broadcast(bigsigma_t, mu_t, axis=1)

        # q(X_t|X_0,X_1) = N(X_t;mu_t,bigsigma_t)
        noise = jax.random.normal(n_rng, mu_t.shape)  # (B, d)
        x_t = mu_t + noise*jnp.sqrt(bigsigma_t)
        t = _ts/n_T
        return x_t, t, mu_t, sigma_t, _ts

    def sample(self, *args, **kwargs):
        return self.conditional_sample(*args, **kwargs)

    def conditional_sample(self, rng, sampler, x, num_models):
        z, _ = self.resnet(x, training=False)
        z = z.repeat(num_models, axis=0)
        
        init_logit = self.var * jax.random.normal(rng, (z.shape[0], self.num_classes))
        logit, val = sampler(
            functools.partial(self.score, training=False), init_logit, z)
        return logit, val
