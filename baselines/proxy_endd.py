import jax
import jax.numpy as jnp
from flax import nnx


class KD(nnx.Module):
    """
    Distilling the Knowledge in a Neural Network
    https://arxiv.org/abs/1503.02531
    """
    def __init__(self, temperature, *, rngs: nnx.Rngs):
        self.temperature = float(temperature)

    def __call__(self, s_logits, t_logits):
        """
        Args:
            s_logits: a real-valued array with shape [B, K].
            t_logits: a real-valued array with shape [M, B, K].
        """
        t_probs = jax.nn.softmax(t_logits / self.temperature, axis=-1)
        s_probs = jax.nn.softmax(s_logits / self.temperature, axis=-1)
        loss = jnp.mean(jnp.sum(
            jax.scipy.special.kl_div(t_probs, s_probs), axis=-1))
        loss = loss * max(self.temperature, self.temperature**2)
        return loss


class LSKD(nnx.Module):
    """
    Logit Standardization in Knowledge Distillation
    https://arxiv.org/abs/2403.01427
    """
    def __init__(self, temperature, eps, *, rngs: nnx.Rngs):
        self.temperature = float(temperature)
        self.eps = float(eps)

    def _normalize(self, logits):
        mean = jnp.mean(logits, axis=-1, keepdims=True)
        stdv = jnp.std(logits, axis=-1, keepdims=True)
        return (logits - mean) / (stdv + self.eps)

    def __call__(self, s_logits, t_logits):
        """
        Args:
            s_logits: a real-valued array with shape [B, K].
            t_logits: a real-valued array with shape [M, B, K].
        """
        t_probs = jax.nn.softmax(
            self._normalize(t_logits) / self.temperature, axis=-1)
        s_probs = jax.nn.softmax(
            self._normalize(s_logits) / self.temperature, axis=-1)
        loss = jnp.mean(jnp.sum(
            jax.scipy.special.kl_div(t_probs, s_probs), axis=-1))
        loss = loss * max(self.temperature, self.temperature**2)
        return loss


class ProxyEnDD(nnx.Module):
    """
    Scaling Ensemble Distribution Distillation to Many Classes with Proxy Targets
    https://arxiv.org/abs/2105.06987
    """
    def __init__(
            self,
            temperature,
            s_offset,
            t_offset,
            dtype,
            eps,
            *,
            rngs: nnx.Rngs
        ):
        self.temperature = float(temperature)
        self.s_offset = float(s_offset)
        self.t_offset = float(t_offset)
        self.dtype = jnp.dtype(dtype)
        self.eps = float(eps)

    def __call__(self, s_logits, t_logits):
        """
        Args:
            s_logits: a real-valued array with shape [B, K].
            t_logits: a real-valued array with shape [M, B, K].
        """
        _max_val = jnp.finfo(self.dtype).max / s_logits.shape[-1] - 1

        s_alphas = jnp.exp(s_logits.astype(self.dtype) / self.temperature)
        s_alphas = jnp.clip(s_alphas + self.s_offset, max=_max_val)
        s_prec = jnp.sum(s_alphas, axis=-1)
        assert s_alphas.dtype == self.dtype
        assert s_prec.dtype == self.dtype

        t_probs = jax.nn.softmax(t_logits / self.temperature, axis=-1)
        t_means = jnp.mean(t_probs, axis=0)
        t_prec = 0.5 * (t_means.shape[-1] - 1) / jnp.sum(t_means * (
            jnp.log(t_means) - jnp.mean(jnp.log(t_probs), axis=0)), axis=1)
        t_alphas = t_means * t_prec[:, None] + self.t_offset
        assert t_alphas.dtype == self.dtype
        assert t_prec.dtype == self.dtype

        prior_term = \
            jnp.sum(jax.lax.lgamma(s_alphas + self.eps), axis=1) \
            - jax.lax.lgamma(s_prec + self.eps) \
            - jnp.sum((s_alphas - 1.0) * (
                jax.lax.digamma(s_alphas + self.eps)
                - jax.lax.digamma(s_prec[:, None] + self.eps)),axis=1)
        prior_term = prior_term / t_prec[:, None]

        recon_term = jnp.negative(jnp.sum(t_means * (
            jax.lax.digamma(s_alphas + self.eps)
            - jax.lax.digamma(s_prec[:, None] + self.eps)), axis=1))

        loss = jnp.mean(recon_term - prior_term)
        return loss