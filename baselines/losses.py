import jax
import jax.numpy as jnp
import flax.linen as nn

from jax import lax
from jax._src import custom_derivatives
from jax._src.typing import Array, ArrayLike
from jax._src.numpy.util import _promote_args_inexact
from jax._src.lax.lax import _const as _lax_const

@custom_derivatives.custom_jvp
def xlogy(x: ArrayLike, y: ArrayLike) -> Array:
  """Compute x*log(y), returning 0 for x=0.

  JAX implementation of :obj:`scipy.special.xlogy`.

  This is defined to return zero when :math:`(x, y) = (0, 0)`, with a custom
  derivative rule so that automatic differentiation is well-defined at this point.

  Args:
    x: arraylike, real-valued.
    y: arraylike, real-valued.

  Returns:
    array containing xlogy values.

  See also:
    :func:`jax.scipy.special.xlog1py`
  """
  # Note: xlogy(0, 0) should return 0 according to the function documentation.
  x, y = _promote_args_inexact("xlogy", x, y)
  x_ok = x != 0.
  return jnp.where(x_ok, lax.mul(x, lax.log(y)), jnp.zeros_like(x))

def _xlogy_jvp(primals, tangents):
  (x, y) = primals
  (x_dot, y_dot) = tangents
  result = xlogy(x, y)
  return result, (x_dot * lax.log(y) + y_dot * x / y).astype(result.dtype)
xlogy.defjvp(_xlogy_jvp)

@custom_derivatives.custom_jvp
def _xlogx(x):
  """Compute x log(x) with well-defined derivatives."""
  return xlogy(x, x)

def _xlogx_jvp(primals, tangents):
  x, = primals
  x_dot, = tangents
  return  _xlogx(x), x_dot * (lax.log(x) + 1)
_xlogx.defjvp(_xlogx_jvp)

def kl_div(
    p: ArrayLike,
    q: ArrayLike,
) -> Array:
  r"""The Kullback-Leibler divergence.

  JAX implementation of :obj:`scipy.special.kl_div`.

  .. math::

     \mathrm{kl\_div}(p, q) = \begin{cases}
       p\log(p/q)-p+q & p>0,q>0\\
       q & p=0,q\ge 0\\
       \infty & \mathrm{otherwise}
    \end{cases}

  Args:
    p: arraylike, real-valued.
    q: arraylike, real-valued.

  Returns:
    array of KL-divergence values

  See also:
    - :func:`jax.scipy.special.entr`
    - :func:`jax.scipy.special.rel_entr`
  """
  p, q = _promote_args_inexact("kl_div", p, q)
  return rel_entr(p, q) - p + q


def rel_entr(
    p: ArrayLike,
    q: ArrayLike,
) -> Array:
  r"""The relative entropy function.

  JAX implementation of :obj:`scipy.special.rel_entr`.

  .. math::

     \mathrm{rel\_entr}(p, q) = \begin{cases}
       p\log(p/q) & p>0,q>0\\
       0 & p=0,q\ge 0\\
       \infty & \mathrm{otherwise}
    \end{cases}

  Args:
    p: arraylike, real-valued.
    q: arraylike, real-valued.

  Returns:
    array of relative entropy values.

  See also:
    - :func:`jax.scipy.special.entr`
    - :func:`jax.scipy.special.kl_div`
  """
  p, q = _promote_args_inexact("rel_entr", p, q)
  zero = _lax_const(p, 0.0)
  both_gt_zero_mask = lax.bitwise_and(lax.gt(p, zero), lax.gt(q, zero))
  one_zero_mask = lax.bitwise_and(lax.eq(p, zero), lax.ge(q, zero))

  safe_p = jnp.where(both_gt_zero_mask, p, 1)
  safe_q = jnp.where(both_gt_zero_mask, q, 1)
  log_val = lax.sub(_xlogx(safe_p), xlogy(safe_p, safe_q))
  result = jnp.where(
      both_gt_zero_mask, log_val, jnp.where(one_zero_mask, zero, jnp.inf)
  )
  return result

class KD(nn.Module):
    """
    Distilling the Knowledge in a Neural Network
    https://arxiv.org/abs/1503.02531
    """
    def __init__(self, temperature):
        self.temperature = float(temperature)

    def __call__(self, s_logits, t_logits):
        """
        Args:
            s_logits: a real-valued array with shape [B, K].
            t_logits: a real-valued array with shape [M, B, K].
        """
        t_probs = jax.nn.softmax(t_logits / self.temperature, axis=-1)
        s_probs = jax.nn.softmax(s_logits / self.temperature, axis=-1)
        # loss = jnp.mean(jnp.sum(
        #     jax.scipy.special.kl_div(t_probs, s_probs), axis=-1))
        loss = jnp.mean(jnp.sum(kl_div(t_probs, s_probs), axis=-1))
        loss = loss * max(self.temperature, self.temperature**2)
        return loss


class LSKD(nn.Module):
    """
    Logit Standardization in Knowledge Distillation
    https://arxiv.org/abs/2403.01427
    """
    def __init__(self, temperature, eps):
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


class ProxyEnDD(nn.Module):
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
        # _max_val = jnp.finfo(self.dtype).max / s_logits.shape[-1] - 1

        s_alphas = jnp.exp(s_logits.astype(self.dtype) / self.temperature)
        s_alphas += self.s_offset
        # s_alphas = jnp.clip(s_alphas + self.s_offset, max=_max_val)
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
      
      
class MMD:
    def __init__(self, bandwidth_range=[2.0], kernel='multiscale', return_matrix=False):
        self.bandwidth_range = bandwidth_range
        self.kernel = kernel
        self.return_matrix = return_matrix

    def __call__(self, x, y):
        """
        Empirical maximum mean discrepancy. The lower the result, the more evidence that distributions are the same.

        Args:
            x: first sample, distribution P
            y: second sample, distribution Q
            kernel: kernel type such as "multiscale" or "rbf"
        """
        xx = jnp.dot(x, x.T)
        yy = jnp.dot(y, y.T)
        zz = jnp.dot(x, y.T)
        
        rx = jnp.expand_dims(jnp.diag(xx), axis=0).repeat(xx.shape[0], axis=0)
        ry = jnp.expand_dims(jnp.diag(yy), axis=0).repeat(yy.shape[0], axis=0)
        
        dxx = rx.T + rx - 2. * xx  # Used for A in (1)
        dyy = ry.T + ry - 2. * yy  # Used for B in (1)
        dxy = rx.T + ry - 2. * zz  # Used for C in (1)
        
        XX = jnp.zeros_like(xx)
        YY = jnp.zeros_like(xx)
        XY = jnp.zeros_like(xx)
        
        if self.kernel == "multiscale":
            for a in self.bandwidth_range:
                XX += a**2 * (a**2 + dxx)**-1
                YY += a**2 * (a**2 + dyy)**-1
                XY += a**2 * (a**2 + dxy)**-1
        
        elif self.kernel == "rbf":
            for a in self.bandwidth_range:
                XX += jnp.exp(-0.5 * dxx / a)
                YY += jnp.exp(-0.5 * dyy / a)
                XY += jnp.exp(-0.5 * dxy / a)
        
        elif self.kernel == 'linear':
            XX += xx
            YY += yy
            XY += zz
        
        if self.return_matrix:
            return jnp.mean(XX + YY - 2. * XY), XX, XY, YY
        
        return jnp.mean(XX + YY - 2. * XY)