# Largely adopted from "optax-swag" (https://github.com/activatedgeek/optax-swag)

from functools import partial
from typing import NamedTuple, Tuple
import logging
import jax
import jax.numpy as jnp
import optax
import chex


class SWAState(NamedTuple):
  """State for SWAG mean and non-centered variance."""
  mean: optax.Params  # Running mean of iterates.
  step: chex.Array = jnp.zeros([], jnp.int32)  # Step count.
  n: chex.Array = jnp.zeros([], jnp.int32)  # Iterate count using for running stats.
  train_step: chex.Array = jnp.zeros([], jnp.int32)  # Training step count.
  swa_batch_stats: chex.Array = None  # Batch stats for SWA.


class SWAGDiagState(NamedTuple):
  """State for SWAG mean and diagonal non-centered variance."""
  mean: optax.Params  # Running mean of iterates.
  params2: optax.Params  # Running non-centered variance of iterates.
  step: chex.Array = jnp.zeros([], jnp.int32)  # Step count.
  n: chex.Array = jnp.zeros([], jnp.int32)  # Iterate count using for running stats.
  train_step: chex.Array = jnp.zeros([], jnp.int32)  # Training step count.
  swa_batch_stats: chex.Array = None  # Batch stats for SWA.


class SWAGState(NamedTuple):
  """State for SWAG mean, diagonal non-centered variance and low rank terms."""
  mean: optax.Params  # Running mean of iterates.
  params2: optax.Params  # Running non-centered variance of iterates.
  dparams: optax.Params  # Low rank delta columns.
  step: chex.Array = jnp.zeros([], jnp.int32)  # Step count.
  n: chex.Array = jnp.zeros([], jnp.int32)  # Iterate count using for running stats.
  train_step: chex.Array = jnp.zeros([], jnp.int32)  # Training step count.
  c: chex.Array = jnp.zeros([], jnp.int32)  # Current column to update.
  swa_batch_stats: chex.Array = None  # Batch stats for SWA.
  

def swa(freq: int, start_step: int) -> optax.GradientTransformation:
  assert freq > 0, 'freq must be positive integer.'

  def init_fn(params: optax.Params) -> SWAState:
    return SWAState(mean=params)

  def update_fn(updates: optax.Updates, state: SWAState,
                params: optax.Params) -> Tuple[optax.Updates, SWAState]:
    
    next_train_step = state.train_step + 1

    @jax.jit
    def no_swa_fn():
      next_step = jnp.zeros([], jnp.int32)
      n = jnp.zeros([], jnp.int32)
      next_mean = jax.tree_util.tree_map(lambda p: jnp.zeros_like(p), params)
      return next_step, n, next_mean

    @jax.jit    
    def swa_fn():
      next_step = (state.step + 1) % freq
      update_mask = next_step == 0
      n = state.n + 1 * update_mask
      next_params = jax.tree_util.tree_map(lambda p, u: jnp.where(update_mask, p + u, p),
                                          params, updates)
      next_mean = jax.tree_util.tree_map(lambda mu, np: jnp.where(update_mask, (n * mu + np) / (n + 1), mu),
                                         state.mean, next_params)
      return next_step, n, next_mean
    
    next_step, n, next_mean \
    = jax.lax.cond(state.train_step >= start_step, swa_fn, no_swa_fn)
    
    return updates, SWAState(step=next_step, n=n, mean=next_mean, train_step=next_train_step)

  return optax.GradientTransformation(init_fn, update_fn)


def swag_diag(freq: int, start_step: int) -> optax.GradientTransformation:
  assert freq > 0, 'freq must be positive integer.'
  assert start_step % freq == 0, 'start_step must be divisible by freq.'

  def init_fn(params: optax.Params) -> SWAGDiagState:
    return SWAGDiagState(
        mean=params,
        params2=jax.tree_util.tree_map(lambda p: jnp.square(p), params))

  def update_fn(updates: optax.Updates, state: SWAGDiagState,
                params: optax.Params) -> Tuple[optax.Updates, SWAGDiagState]:
    
    next_train_step = state.train_step + 1

    @jax.jit
    def no_swa_fn():
      next_step = jnp.zeros([], jnp.int32)
      n = jnp.zeros([], jnp.int32)
      next_mean = jax.tree_util.tree_map(lambda p: jnp.zeros_like(p), params)
      next_params2 = jax.tree_util.tree_map(lambda p: jnp.zeros_like(p), params)
      return next_step, n, next_mean, next_params2

    @jax.jit    
    def swa_fn():
      next_step = (state.step + 1) % freq
      update_mask = next_step == 0
      n = state.n + 1 * update_mask
      next_params = jax.tree_util.tree_map(lambda p, u: jnp.where(update_mask, p + u, p),
                                          params, updates)
      next_mean = jax.tree_util.tree_map(lambda mu, np: jnp.where(update_mask, (n * mu + np) / (n + 1), mu),
                                         state.mean, next_params)
      next_params2 = jax.tree_util.tree_map(lambda p2, np: jnp.where(update_mask, (n * p2 + jnp.square(np)) / (n + 1), p2),
                                            state.params2, next_params)
      return next_step, n, next_mean, next_params2

    next_step, n, next_mean, next_params2 \
    = jax.lax.cond(state.train_step >= start_step, swa_fn, no_swa_fn)

    return updates, SWAGDiagState(step=next_step, n=n, mean=next_mean, 
                                  params2=next_params2, train_step=next_train_step)

  return optax.GradientTransformation(init_fn, update_fn)


def swag(freq: int, rank: int, start_step: int) -> optax.GradientTransformation:
  assert freq > 0, 'freq must be positive integer.'
  assert start_step % freq == 0, 'start_step must be divisible by freq.'
  if rank < 2:
    logging.warning('Rank must be greater than 1. Switching to swag_diag.')
    return swag_diag(freq)

  def init_fn(params: optax.Params) -> SWAGState:
    return SWAGState(
        mean=params,
        params2=jax.tree_util.tree_map(lambda p: jnp.square(p), params),
        dparams=jax.tree_util.tree_map(lambda p: jnp.zeros_like(jnp.repeat(p[jnp.newaxis, ...], rank, axis=0)), params))

  def update_fn(updates: optax.Updates, state: SWAGState,
                params: optax.Params) -> Tuple[optax.Updates, SWAGState]:
    
    next_train_step = state.train_step + 1
    
    @jax.jit
    def no_swa_fn():
      next_step = jnp.zeros([], jnp.int32)
      n = jnp.zeros([], jnp.int32)
      next_mean = jax.tree_util.tree_map(lambda p: jnp.zeros_like(p), params)
      next_params2 = jax.tree_util.tree_map(lambda p: jnp.zeros_like(p), params)
      next_dparams = jax.tree_util.tree_map(lambda p: jnp.zeros_like(jnp.repeat(p[jnp.newaxis, ...], rank, axis=0)), params)
      c = jnp.zeros([], jnp.int32)
      return next_step, n, next_mean, next_params2, next_dparams, c
    
    @jax.jit  
    def swa_fn():
      next_step = (state.step + 1) % freq
      update_mask = next_step == 0
      n = state.n + 1 * update_mask
      next_params = jax.tree_util.tree_map(lambda p, u: jnp.where(update_mask, p + u, p),
                                          params, updates)
      next_mean = jax.tree_util.tree_map(lambda mu, np: jnp.where(update_mask, (n * mu + np) / (n + 1), mu),
                                         state.mean, next_params)
      next_params2 = jax.tree_util.tree_map(lambda p2, np: jnp.where(update_mask, (n * p2 + jnp.square(np)) / (n + 1), p2),
                                            state.params2, next_params)
      next_dparams = jax.tree_util.tree_map(lambda dp, np, nmu: jnp.where(update_mask, dp.at[state.c].set(np - nmu), dp),
                                            state.dparams, next_params, next_mean)
      c = (state.c + 1 * update_mask) % rank
      return next_step, n, next_mean, next_params2, next_dparams, c
    
    next_step, n, next_mean, next_params2, next_dparams, c \
    = jax.lax.cond(state.train_step >= start_step, swa_fn, no_swa_fn)

    return updates, SWAGState(step=next_step, n=n, c=c, mean=next_mean, 
                              params2=next_params2, dparams=next_dparams, train_step=next_train_step)

  return optax.GradientTransformation(init_fn, update_fn)


def sample_swag_diag(num_samples: int, key: chex.PRNGKey, state: SWAGDiagState, eps: float = 1e-30) -> optax.Params:
    mean, tree_unflatten_fn = jax.flatten_util.ravel_pytree(state.mean)
    p2, _ = jax.flatten_util.ravel_pytree(state.params2)
    std = jnp.sqrt(jnp.clip(p2 - jnp.square(mean), a_min=eps))
    
    z = jax.random.normal(key, (num_samples, *mean.shape))
    
    construct_swagdiag_params = lambda z: mean + std * z
    swagdiag_params = jax.vmap(construct_swagdiag_params)(z)

    swagdiag_param_list = []
    for i in range(len(swagdiag_params)):
        swagdiag_param_list.append(tree_unflatten_fn(swagdiag_params[i]))
    
    return swagdiag_param_list
  

def sample_swag(num_samples: int, key: chex.PRNGKey, state: SWAGState, scale: float = 1., eps: float = 1e-30):
    mean, tree_unflatten_fn = jax.flatten_util.ravel_pytree(state.mean)
    p2, _ = jax.flatten_util.ravel_pytree(state.params2)
    
    std = jnp.sqrt(jnp.clip(p2 - jnp.square(mean), a_min=eps))

    dparams = jax.vmap(lambda tree: jax.flatten_util.ravel_pytree(tree)[0])(state.dparams)
    rank = dparams.shape[0]
    
    z1_key, z2_key = jax.random.split(key, 2)
    z1 = jax.random.normal(z1_key, (num_samples, *mean.shape))
    z2 = jax.random.normal(z2_key, (num_samples, rank))
    
    z1_scale = scale / jnp.sqrt(2)
    z2_scale = scale / jnp.sqrt(2 * (rank - 1))
    
    construct_swag_params = lambda z1, z2: mean + z1_scale * std * z1 + z2_scale * jnp.matmul(dparams.T, z2)
    swag_params = jax.vmap(construct_swag_params, in_axes=(0, 0))(z1, z2)
    
    swag_param_list = []
    for i in range(len(swag_params)):
        swag_param_list.append(tree_unflatten_fn(swag_params[i]))

    return swag_param_list