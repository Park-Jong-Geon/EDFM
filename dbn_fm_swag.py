from builtins import NotImplementedError
import os
import orbax
from easydict import EasyDict
import random

from typing import Any

import flax
from flax.training import train_state, common_utils, checkpoints
from flax.core.frozen_dict import freeze
from flax import jax_utils

import optax

import jax.numpy as jnp
import numpy as np
import jax
import jaxlib

import datetime
import wandb

import defaults_dsb as defaults
from tabulate import tabulate
import sys
from data.build import build_dataloaders 
from giung2.metrics import evaluate_acc, evaluate_nll
from giung2.models.layers import FilterResponseNorm
from models.resnet import FlaxResNet, FlaxResNetBase
from models.i2sb import ClsUnet
from models.flowmatching import MlpBridge, ClsUnet, FlowMatching
from collections import OrderedDict
from tqdm import tqdm
from utils import WandbLogger
from utils import get_config
from tqdm import tqdm
from functools import partial
import defaults_sgd
from einops import rearrange

from swag import sample_swag_diag
from sgd_swag import update_swag_batch_stats
from collections import namedtuple
import copy
from models.hungarian_cover import hungarian_cover_tpu_matcher
from utils import batch_mul

random.seed(0)

class TrainState(train_state.TrainState):
    rng: Any
    ema_params: Any
    batch_stats: Any = None


def get_resnet(config, head=False, return_emb=False):
    if config.model_name == 'FlaxResNet':
        _ResNet = partial(
            FlaxResNet if head else FlaxResNetBase,
            depth=config.model_depth,
            widen_factor=config.model_width,
            dtype=config.dtype,
            pixel_mean=defaults.PIXEL_MEAN,
            pixel_std=defaults.PIXEL_STD,
            num_classes=config.num_classes,
            num_planes=config.model_planes,
            num_blocks=tuple(
                int(b) for b in config.model_blocks.split(",")
            ) if config.model_blocks is not None else None,
            first_conv=config.first_conv,
            first_pool=config.first_pool,
            return_emb=return_emb,
        )

    if config.model_style == 'BN-ReLU':
        model = _ResNet
    elif config.model_style == "FRN-Swish":
        model = partial(
            _ResNet,
            conv=partial(
                flax.linen.Conv,
                use_bias=not config.model_nobias,
                kernel_init=jax.nn.initializers.he_normal(),
                bias_init=jax.nn.initializers.zeros),
            norm=FilterResponseNorm,
            relu=flax.linen.swish)

    if head:
        return model
    return partial(model, out=config.feature_name)


def load_resnet(ckpt_dir):
    ckpt = checkpoints.restore_checkpoint(
        ckpt_dir=ckpt_dir,
        target=None
    )
    if ckpt.get("model") is not None:
        params = ckpt["model"]["params"]
        batch_stats = ckpt["model"].get("batch_stats")
        image_stats = ckpt["model"].get("image_stats")
    else:
        params = ckpt["params"]
        batch_stats = ckpt.get("batch_stats")
        image_stats = ckpt.get("image_stats")
    return ckpt, params, batch_stats, image_stats


def load_teacher(ckpt_dir):
    ckpt = checkpoints.restore_checkpoint(
        ckpt_dir=ckpt_dir,
        target=None
    )
    params = ckpt["params"]
    batch_stats = ckpt.get("batch_stats")
    config = ckpt["config"]
    return params, batch_stats, config


def load_saved(ckpt_dir):
    ckpt = checkpoints.restore_checkpoint(
        ckpt_dir=ckpt_dir,
        target=None
    )
    params = ckpt["params"]
    batch_stats = ckpt.get("batch_stats")
    config = ckpt["config"]
    tx_saved = ckpt.get("state") is not None
    return params, batch_stats, config, tx_saved


def pdict(params, batch_stats=None, image_stats=None):
    params_dict = dict(params=params)
    if batch_stats is not None:
        params_dict["batch_stats"] = batch_stats
    if image_stats is not None:
        params_dict["image_stats"] = image_stats
    return params_dict


# def get_scorenet(config):
#     score_func = partial(
#         MlpBridge,
#         emb_dim=config.emb_dim,
#         num_blocks=config.num_blocks,
#         fourier_scale=config.fourier_scale,
#     )
#     return score_func
def get_scorenet(config):
    score_func = partial(
        ClsUnet,        
        num_classes = config.num_classes,
        ch = config.ch,
        droprate = config.droprate,
    )
    return score_func

def build_dbn(config):
    resnet = get_resnet(config, head=True, return_emb=True)
    score_net = get_scorenet(config)
    dbn = FlowMatching(
        res_net=resnet,
        score_net=score_net,
        steps=config.T,
        var=config.var,
        num_classes=config.num_classes,
        eps=config.train_timestep_truncation,
        alpha=config.train_timestep_alpha,
    )
    return dbn

def fm_sample(score, l0, z, c, config, steps, num_models):
    batch_size = l0.shape[0]
    # timesteps = jnp.linspace(0., 1., steps+1)
    a = config.sample_timestep_alpha
    timesteps = jnp.array([(1-a**i)/(1-a**steps) for i in range(steps+1)])

    @jax.jit
    def body_fn(n, l_n):
        current_t = jnp.array([timesteps[n]])
        current_t = jnp.tile(current_t, [batch_size])

        next_t = jnp.array([timesteps[n+1]])
        next_t = jnp.tile(next_t, [batch_size])

        eps = score(l_n, z, t=current_t)
        euler_l_n = l_n + batch_mul(next_t-current_t, eps)
        # return euler_l_n
        
        eps2 = score(euler_l_n, z, t=next_t)
        heun_l_n = l_n + batch_mul((next_t-current_t)/2, eps+eps2)
        
        return heun_l_n

    val = l0
    for i in range(0, steps-1):
        val = body_fn(i, val)
    current_t = jnp.array([timesteps[steps-1]])
    current_t = jnp.tile(current_t, [batch_size])

    next_t = jnp.array([timesteps[steps]])
    next_t = jnp.tile(next_t, [batch_size])

    eps = score(val, z, t=current_t)
    val += batch_mul(next_t-current_t, eps)
        
    prob = jax.nn.softmax(val).reshape(-1, num_models, config.num_classes).mean(1)
    logits = val.reshape(-1, num_models, config.num_classes)
    return prob, logits

def wasserstein_2_distance(x, y):
    assert x.shape == y.shape
    cost_matrix = jnp.sum((x[:, None, :] - y[None, :, :])**2, axis=-1)
    cost_matrix = cost_matrix[None, ...]
    idx = hungarian_cover_tpu_matcher(cost_matrix)
    idx = idx[0]
    return (cost_matrix[0][idx[0], idx[1]].sum() / x.shape[0])**0.5

def launch(config, print_fn):
    # ------------------------------------------------------------------------
    # load teacher for distillation
    # ------------------------------------------------------------------------
    if config.distill:
        print(f"Loading teacher DBN {config.distill.split('/')[-1]}")
        teacher_params, teacher_batch_stats, teacher_config = load_teacher(
            config.distill)
        teacher_props = [
            "crt", "ensemble_prediction", "ensemble_exclude_a",
            "residual_prediction", "dsc", "frn_swish", "input_scaling",
            "width_multi", "dsb_continuous", "distribution",
            "beta1", "beta2", "linear_noise", "fat", "joint_depth",
            "kld_temp", "forget", "mimo_cond", "start_temp", "n_feat",
            "version", "droprate", "feature_name"
        ]
        for k in teacher_props:
            if teacher_config.get(k) is None:
                continue
            setattr(config, k, teacher_config[k])
        config.T = 1
        config.optim_lr = 0.1*teacher_config["optim_lr"]
        teacher_dsb_stats = dsb_schedules(
            teacher_config["beta1"], teacher_config["beta2"],
            teacher_config["T"], teacher_config["linear_noise"]
        )
    if config.resume:
        def props(cls):   
            return [i for i in cls.__dict__.keys() if i[:1] != '_']
        print(f"Loading checkpoint {config.resume}")
        saved_params, saved_batch_stats, saved_config, tx_saved = load_saved(
            config.resume)
        last_epoch_idx = config.last_epoch_idx
        resume = config.resume
        _config = get_config(saved_config)
        for k in props(_config):
            v = getattr(_config, k)
            setattr(config, k, v)
        config.last_epoch_idx = last_epoch_idx
        config.resume = resume

    rng = jax.random.PRNGKey(config.seed)
    model_dtype = jnp.float32
    config.dtype = model_dtype
    config.image_stats = dict(
        m=jnp.array(defaults_sgd.PIXEL_MEAN),
        s=jnp.array(defaults_sgd.PIXEL_STD))

    # ------------------------------------------------------------------------
    # load image dataset (C10, C100, TinyImageNet, ImageNet)
    # ------------------------------------------------------------------------
    dataloaders = build_dataloaders(config)
    config.num_classes = dataloaders["num_classes"]
    config.trn_steps_per_epoch = dataloaders["trn_steps_per_epoch"] * config.num_steps_per_sample

    # ------------------------------------------------------------------------
    # define and load resnet
    # ------------------------------------------------------------------------
    swag_state_list = []
    for swag_ckpt_dir in config.swag_ckpt_dir:
        resnet_state, _, batch_stats, image_stats = load_resnet(swag_ckpt_dir)
        d = resnet_state['model']['opt_state']['1']
        swag_state = namedtuple('SWAGState', d.keys())(*d.values())
        swag_state_list.append(swag_state)

    resnet = get_resnet(config, head=True)
    resnet = resnet()
    @jax.jit
    def forward_resnet(params_dict, images):
        mutable = ["intermediates"]
        _, state = resnet.apply(
            params_dict, images, rngs=None,
            mutable=mutable,
            training=False, use_running_average=True)
        logits = state["intermediates"]["cls.logit"][0]
        return logits

    # ------------------------------------------------------------------------
    # define score and cls
    # ------------------------------------------------------------------------
    print("Building Diffusion Bridge Network (DBN)...")
    dbn = build_dbn(config)

    # ------------------------------------------------------------------------
    # initialize score & cls and replace base and cls with loaded params
    # ------------------------------------------------------------------------
    print("Initializing DBN...")
    _, h, w, d = dataloaders["image_shape"]
    x_dim = (h, w, d)
    init_rng, sub_rng = jax.random.split(rng)
    variables = dbn.init(
        {"params": init_rng, "dropout": init_rng},
        rng=init_rng,
        l0=jnp.empty((1, config.num_classes)),
        x=jnp.empty((1, *x_dim)),
        training=False
    )
    if variables.get('batch_stats'):
        initial_batch_stats = copy.deepcopy(variables['batch_stats'])

    if config.distill:
        variables = variables.unfreeze()
        variables["params"] = teacher_params
        if variables.get("batch_stats") is not None:
            variables["batch_stats"] = teacher_batch_stats
        variables = freeze(variables)
    if config.resume:
        variables = variables.unfreeze()
        variables["params"] = saved_params
        if variables.get("batch_stats") is not None:
            variables["batch_stats"] = saved_batch_stats
        variables = freeze(variables)

    # ------------------------------------------------------------------------
    # define optimizers
    # ------------------------------------------------------------------------
    scheduler = optax.cosine_decay_schedule(
        init_value=config.optim_lr,
        decay_steps=config.optim_ne * config.trn_steps_per_epoch)

    # if config.optim_base == "adam":
    #     optimizer = optax.adamw(learning_rate=scheduler, weight_decay=config.optim_weight_decay)
    # elif config.optim_base == "sgd":
    #     optimizer = optax.sgd(learning_rate=scheduler, momentum=config.optim_momentum)
    # else:
    #     raise NotImplementedError
    if config.optim_base == "adam":
        base_optim = partial(
            optax.adamw, learning_rate=scheduler, weight_decay=config.optim_weight_decay
        )
    elif config.optim_base == "sgd":
        base_optim = partial(optax.sgd, learning_rate=scheduler,
                             momentum=config.optim_momentum)
    else:
        raise NotImplementedError
    
    partition_optimizers = {
        "resnet": base_optim(), #optax.set_to_zero(), 
        "score": base_optim(),
    }
    def tagging(path, v):
        if "resnet" in path:
            return "resnet"
        elif "score" in path:
            return "score"
        else:
            raise NotImplementedError
    partitions = flax.core.freeze(
        flax.traverse_util.path_aware_map(tagging, variables["params"]))
    optimizer = optax.multi_transform(partition_optimizers, partitions)
    
    # ------------------------------------------------------------------------
    # create train state
    # ------------------------------------------------------------------------
    state_rng, sub_rng = jax.random.split(sub_rng)
    state = TrainState.create(
        apply_fn=dbn.apply,
        params=variables["params"],
        ema_params=variables["params"],
        tx=optimizer,
        batch_stats=variables.get("batch_stats"),
        rng=state_rng
    )
    # Load ResNet
    state = state.replace(params=freeze(dict(resnet=swag_state_list[0].mean, 
                                             score=state.params["score"])),)

    if config.resume:
        last_epochs = config.last_epoch_idx + 1
        saved_iteration = config.trn_steps_per_epoch * last_epochs
        state = state.replace(step=saved_iteration)
        if tx_saved:
            print("Load saved optimizer")
            _ckpt = dict(
                params=state.params,
                batch_stats=state.batch_stats,
                config=dict(),
                state=state
            )
            ckpt = checkpoints.restore_checkpoint(
                ckpt_dir=config.resume,
                target=_ckpt
            )
            state = state.replace(tx=ckpt["state"].tx)
            del _ckpt
            del ckpt


    # ------------------------------------------------------------------------
    # define sampler and metrics
    # ------------------------------------------------------------------------
    @jax.jit
    def mse_loss(noise, output):
        p = config.mse_power
        sum_axis = list(range(1, len(output.shape[1:])+1))
        loss = jnp.sum(jnp.abs(noise-output)**p, axis=sum_axis)
        return loss

    @jax.jit
    def pseudohuber_loss(noise, output):
        sum_axis = list(range(1, len(output.shape[1:])+1))
        loss = jnp.sum(jnp.sqrt((noise-output)**2 + 0.003**2) - 0.003, axis=sum_axis)
        return loss
    
    @jax.jit
    def ce_loss_with_target(logits, target):
        pred = jax.nn.log_softmax(logits, axis=-1)
        loss = -jnp.sum(target*pred, axis=-1)
        return loss
    
    @jax.jit
    def ce_loss(logits, labels):
        target = common_utils.onehot(labels, num_classes=logits.shape[-1])
        pred = jax.nn.log_softmax(logits, axis=-1)
        loss = -jnp.sum(target*pred, axis=-1)
        return loss

    @jax.jit
    def kld_loss(target_list, refer_list, T=1.):
        if not isinstance(target_list, list):
            target_list = [target_list]
        if not isinstance(refer_list, list):
            refer_list = [refer_list]
        kld_sum = 0
        count = 0
        for tar, ref in zip(target_list, refer_list):
            logq = jax.nn.log_softmax(tar/T, axis=-1)
            logp = jax.nn.log_softmax(ref/T, axis=-1)
            q = jnp.exp(logq)
            integrand = q*(logq-logp)
            kld = jnp.sum(integrand, axis=-1)
            kld_sum += T**2*kld
            count += 1
        return kld_sum/count

    @jax.jit
    def self_kld_loss(target_list, T=1.):
        N = len(target_list)
        kld_sum = 0
        count = N*(N-1)
        logprob_list = [jax.nn.log_softmax(
            tar/T, axis=-1) for tar in target_list]
        for logq in logprob_list:
            for logp in logprob_list:
                q = jnp.exp(logq)
                integrand = q*(logq-logp)
                kld = jnp.sum(integrand, axis=-1)
                kld_sum += T**2*kld
        return kld_sum/count

    @jax.jit
    def reduce_mean(loss, marker):
        count = jnp.sum(marker)
        loss = jnp.where(marker, loss, 0).sum()
        loss = jnp.where(count != 0, loss/count, loss)
        return loss

    @jax.jit
    def reduce_sum(loss, marker):
        loss = jnp.where(marker, loss, 0).sum()
        return loss

    # ------------------------------------------------------------------------
    # define step collecting features and logits
    # ------------------------------------------------------------------------
    @partial(jax.pmap, axis_name="batch")
    def step_label(state, batch):
        rng = state.rng
        swag_param_list = []        
        for _ in range(config.num_steps_per_sample):
            swag_state = random.choice(swag_state_list)
            swag_params_per_seed = sample_swag_diag(1, rng, swag_state)
            swag_param_list += swag_params_per_seed
            rng, _ = jax.random.split(rng)
            
        logitsA = []
        for swag_param in swag_param_list:
            res_params_dict = dict(params=swag_param)
            if image_stats is not None:
                res_params_dict["image_stats"] = image_stats
            if batch_stats is not None:
                # Update batch_stats
                trn_loader = dataloaders['dataloader'](rng=state.rng)
                trn_loader = jax_utils.prefetch_to_device(trn_loader, size=2)
                iter_batch_stats = initial_batch_stats
                for _, batch in enumerate(trn_loader, start=1):
                    iter_batch_stats = update_swag_batch_stats(resnet, res_params_dict, batch, iter_batch_stats)
                res_params_dict["batch_stats"] = cross_replica_mean(iter_batch_stats)
            
            images = batch["images"]
            logits0 = forward_resnet(res_params_dict, images)
            # logits0: (B, d)
            logits0 = logits0 - logits0.mean(-1, keepdims=True)
            logitsA.append(logits0)
        logitsB = logitsA[0]

        batch["logitsB"] = logitsB
        batch["logitsA"] = logitsA
        if config.distill:
            drop_rng, score_rng = jax.random.split(state.rng)
            params_dict = pdict(
                params=teacher_params,
                image_stats=config.image_stats,
                batch_stats=teacher_batch_stats
            )
            rngs_dict = dict(dropout=drop_rng)
            model_bd = dbn.bind(params_dict, rngs=rngs_dict)
            teacher_steps = teacher_config["T"]
            _fm_sample = partial(
                fm_sample,
                config=EasyDict(teacher_config),
                dsb_stats=teacher_dsb_stats,
                z_dsb_stats=None,
                steps=teacher_steps)
            logitsC, _ = model_bd.sample(
                score_rng, _fm_sample, batch["images"])
            logitsC = rearrange(
                logitsC, "n (t b) z -> t b (n z)", t=teacher_steps+1)
            batch["logitsC"] = logitsC[-1]
        return batch

    # ------------------------------------------------------------------------
    # define train step
    # ------------------------------------------------------------------------

    def loss_func(params, state, batch, logitsA, train=True):
        drop_rng, score_rng = jax.random.split(state.rng)
        params_dict = pdict(
            params=params,
            image_stats=config.image_stats,
            batch_stats=state.batch_stats)
        rngs_dict = dict(dropout=drop_rng)
        mutable = ["batch_stats"]
        output = state.apply_fn(
            params_dict, score_rng,
            logitsA, batch["images"],
            training=train,
            rngs=rngs_dict,
            **(dict(mutable=mutable) if train else dict()),
        )
        new_model_state = output[1] if train else None
        epsilon, next_l_t = output[0] if train else output

        # score_loss = ce_loss_with_target(epsilon, next_l_t)
        score_loss = mse_loss(epsilon, next_l_t)
        # score_loss = pseudohuber_loss(epsilon, next_l_t)
        if batch.get("logitsC") is not None:
            logitsC = batch["logitsC"]
            a = config.distill_alpha
            score_loss = (
                a*mse_loss(epsilon, (l1-logitsC)) + (1-a)*score_loss
            )
        total_loss = reduce_mean(score_loss, batch["marker"])

        count = jnp.sum(batch["marker"])
        metrics = OrderedDict({
            "loss": total_loss*count,
            "score_loss": total_loss*count,
            "count": count,
        })

        return total_loss, (metrics, new_model_state)

    def get_loss_func():
        return loss_func

    @ partial(jax.pmap, axis_name="batch")
    def step_train(state, batch):
        for logitsA in batch["logitsA"]:
            def loss_fn(params):
                _loss_func = get_loss_func()
                return _loss_func(params, state, batch, logitsA)
            
            (loss, (metrics, new_model_state)), grads = jax.value_and_grad(
                loss_fn, has_aux=True)(state.params)
            grads = jax.lax.pmean(grads, axis_name="batch")

            state = state.apply_gradients(
                grads=grads, batch_stats=new_model_state.get("batch_stats"))
            a = config.ema_decay
            def update_ema(wt, ema_tm1): return jnp.where(
                (wt != ema_tm1) & (a < 1), (1-a)*wt + a*ema_tm1, wt)
            state = state.replace(
                ema_params=jax.tree_util.tree_map(
                    update_ema,
                    state.params,
                    state.ema_params))
            metrics = jax.lax.psum(metrics, axis_name="batch")
        return state, metrics

    # ------------------------------------------------------------------------
    # define sampling step
    # ------------------------------------------------------------------------
    def evaluate_accnll(prob, labels, marker):
        acc = evaluate_acc(
            prob, labels, log_input=False, reduction="none")
        nll = evaluate_nll(
            prob, labels, log_input=False, reduction='none')
        acc = reduce_sum(acc, marker)
        nll = reduce_sum(nll, marker)
        
        return acc, nll

    def sample_func(state, batch, steps):
        labels = batch["labels"]
        drop_rng, score_rng = jax.random.split(state.rng)

        params_dict = pdict(
            params=state.ema_params,
            image_stats=config.image_stats,
            batch_stats=state.batch_stats)
        rngs_dict = dict(dropout=drop_rng)
        model_bd = dbn.bind(params_dict, rngs=rngs_dict)
        
        # Ensemble
        _fm_sample = partial(
            fm_sample, config=config, steps=steps, num_models=config.num_ensembles)
        prob_ens, _ = model_bd.sample(
            score_rng, _fm_sample, batch["images"], config.num_ensembles)
        
        # Single model
        _fm_sample = partial(
            fm_sample, config=config, steps=steps, num_models=1)
        prob_1, _ = model_bd.sample(
            score_rng, _fm_sample, batch["images"], 1)

        acc_ens, nll_ens = evaluate_accnll(prob_ens, labels, batch["marker"])
        acc_1, nll_1 = evaluate_accnll(prob_1, labels, batch["marker"])
        metrics = OrderedDict({
            "count": jnp.sum(batch["marker"]),
        })
        metrics[f"acc_ens"] = acc_ens
        metrics[f"nll_ens"] = nll_ens
        metrics[f"acc_1"] = acc_1
        metrics[f"nll_1"] = nll_1
        return metrics

    def _sample_func(state, batch, steps):
        drop_rng, score_rng = jax.random.split(state.rng)

        params_dict = pdict(
            params=state.ema_params,
            image_stats=config.image_stats,
            batch_stats=state.batch_stats)
        rngs_dict = dict(dropout=drop_rng)
        model_bd = dbn.bind(params_dict, rngs=rngs_dict)
        _fm_sample = partial(
            fm_sample, config=config, steps=steps, num_models=config.num_models)
        _, val = model_bd.sample(
            score_rng, _fm_sample, batch["images"], config.num_models)
        return val

    @partial(jax.pmap, axis_name="batch")
    def step_sample(state, batch):
        steps = config.T
        metrics = sample_func(state, batch, steps=steps)
        metrics = jax.lax.psum(metrics, axis_name="batch")
        return metrics
    
    @partial(jax.pmap, axis_name="batch")
    def step_measure_distance(state, batch):
        rng = state.rng
        swag_param_list = []
        swag_state = random.choice(swag_state_list)
        
        for _ in range(config.num_models):
            swag_state = random.choice(swag_state_list)
            swag_params_per_seed = sample_swag_diag(1, rng, swag_state)
            swag_param_list += swag_params_per_seed
            rng, _ = jax.random.split(rng)
            
        logitsA = []
        for swag_param in swag_param_list:
            res_params_dict = dict(params=swag_param)
            if image_stats is not None:
                res_params_dict["image_stats"] = image_stats
            if batch_stats is not None:
                # Update batch_stats
                trn_loader = dataloaders['dataloader'](rng=state.rng)
                trn_loader = jax_utils.prefetch_to_device(trn_loader, size=2)
                iter_batch_stats = initial_batch_stats
                for _, batch in enumerate(trn_loader, start=1):
                    iter_batch_stats = update_swag_batch_stats(resnet, res_params_dict, batch, iter_batch_stats)
                res_params_dict["batch_stats"] = cross_replica_mean(iter_batch_stats)
            
            images = batch["images"]
            logits0 = forward_resnet(res_params_dict, images)
            # logits0: (B, d)
            logits0 = logits0 - logits0.mean(-1, keepdims=True)
            logitsA.append(logits0)
            
        steps = config.T
        ens_logits = jnp.stack(logitsA, axis=0).transpose(1, 0, 2)
        ens_logits = ens_logits - ens_logits.mean(-1, keepdims=True)
        pred_logits = _sample_func(state, batch, steps=steps)
        pred_logits = pred_logits - pred_logits.mean(-1, keepdims=True)
        assert len(ens_logits) == len(pred_logits)
        
        w2_dist = 0
        ens_var = 0
        pred_var = 0
        for ens, pred in zip(ens_logits, pred_logits):
            w2_dist += wasserstein_2_distance(ens, pred)
            ens_cov = jnp.cov(ens, rowvar=False)
            pred_cov = jnp.cov(pred, rowvar=False)
            ens_var += jnp.trace(jnp.abs(ens_cov))
            pred_var += jnp.trace(jnp.abs(pred_cov))
        w2_dist /= len(pred_logits)
        w2_dist = jax.lax.pmean(w2_dist, axis_name="batch")
        ens_var /= len(pred_logits)
        ens_var = jax.lax.pmean(ens_var, axis_name="batch")
        pred_var /= len(pred_logits)
        pred_var = jax.lax.pmean(pred_var, axis_name="batch")
        
        return w2_dist, ens_var, pred_var

    # ------------------------------------------------------------------------
    # define mixup
    # ------------------------------------------------------------------------
    @partial(jax.pmap, axis_name="batch")
    def step_mixup(state, batch):
        count = jnp.sum(batch["marker"])
        x = batch["images"]
        y = batch["labels"]
        batch_size = x.shape[0]
        a = config.mixup_alpha
        beta_rng, perm_rng = jax.random.split(state.rng)

        lamda = jnp.where(a > 0, jax.random.beta(beta_rng, a, a), 1)

        perm_x = jax.random.permutation(perm_rng, x)
        perm_y = jax.random.permutation(perm_rng, y)
        mixed_x = (1-lamda)*x+lamda*perm_x
        mixed_y = jnp.where(lamda < 0.5, y, perm_y)
        mixed_x = jnp.where(count == batch_size, mixed_x, x)
        mixed_y = jnp.where(count == batch_size, mixed_y, y)

        batch["images"] = mixed_x
        batch["labels"] = mixed_y
        return batch
    # ------------------------------------------------------------------------
    # init settings and wandb
    # ------------------------------------------------------------------------
    cross_replica_mean = jax.pmap(lambda x: jax.lax.pmean(x, 'x'), 'x')
    best_acc = -float("inf")
    best_nll = float("inf")
    train_summary = dict()
    valid_summary = dict()
    test_summary = dict()
    state = jax_utils.replicate(state)

    wandb.init(
        project="dbn",
        config=vars(config),
        mode="disabled" if config.nowandb else "online"
    )
    wandb.define_metric("val/loss", summary="min")
    wandb.define_metric("val/acc", summary="max")
    wandb.run.summary["params_resnet"] = (
        sum(x.size for x in jax.tree_util.tree_leaves(
            variables["params"]["resnet"]))
    )
    wandb.run.summary["params_score"] = (
        sum(x.size for x in jax.tree_util.tree_leaves(
            variables["params"]["score"]))
    )
    # params_flatten = flax.traverse_util.flatten_dict(
    #     variables["params"]["score"])
    # for k, v in params_flatten.items():
    #     print("score", k, v.shape, flush=True)
    wl = WandbLogger()

    def summarize_metrics(metrics, key="trn"):
        metrics = common_utils.get_metrics(metrics)
        summarized = {
            f"{key}/{k}": v for k, v in jax.tree_util.tree_map(lambda e: e.sum(0), metrics).items()}
        inter_samples = False
        for k, v in summarized.items():
            if "count" in k:
                continue
            elif "lr" in k:
                continue
            if "_inter" in k:
                inter_samples = True
            summarized[k] /= summarized[f"{key}/count"]
        if inter_samples and key == "tst":
            assert summarized.get(f"tst/acc{config.T+1}_inter") is None
            acc_str = ",".join(
                [f"ACC({config.T+1-i})   {summarized[f'{key}/acc{i}_inter']:.4f}" for i in range(1, config.T+1)])
            print(acc_str, flush=True)
        if inter_samples:
            for i in range(1, config.T+1):
                del summarized[f"{key}/acc{i}_inter"]

        del summarized[f"{key}/count"]
        return summarized

    for epoch_idx in tqdm(range(config.last_epoch_idx, config.optim_ne)):
        epoch_rng = jax.random.fold_in(sub_rng, epoch_idx)
        train_rng, valid_rng, test_rng = jax.random.split(epoch_rng, 3)

        valid_only_cond =(config.last_epoch_idx > 0 and epoch_idx==config.last_epoch_idx) 
        # ------------------------------------------------------------------------
        # train by getting features from each resnet
        # ------------------------------------------------------------------------
        if not valid_only_cond:
            train_loader = dataloaders["dataloader"](rng=epoch_rng)
            train_loader = jax_utils.prefetch_to_device(train_loader, size=2)
            train_metrics = []
            for batch_idx, batch in enumerate(train_loader):
                batch_rng = jax.random.fold_in(train_rng, batch_idx)
                state = state.replace(rng=jax_utils.replicate(batch_rng))
                if config.mixup_alpha > 0:
                    batch = step_mixup(state, batch)
                batch = step_label(state, batch)
                state, metrics = step_train(state, batch)
                train_metrics.append(metrics)

            train_summarized = summarize_metrics(train_metrics, "trn")
            train_summary.update(train_summarized)
            train_summary["trn/lr"] = scheduler(jax_utils.unreplicate(state.step))
            wl.log(train_summary)

            if state.batch_stats is not None:
                state = state.replace(
                    batch_stats=cross_replica_mean(state.batch_stats))

        # ------------------------------------------------------------------------
        # valid by getting features from each resnet
        # ------------------------------------------------------------------------
        valid_loader = dataloaders["val_loader"](rng=None)
        valid_loader = jax_utils.prefetch_to_device(valid_loader, size=2)
        valid_metrics = []
        for batch_idx, batch in enumerate(valid_loader):
            batch_rng = jax.random.fold_in(valid_rng, batch_idx)
            state = state.replace(rng=jax_utils.replicate(batch_rng))
            # batch = step_label(state, batch)
            # metrics = step_valid(state, batch)
            acc_metrics = step_sample(state, batch)
            metrics = acc_metrics
            valid_metrics.append(metrics)

        valid_summarized = summarize_metrics(valid_metrics, "val")
        valid_summary.update(valid_summarized)
        wl.log(valid_summary)
        
        w2_dist, ens_var, pred_var = step_measure_distance(state, batch)
        wl.log({"val/w2_dist": w2_dist[0]})
        wl.log({"val/ens_var": ens_var[0]})
        wl.log({"val/pred_var": pred_var[0]})

        # ------------------------------------------------------------------------
        # test by getting features from each resnet
        # ------------------------------------------------------------------------
        acc_criteria = valid_summarized[f"val/acc_ens"]
        nll_criteria = valid_summarized[f"val/nll_ens"]
        if (best_acc < acc_criteria) or (best_nll > nll_criteria):
            if valid_only_cond:
                best_acc = acc_criteria
                best_nll = nll_criteria
                continue
            test_loader = dataloaders["tst_loader"](rng=None)
            test_loader = jax_utils.prefetch_to_device(test_loader, size=2)
            test_metrics = []
            for batch_idx, batch in enumerate(test_loader):
                batch_rng = jax.random.fold_in(test_rng, batch_idx)
                state = state.replace(rng=jax_utils.replicate(batch_rng))
                # batch = step_label(state, batch)
                # metrics = step_valid(state, batch)
                acc_metrics = step_sample(state, batch)
                metrics = acc_metrics
                test_metrics.append(metrics)

            test_summarized = summarize_metrics(test_metrics, "tst")
            test_summary.update(test_summarized)
            wl.log(test_summary)
            best_acc = acc_criteria
            best_nll = nll_criteria
            if config.save:
                save_path = config.save
                save_state = jax_utils.unreplicate(state)
                if getattr(config, "dtype", None) is not None:
                    config.dtype = str(config.dtype)
                ckpt = dict(
                    params=save_state.ema_params,
                    batch_stats=save_state.batch_stats,
                    config=vars(config),
                    state=save_state
                )
                orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
                checkpoints.save_checkpoint(
                    ckpt_dir=save_path,
                    target=ckpt,
                    step=epoch_idx,
                    overwrite=True,
                    orbax_checkpointer=orbax_checkpointer
                )

        jax.random.normal(rng, ()).block_until_ready()
        if jnp.isnan(train_summarized["trn/loss"]):
            print("NaN detected")
            break

    # for epoch_idx in tqdm(range(config.last_epoch_idx, config.optim_ne)):
    #     epoch_rng = jax.random.fold_in(sub_rng, epoch_idx)
    #     train_rng, valid_rng, test_rng = jax.random.split(epoch_rng, 3)

    #     # ------------------------------------------------------------------------
    #     # test by getting features from each resnet
    #     # ------------------------------------------------------------------------
    #     test_loader = dataloaders["tst_loader"](rng=None)
    #     test_loader = jax_utils.prefetch_to_device(test_loader, size=2)
    #     test_metrics = []
    #     for batch_idx, batch in enumerate(test_loader):
    #         batch_rng = jax.random.fold_in(test_rng, batch_idx)
    #         state = state.replace(rng=jax_utils.replicate(batch_rng))
    #         batch = step_label(state, batch)
    #         # metrics = step_valid(state, batch)
    #         acc_metrics = step_sample(state, batch)
    #         metrics = acc_metrics
    #         test_metrics.append(metrics)

    #     test_summarized = summarize_metrics(test_metrics, "tst")
    #     test_summary.update(test_summarized)
    #     wl.log(test_summary)
        
        wl.flush()

    wandb.finish()


def main():
    parser = defaults.default_argument_parser()

    parser.add_argument("--config", default=None, type=str)
    args, argv = parser.parse_known_args(sys.argv[1:])
    if args.config is not None:
        import yaml
        with open(args.config, 'r') as f:
            arg_defaults = yaml.safe_load(f)

    parser.add_argument("--model_planes", default=16, type=int)
    parser.add_argument("--model_blocks", default=None, type=str)
    parser.add_argument("--model_nobias", action="store_true")
    parser.add_argument('--first_conv', nargs='+', type=int)
    parser.add_argument('--first_pool', nargs='+', type=int)
    # ---------------------------------------------------------------------------------------
    # optimizer
    # ---------------------------------------------------------------------------------------
    parser.add_argument('--optim_ne', default=350, type=int,
                        help='the number of training epochs (default: 200)')
    parser.add_argument('--optim_lr', default=2e-4, type=float,
                        help='base learning rate (default: 1e-4)')
    parser.add_argument('--optim_momentum', default=0.9, type=float,
                        help='momentum coefficient (default: 0.9)')
    parser.add_argument('--optim_weight_decay', default=0.1, type=float,
                        help='weight decay coefficient (default: 0.0001)')
    parser.add_argument('--optim_base', default="adam", type=str)
    parser.add_argument('--resume', default=None, type=str)
    parser.add_argument('--last_epoch_idx', default=0, type=int)
    # ---------------------------------------------------------------------------------------
    # training
    # ---------------------------------------------------------------------------------------
    parser.add_argument('--save', default=None, type=str)
    parser.add_argument('--seed', default=2023, type=int)
    parser.add_argument("--ema_decay", default=0.9999, type=float)
    parser.add_argument("--mse_power", default=2, type=int)
    parser.add_argument("--mixup_alpha", default=0, type=float)
    # ---------------------------------------------------------------------------------------
    # experiemnts
    # ---------------------------------------------------------------------------------------
    parser.add_argument("--nowandb", action="store_true")
    parser.add_argument("--base_type", default="A", type=str)
    # ---------------------------------------------------------------------------------------
    # diffusion
    # ---------------------------------------------------------------------------------------
    parser.add_argument("--T", default=2000, type=int)
    parser.add_argument("--max_t", default=2000., type=float)
    # ---------------------------------------------------------------------------------------
    # Progressive Distillation
    # ---------------------------------------------------------------------------------------
    parser.add_argument("--distill", default=None, type=str)

    if args.config is not None:
        parser.set_defaults(**arg_defaults)

    args = parser.parse_args()

    if args.seed < 0:
        args.seed = (
            os.getpid()
            + int(datetime.datetime.now().strftime('%S%f'))
            + int.from_bytes(os.urandom(2), 'big')
        )

    print_fn = partial(print, flush=True)
    log_str = tabulate([
        ('sys.platform', sys.platform),
        ('Python', sys.version.replace('\n', '')),
        ('JAX', jax.__version__ + ' @' + os.path.dirname(jax.__file__)),
        ('jaxlib', jaxlib.__version__ + ' @' + os.path.dirname(jaxlib.__file__)),
        ('Flax', flax.__version__ + ' @' + os.path.dirname(flax.__file__)),
        ('Optax', optax.__version__ + ' @' + os.path.dirname(optax.__file__)),
    ]) + '\n'
    log_str = f'Environments:\n{log_str}'
    log_str = datetime.datetime.now().strftime(
        '[%Y-%m-%d %H:%M:%S] ') + log_str
    print_fn(log_str)

    log_str = ''
    max_k_len = max(map(len, vars(args).keys()))
    for k, v in vars(args).items():
        log_str += f'- args.{k.ljust(max_k_len)} : {v}\n'
    log_str = f'Command line arguments:\n{log_str}'
    log_str = datetime.datetime.now().strftime(
        '[%Y-%m-%d %H:%M:%S] ') + log_str
    print_fn(log_str)

    if jax.local_device_count() > 1:
        log_str = f'Multiple local devices are detected:\n{jax.local_devices()}\n'
        log_str = datetime.datetime.now().strftime(
            '[%Y-%m-%d %H:%M:%S] ') + log_str
        print_fn(log_str)

    launch(args, print_fn)


if __name__ == '__main__':
    main()
