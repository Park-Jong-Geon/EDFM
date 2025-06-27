from builtins import NotImplementedError
import os
import sys
import argparse
import pickle
import random
from typing import Any
import wandb
from collections import OrderedDict, namedtuple
from functools import partial
from tqdm import tqdm

import tensorflow as tf
import jax
import jax.numpy as jnp
import flax
from flax import jax_utils
from flax.training import train_state, common_utils, checkpoints
from flax.core.frozen_dict import freeze
import orbax
import optax

from data.build import build_dataloaders 

from models.resnet import FlaxResNet
from models.flowmatching import FlowMatching
from giung2.metrics import evaluate_acc, evaluate_nll
from giung2.models.layers import FilterResponseNorm
from models.mlp import Mlp
from models.transformer import DiT
from models.twod_unet import ClsUnet

from models.swag import sample_swag_diag
from utils.utils import batch_mul, get_config, WandbLogger
import utils.image_processing as image_processing

random.seed(0)

# NOTE: The pixel mean and std are for CIFAR-10/100.
# If you use other datasets, please calculate and change the values accordingly.
PIXEL_MEAN = (0.49, 0.48, 0.44)
PIXEL_STD = (0.2, 0.2, 0.2)

# NOTE: If the teacher models contain BatchNorm, batch_stats has to be recalculated for each SWAG model.
# However, we assume no BatchNorm in neither teacher nor student models for convenience.


class TrainState(train_state.TrainState):
    rng: Any
    ema_params: Any
    batch_stats: Any = None


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


def pdict(params, batch_stats=None):
    params_dict = dict(params=params)
    if batch_stats is not None:
        params_dict["batch_stats"] = batch_stats
    return params_dict


# ResNet for (frozen) teacher network
def get_resnet(config, return_emb=False):
    if config.model_name == 'FlaxResNet':
        _ResNet = partial(
            FlaxResNet,
            depth=config.model_depth,
            widen_factor=config.model_width,
            dtype=config.dtype,
            pixel_mean=PIXEL_MEAN,
            pixel_std=PIXEL_STD,
            num_classes=config.num_classes,
            num_planes=config.model_planes,
            num_blocks=tuple(
                int(b) for b in config.model_blocks.split(",")
            ) if config.model_blocks is not None else None,
            first_conv=config.first_conv,
            first_pool=config.first_pool,
            return_emb=return_emb,
        )
    else:
        raise NotImplementedError

    if config.model_style == "FRN-Swish":
        model = partial(
            _ResNet,
            conv=partial(
                flax.linen.Conv,
                use_bias=not config.model_nobias,
                kernel_init=jax.nn.initializers.he_normal(),
                bias_init=jax.nn.initializers.zeros),
            norm=FilterResponseNorm,
            relu=flax.linen.swish)
    elif config.model_style == 'BN-ReLU':
        model = _ResNet
    else:
        raise NotImplementedError
    return model


# MLP for student network
def get_mlp(config):
    score_func = partial(
        Mlp,        
        hidden_size=config.hidden_size,
        time_embed_dim=config.time_embed_dim,
        num_blocks=config.num_blocks,
        num_classes=config.num_classes,
        droprate=config.droprate,
        time_scale=config.time_scale,
    )
    return score_func


# U-Net for student network
def get_unet(config):
    score_func = partial(
        ClsUnet,        
        num_classes=config.num_classes,
        ch=config.ch,
        droprate=config.droprate,
        time_scale=config.time_scale,
    )
    return score_func


# Transformer for student network
def get_transformer(config):
    score_func = partial(
        DiT,        
        hidden_size=config.hidden_size,
        depth=config.depth,
        num_heads=config.num_heads,
        mlp_ratio=config.mlp_ratio,
        droprate=config.droprate,
    )
    return score_func


# Build model for flow matching
def build_model(config):
    resnet = get_resnet(config, return_emb=True)
    score_net = get_mlp(config)
    edfm = FlowMatching(
        res_net=resnet,
        score_net=score_net,
        noise_var=config.noise_var,
        num_classes=config.num_classes,
        eps=config.train_time_trunc,
        base=config.train_time_exp_base,
    )
    return edfm


# Sample from flow matching
def fm_sample(score, l0, z, config, num_models):
    batch_size = l0.shape[0]

    # Sample step scheduling
    a = config.sample_time_exp_base
    steps = config.sample_num_steps
    timesteps = jnp.array([(1-a**i)/(1-a**steps) 
                           for i in range(steps+1)])

    @jax.jit
    def euler_solver(n, l_n):
        current_t = jnp.array([timesteps[n]])
        current_t = jnp.tile(current_t, [batch_size])

        next_t = jnp.array([timesteps[n+1]])
        next_t = jnp.tile(next_t, [batch_size])

        eps = score(l_n, z, t=current_t)
        euler_l_n = l_n + batch_mul(next_t-current_t, eps)
        return euler_l_n, eps
    
    @jax.jit
    def heun_solver(n, l_n):
        current_t = jnp.array([timesteps[n]])
        current_t = jnp.tile(current_t, [batch_size])

        next_t = jnp.array([timesteps[n+1]])
        next_t = jnp.tile(next_t, [batch_size])
    
        euler_l_n, eps = euler_solver(n, l_n)
        eps2 = score(euler_l_n, z, t=next_t)
        heun_l_n = l_n + batch_mul((next_t-current_t)/2, eps+eps2)
        return heun_l_n

    val = l0
    for i in range(0, steps-1):
        val = heun_solver(i, val)
    val, _ = euler_solver(steps-1, val)

    logits = val.reshape(-1, num_models, config.num_classes)
    prob = jax.nn.softmax(logits).mean(1)
    return prob, logits


def launch(config):
    # Resume from checkpoint
    if config.resume:
        def props(cls):   
            return [i for i in cls.__dict__.keys() if i[:1] != '_']
        print(f"Loading checkpoint {config.resume}")
        saved_params, saved_batch_stats, saved_config, tx_saved = load_saved(
            config.resume)
        _config = get_config(saved_config)
        for k in props(_config):
            v = getattr(_config, k)
            setattr(config, k, v)

    rng = jax.random.PRNGKey(config.seed)
    model_dtype = jnp.float32
    config.dtype = model_dtype

    # Load image dataset
    dataloaders = build_dataloaders(config)
    config.num_classes = dataloaders["num_classes"]
    config.trn_steps_per_epoch = dataloaders["trn_steps_per_epoch"]

    # Prepare teachers for distillation
    # Assuming that the teacher models are saved as pickle files...
    swag_state_list = []
    for ckpt in os.listdir(config.teacher_dir):
        with open(f'{config.teacher_dir}/{ckpt}', 'rb') as fp:
            ckpt = pickle.load(fp)
            swag_state = ckpt['swag_state']
        swag_state = namedtuple('SWAGState', swag_state.keys())(*swag_state.values())
        swag_state_list.append(swag_state)

    # Define and load resnet
    resnet = get_resnet(config)
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

    # Build model (both frozen teacher and student)
    print("Building model")
    edfm = build_model(config)

    # Initialize model
    print("Initializing model")
    _, h, w, d = dataloaders["image_shape"]
    x_dim = (h, w, d)
    init_rng, sub_rng = jax.random.split(rng)
    variables = edfm.init(
        {"params": init_rng, "dropout": init_rng},
        rng=init_rng,
        l0=jnp.empty((1, config.num_classes)),
        x=jnp.empty((1, *x_dim)),
        training=False
    )
    if config.resume:
        variables = variables.unfreeze()
        variables["params"] = saved_params
        variables = freeze(variables)

    # Define optimizers
    scheduler = optax.cosine_decay_schedule(
        init_value=config.optim_lr,
        decay_steps=config.optim_ne * config.trn_steps_per_epoch
    )
    if config.optim_base == "adam":
        base_optim = optax.adamw(learning_rate=scheduler, 
                                 weight_decay=config.optim_weight_decay)
    elif config.optim_base == "sgd":
        base_optim = optax.chain(
            optax.add_decayed_weights(config.optim_weight_decay),
            optax.sgd(learning_rate=scheduler, 
                      momentum=config.optim_momentum)
        )
    else:
        raise NotImplementedError
    
    partition_optimizers = {
        "resnet": optax.set_to_zero(), # Frozen teacher
        "score": base_optim, # Student
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
    
    # Create train state
    state_rng, sub_rng = jax.random.split(sub_rng)
    state = TrainState.create(
        apply_fn=edfm.apply,
        params=variables["params"],
        ema_params=variables["params"],
        tx=optimizer,
        batch_stats=variables.get("batch_stats"),
        rng=state_rng
    )

    # Load ResNet
    pretrained_resnet_param, _, _, _ = load_saved(config.pretrained_resnet)
    state = state.replace(params=freeze(dict(resnet=pretrained_resnet_param,
                                             score=state.params["score"])),)
    if config.resume:
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

    # Helper functions for loss and metrics
    @jax.jit
    def mse_loss(noise, output):
        p = config.mse_power
        sum_axis = list(range(1, len(output.shape[1:])+1))
        loss = jnp.sum(jnp.abs(noise-output)**p, axis=sum_axis)
        return loss

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

    # Label with teacher predictions
    @partial(jax.pmap, axis_name="batch")
    def step_label(state, batch):
        rng = state.rng
        swag_state = random.choice(swag_state_list)
        swag_param = sample_swag_diag(1, rng, swag_state)[0]
        rng, _ = jax.random.split(rng)
        res_params_dict = dict(params=swag_param)
        
        images = batch["images"]
        logits = forward_resnet(res_params_dict, images)
        logits = logits - logits.mean(-1, keepdims=True)
        batch["logits"] = logits
        return batch

    # Loss function
    def loss_func(params, state, batch, train=True):
        logits = batch["logits"]
        
        drop_rng, score_rng = jax.random.split(state.rng)
        params_dict = pdict(
            params=params,
            batch_stats=state.batch_stats
        )
        rngs_dict = dict(dropout=drop_rng)
        mutable = ["batch_stats"]
        output = state.apply_fn(
            params_dict, score_rng,
            logits, batch["images"],
            training=train,
            rngs=rngs_dict,
            **(dict(mutable=mutable) if train else dict()),
        )
        new_model_state = output[1] if train else None
        epsilon, u_t = output[0] if train else output 
        total_loss = reduce_mean(mse_loss(epsilon, u_t), batch["marker"])

        count = jnp.sum(batch["marker"])
        metrics = OrderedDict({
            "loss": total_loss*count,
            "count": count,
        })
        return total_loss, (metrics, new_model_state)

    def get_loss_func():
        return loss_func

    # Training step
    @partial(jax.pmap, axis_name="batch")
    def step_train(state, batch):
        def loss_fn(params):
            _loss_func = get_loss_func()
            return _loss_func(params, state, batch)
        
        (_, (metrics, new_model_state)), grads = jax.value_and_grad(
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

    # Metrics
    def evaluate_accnll(prob, labels, marker):
        acc = evaluate_acc(
            prob, labels, log_input=False, reduction="none")
        nll = evaluate_nll(
            prob, labels, log_input=False, reduction='none')
        
        acc = reduce_sum(acc, marker)
        nll = reduce_sum(nll, marker)
        return acc, nll

    # Sample function
    def sample_func(state, batch):
        labels = batch["labels"]
        drop_rng, score_rng = jax.random.split(state.rng)

        params_dict = pdict(
            params=state.ema_params,
            batch_stats=state.batch_stats
        )
        rngs_dict = dict(dropout=drop_rng)
        model_bd = edfm.bind(params_dict, rngs=rngs_dict)
        
        # Ensemble
        _fm_sample = partial(
            fm_sample, config=config, num_models=config.num_ensembles)
        prob_ens, _ = model_bd.sample(
            score_rng, _fm_sample, batch["images"], config.num_ensembles)
        
        # Single model
        _fm_sample = partial(
            fm_sample, config=config, num_models=1)
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

    # Sampling step
    @partial(jax.pmap, axis_name="batch")
    def step_sample(state, batch):
        metrics = sample_func(state, batch)
        metrics = jax.lax.psum(metrics, axis_name="batch")
        return metrics
 
    # Mixup
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

    cross_replica_mean = jax.pmap(lambda x: jax.lax.pmean(x, 'x'), 'x')
    best_acc = -float("inf")
    best_nll = float("inf")
    train_summary = dict()
    val_summary = dict()
    state = jax_utils.replicate(state)

    # WandB logging
    wandb.init(
        project="fmed",
        name=config.exp_name,
        config=vars(config),
        mode="disabled" if config.nowandb else "online"
    )
    wandb.run.summary["params_resnet"] = (
        sum(x.size for x in jax.tree_util.tree_leaves(
            variables["params"]["resnet"]))
    )
    wandb.run.summary["params_score"] = (
        sum(x.size for x in jax.tree_util.tree_leaves(
            variables["params"]["score"]))
    )
    wl = WandbLogger()

    def summarize_metrics(metrics, key="trn"):
        metrics = common_utils.get_metrics(metrics)
        summarized = {
            f"{key}/{k}": v for k, v in jax.tree_util.tree_map(lambda e: e.sum(0), metrics).items()}
        for k, v in summarized.items():
            if "count" in k:
                continue
            elif "lr" in k:
                continue
            summarized[k] /= summarized[f"{key}/count"]
        del summarized[f"{key}/count"]
        return summarized

    # RandAugment
    augment_1 = image_processing.RandAugment(
        num_layers=2, magnitude=9.0,
        cutout_const=16.0, translate_const=8.0,
        magnitude_std=0.5, prob_to_apply=0.5)
    
    augment_2 = jax.pmap(jax.vmap(
        image_processing.TransformChain([
            image_processing.RandomCropTransform(size=32, padding=4),
            image_processing.RandomHFlipTransform(prob=0.5)]),
                         axis_name="batch"))
    
    def augment_fn(rng, img):
        aug_img = augment_1.distort(
            tf.convert_to_tensor(img.reshape(-1, *img.shape[-3:]))
        )
        aug_img = jnp.array(aug_img).reshape(img.shape)
        aug_img = augment_2(
            jax.vmap(lambda r: jax.random.split(r, aug_img.shape[1]))(rng),
            aug_img,
        )
        return aug_img

    for epoch_idx in tqdm(range(config.optim_ne)):
        epoch_rng = jax.random.fold_in(sub_rng, epoch_idx)
        train_rng, val_rng = jax.random.split(epoch_rng, 2)

        # Train
        train_loader = dataloaders["dataloader"](rng=epoch_rng)
        train_loader = jax_utils.prefetch_to_device(train_loader, size=2)
        train_metrics = []
        for batch_idx, batch in enumerate(train_loader):
            batch_rng = jax.random.fold_in(train_rng, batch_idx)
            state = state.replace(rng=jax_utils.replicate(batch_rng))
            # RandAug + Mixup
            batch["images"] = augment_fn(jax_utils.replicate(batch_rng),
                                         (255.0*batch["images"]).astype(jnp.uint8))
            batch["images"] /= 255.0
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

        # Validation
        val_loader = dataloaders["val_loader"](rng=None)
        val_loader = jax_utils.prefetch_to_device(val_loader, size=2)
        val_metrics = []
        for batch_idx, batch in enumerate(val_loader):
            batch_rng = jax.random.fold_in(val_rng, batch_idx)
            state = state.replace(rng=jax_utils.replicate(batch_rng))
            acc_metrics = step_sample(state, batch)
            metrics = acc_metrics
            val_metrics.append(metrics)

        val_summarized = summarize_metrics(val_metrics, "val")
        val_summary.update(val_summarized)
        wl.log(val_summary)

        acc_criteria = val_summarized[f"val/acc_ens"]
        nll_criteria = val_summarized[f"val/nll_ens"]

        if (best_acc <= acc_criteria) or (best_nll >= nll_criteria):
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
        
        wl.flush()

    wandb.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='./data/', type=str,
                        help='root directory containing datasets (default: ./data/)')
    parser.add_argument('--seed', default=2025, type=int)
    parser.add_argument('--save', default=None, type=str)
    parser.add_argument("--config", default=None, type=str)
    parser.add_argument('--exp_name', type=str)
    parser.add_argument("--nowandb", action="store_true")
    parser.add_argument('--resume', default=None, type=str)
    
    args, argv = parser.parse_known_args(sys.argv[1:])
    if args.config is not None:
        import yaml
        with open(args.config, 'r') as f:
            arg_defaults = yaml.safe_load(f)

    if args.config is not None:
        parser.set_defaults(**arg_defaults)

    args = parser.parse_args()

    if args.save is not None:
        args.save = os.path.abspath(args.save)
        if os.path.exists(args.save):
            raise AssertionError(f'already existing args.save = {args.save}')
        os.makedirs(args.save, exist_ok=True)

    launch(args)

if __name__ == '__main__':
    main()