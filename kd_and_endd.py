from builtins import NotImplementedError
import os
import sys
import copy
import argparse
import pickle
import random
from typing import Any
import wandb
from collections import OrderedDict, namedtuple
from functools import partial
from tqdm import tqdm

import jax
import jax.numpy as jnp
import flax
from flax.training import train_state, common_utils, checkpoints
from flax import jax_utils
import orbax
import optax

import tensorflow as tf

from data.build import build_dataloaders
from giung2.metrics import evaluate_acc, evaluate_nll
from giung2.models.layers import FilterResponseNorm

from models.resnet import FlaxResNet
from utils.utils import WandbLogger

from models.swag import sample_swag_diag
from metrics.losses import KD, ProxyEnDD
import utils.image_processing as image_processing

random.seed(0)

PIXEL_MEAN = (0.49, 0.48, 0.44)
PIXEL_STD = (0.2, 0.2, 0.2)

class TrainState(train_state.TrainState):
    rng: Any
    image_stats: Any = None
    batch_stats: Any = None


def baseline_method(config):
    if config.baseline_method == "KD":
        return KD(temperature=config.dist_temp)
    elif config.baseline_method == "ProxyEnDD":
        return ProxyEnDD(temperature=config.dist_temp,
                         s_offset=config.s_offset,
                         t_offset=config.t_offset,
                         dtype=jnp.float32,
                         eps=config.eps)
    else:
        raise NotImplementedError


def get_resnet(config):
    if config.model_name == 'FlaxResNet':
        _ResNet = partial(
            FlaxResNet,
            depth=config.model_depth,
            widen_factor=config.model_width,
            dtype=jnp.float32,
            pixel_mean=PIXEL_MEAN,
            pixel_std=PIXEL_STD,
            num_classes=config.num_classes,
            num_planes=config.model_planes,
            num_blocks=tuple(
                int(b) for b in config.model_blocks.split(",")
            ) if config.model_blocks is not None else None,
        )
    else:
        raise NotImplementedError

    if config.model_style == "FRN-Swish":
        model = _ResNet(
            conv=partial(
                flax.linen.Conv,
                use_bias=True,
                kernel_init=jax.nn.initializers.he_normal(),
                bias_init=jax.nn.initializers.zeros),
            norm=FilterResponseNorm,
            relu=flax.linen.swish)
    elif config.model_style == 'BN-ReLU':
        model = _ResNet()
    else:
        raise NotImplementedError
    
    return model


def initialize_model(key, shape, model):
    @jax.jit
    def init(*args):
        return model.init(*args)
    return init({'params': key}, jnp.ones(shape, model.dtype))


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


def launch(config):
    rng = jax.random.PRNGKey(config.seed)

    # build dataloaders
    dataloaders = build_dataloaders(config)

    # build model
    model = get_resnet(config)

    # initialize model
    init_rng, sub_rng = jax.random.split(rng)
    variables = initialize_model(init_rng, dataloaders['image_shape'], model)
    if variables.get('batch_stats'):
        initial_batch_stats = copy.deepcopy(variables['batch_stats'])

    # define optimizer with scheduler
    scheduler = optax.warmup_cosine_decay_schedule(
        init_value=config.warmup_factor*config.optim_lr,
        peak_value=config.optim_lr,
        warmup_steps=config.warmup_steps,
        decay_steps=config.optim_ne * dataloaders['trn_steps_per_epoch'])
    if config.optim == "sgd":
        optimizer = optax.sgd(
            learning_rate=scheduler,
            momentum=config.optim_momentum)
    elif config.optim == "adam":
        optimizer = optax.adamw(learning_rate=scheduler,
                                weight_decay=config.optim_weight_decay)

    # build train state
    state_rng, sub_rng = jax.random.split(sub_rng)
    state = TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=optimizer,
        image_stats=variables.get('image_stats'),
        batch_stats=variables.get('batch_stats'),
        rng=state_rng
    )

    # prepare teachers for distillation
    # assuming that the teacher models are saved as pickle files...
    swag_state_list = []
    for ckpt in os.listdir(config.teacher_dir):
        with open(f'{config.teacher_dir}/{ckpt}', 'rb') as fp:
            ckpt = pickle.load(fp)
            swag_state = ckpt['swag_state']
        swag_state = namedtuple('SWAGState', swag_state.keys())(*swag_state.values())
        swag_state_list.append(swag_state)

    # forward through resnet
    def pred(params, state, batch, train):
        params_dict = dict(params=params)
        mutable = ["intermediates"]
        if state.image_stats is not None:
            params_dict["image_stats"] = state.image_stats
        if state.batch_stats is not None:
            params_dict["batch_stats"] = state.batch_stats
            mutable.append("batch_stats")

        output = state.apply_fn(
            params_dict, batch['images'],
            rngs=None,
            **(dict(mutable=mutable) if train else dict()),
            use_running_average=False)
        if train:
            new_model_state = output[1]
            return new_model_state["intermediates"]["cls.logit"][0], new_model_state
        else:
            return output, None

    # define step collecting features and logits
    @partial(jax.pmap, axis_name="batch")
    def step_label(state, batch):
        rng = state.rng
        swag_param_list = []    
        for _ in range(config.num_teachers):
            swag_state = random.choice(swag_state_list)
            swag_params_per_seed = sample_swag_diag(1, rng, swag_state)
            swag_param_list += swag_params_per_seed
            rng, _ = jax.random.split(rng)
            
        logitsA = []
        for swag_param in swag_param_list:
            logits0, _ = pred(swag_param, state, batch, False)
            logitsA.append(logits0)
        logitsB = logitsA[0]
        logitsA = jnp.stack(logitsA)

        batch["logitsB"] = logitsB
        batch["logitsA"] = logitsA
        return batch


    def loss_func(params, state, batch, train=True):
        s_logits, new_model_state = pred(params, state, batch, train)
        t_logits = batch["logitsA"]

        loss = baseline_method(config)(s_logits, t_logits)

        predictions = jax.nn.log_softmax(s_logits, axis=-1)
        # accuracy
        acc = evaluate_acc(
            predictions, batch['labels'], log_input=True, reduction='none') # [B,]
        nll = evaluate_nll(
            predictions, batch['labels'], log_input=True, reduction='none') # [B,]

        # refine and return metrics
        acc = jnp.sum(jnp.where(batch['marker'], acc, jnp.zeros_like(acc)))
        nll = jnp.sum(jnp.where(batch['marker'], nll, jnp.zeros_like(nll)))
        cnt = jnp.sum(batch['marker'])
        # log metrics
        metrics = OrderedDict(
            {'loss': loss, "acc": acc, "nll": nll, "cnt": cnt})

        return loss, (metrics, new_model_state)

    @partial(jax.pmap, axis_name="batch")
    def step_trn(state, batch):
        def loss_fn(params):
            return loss_func(params, state, batch)

        # compute losses and gradients
        aux, grads = jax.value_and_grad(
            loss_fn, has_aux=True)(state.params)
        grads = jax.lax.pmean(grads, axis_name='batch')

        # weight decay regularization in PyTorch-style
        if config.optim == "sgd":
            grads = jax.tree_util.tree_map(
                lambda g, p: g + config.optim_weight_decay * p, grads, state.params)

        # get auxiliaries
        metrics, new_model_state = aux[1]
        metrics = jax.lax.psum(metrics, axis_name='batch')

        # update train state
        if new_model_state.get("batch_stats") is not None:
            new_state = state.apply_gradients(
                grads=grads, batch_stats=new_model_state['batch_stats'])
        else:
            new_state = state.apply_gradients(grads=grads)

        return new_state, metrics

    @partial(jax.pmap, axis_name="batch")
    def step_val(state, batch):
        _, (metrics, _) = loss_func(state.params, state, batch)
        metrics = jax.lax.psum(metrics, axis_name='batch')
        return metrics

    cross_replica_mean = jax.pmap(lambda x: jax.lax.pmean(x, 'x'), 'x')
    state = jax_utils.replicate(state)
    best_acc = -float("inf")
    best_nll = float("inf")

    wandb.init(
        project=config.baseline_method,
        name=config.exp_name,
        config=vars(config),
        mode="disabled" if config.nowandb else "online"
    )
    wandb.run.summary["params"] = sum(
        x.size for x in jax.tree_util.tree_leaves(variables["params"]))
    wl = WandbLogger()

    # define mixup
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

    def summarize_metrics(metrics, key="trn"):
        metrics = common_utils.get_metrics(metrics)
        summarized = {
            f"{key}/{k}": v for k, v in jax.tree_util.tree_map(lambda e: e.sum(0), metrics).items()}
        for k, v in summarized.items():
            if "cnt" in k:
                continue
            elif "lr" in k:
                continue
            summarized[k] /= summarized[f"{key}/cnt"]
        del summarized[f"{key}/cnt"]
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
        epoch_rng = jax.random.fold_in(rng, epoch_idx)

        # train
        trn_metric = []
        trn_loader = dataloaders['dataloader'](rng=epoch_rng)
        trn_loader = jax_utils.prefetch_to_device(trn_loader, size=2)
        for batch_idx, batch in enumerate(trn_loader):
            batch_rng = jax.random.fold_in(epoch_rng, batch_idx)
            state = state.replace(rng=jax_utils.replicate(batch_rng))
            
            # RandAug + Mixup
            batch["images"] = augment_fn(jax_utils.replicate(batch_rng),
                                         (255.0*batch["images"]).astype(jnp.uint8))
            batch["images"] /= 255.0
            
            if config.mixup_alpha > 0:
                batch = step_mixup(state, batch)
            
            batch = step_label(state, batch)
            state, metrics = step_trn(state, batch)
            trn_metric.append(metrics)
        trn_summarized = summarize_metrics(trn_metric, "trn")
        trn_summarized['lr'] = scheduler(jax_utils.unreplicate(state.step))
        wl.log(trn_summarized)

        if state.batch_stats is not None:
            # synchronize batch normalization statistics
            state = state.replace(
                batch_stats=cross_replica_mean(state.batch_stats))

        # save
        val_metric = []
        val_loader = dataloaders['val_loader'](rng=None)
        val_loader = jax_utils.prefetch_to_device(val_loader, size=2)
        for batch_idx, batch in enumerate(val_loader):
            batch = step_label(state, batch)
            metrics = step_val(state, batch)
            val_metric.append(metrics)
        val_summarized = summarize_metrics(val_metric, "val")
        wl.log(val_summarized)
        
        acc_criteria = val_summarized[f"val/acc"]
        nll_criteria = val_summarized[f"val/nll"]

        if (best_acc <= acc_criteria) or (best_nll >= nll_criteria):
            best_acc = val_summarized["val/acc"]
            best_nll = val_summarized["val/nll"]
            if config.save:
                save_state = jax_utils.unreplicate(state)
                ckpt = dict(
                    params=save_state.params,
                    batch_stats=save_state.batch_stats,
                    image_stats=save_state.image_stats,
                    config=vars(config),
                    best_acc=best_acc)
                orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
                checkpoints.save_checkpoint(ckpt_dir=config.save,
                                            target=ckpt,
                                            step=epoch_idx,
                                            overwrite=True,
                                            orbax_checkpointer=orbax_checkpointer)
        wl.flush()

        # wait until computations are done
        jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()
        if jnp.isnan(trn_summarized['trn/loss']):
            print("loss has NaN")
            break

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