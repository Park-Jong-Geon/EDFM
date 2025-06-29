import numpy as np
import wandb
import time
from tqdm import tqdm
import argparse
from giung2.metrics import evaluate_acc, evaluate_nll
from models.resnet import FlaxResNet
from giung2.models.layers import FilterResponseNorm
from data.build import build_dataloaders
from flax.training import common_utils, train_state, checkpoints
from flax import jax_utils
import flax
import orbax
import optax
import jax.numpy as jnp
import jax
from collections import OrderedDict
from typing import Any
from functools import partial
import os
import sys

from utils.utils import WandbLogger
sys.path.append('./')
np.random.seed(0)

from models.swag import swag, sample_swag_diag
import copy

PIXEL_MEAN = (0.49, 0.48, 0.44)
PIXEL_STD = (0.2, 0.2, 0.2)

class TrainState(train_state.TrainState):
    image_stats: Any
    batch_stats: Any

def update_swag_batch_stats(state, params_dict, batch, batch_stats):
    mutable = ["intermediates", "batch_stats"]
    params_dict["batch_stats"] = batch_stats
    _, new_state = state.apply_fn(
        params_dict,
        batch['images'],
        mutable=mutable,
        use_running_average=False,
    )
    return new_state['batch_stats']
        
def launch(config):    
    rng = jax.random.PRNGKey(config.seed)

    # specify precision
    model_dtype = jnp.float32
    if config.precision == 'fp16':
        model_dtype = jnp.bfloat16 if jax.local_devices(
        )[0].platform == 'tpu' else jnp.float16

    # build dataloaders
    dataloaders = build_dataloaders(config)

    # build model
    if config.model_name == 'FlaxResNet':
        _ResNet = partial(
            FlaxResNet,
            depth=config.model_depth,
            widen_factor=config.model_width,
            dtype=model_dtype,
            pixel_mean=PIXEL_MEAN,
            pixel_std=PIXEL_STD,
            num_classes=dataloaders['num_classes'],
            num_planes=config.model_planes,
            num_blocks=tuple(
                int(b) for b in config.model_blocks.split(",")
            ) if config.model_blocks is not None else None,
            first_conv=config.first_conv,
            first_pool=config.first_pool
        )

    if config.model_style == 'BN-ReLU':
        model = _ResNet()
    elif config.model_style == "FRN-Swish":
        model = _ResNet(
            conv=partial(
                flax.linen.Conv,
                use_bias=not config.model_nobias,
                kernel_init=jax.nn.initializers.he_normal(),
                bias_init=jax.nn.initializers.zeros),
            norm=FilterResponseNorm,
            relu=flax.linen.swish)
    
    # initialize model
    def initialize_model(key, model):
        @jax.jit
        def init(*args):
            return model.init(*args)
        return init({'params': key}, jnp.ones(dataloaders['image_shape'], model.dtype))
    _, init_rng = jax.random.split(rng)
    variables = initialize_model(init_rng, model)
    if variables.get('batch_stats'):
        initial_batch_stats = copy.deepcopy(variables['batch_stats'])

    trn_steps_per_epoch = dataloaders['trn_steps_per_epoch']
    scheduler = optax.join_schedules(
        schedules=[
            optax.linear_schedule(
                init_value       = 0.0,
                end_value        = config.optim_lr,
                transition_steps = config.start_decay_epoch * trn_steps_per_epoch,
            ),
            optax.cosine_decay_schedule(
                init_value       = config.optim_lr,
                decay_steps      = (config.start_swa_epoch - config.start_decay_epoch) * trn_steps_per_epoch,
                alpha            = config.optim_swa_lr / config.optim_lr,
            ),
            optax.constant_schedule(
                value            = config.optim_swa_lr,
            ),
        ], boundaries=[
            config.start_decay_epoch * trn_steps_per_epoch,
            config.start_swa_epoch * trn_steps_per_epoch,
        ]
    )
    
    if config.optim == "sgd":
        optimizer = optax.chain(
            optax.sgd(learning_rate=scheduler, momentum=config.optim_momentum),
            swag(config.freq, config.rank, config.start_swa_epoch * trn_steps_per_epoch),
        )
    elif config.optim == "adam":
        optimizer = optax.chain(
            optax.adamw(learning_rate=scheduler, weight_decay=config.optim_weight_decay),
            swag(config.freq, config.rank, config.start_swa_epoch * trn_steps_per_epoch),
        )

    # build train state
    state = TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=optimizer,
        image_stats=variables.get('image_stats'),
        batch_stats=variables.get('batch_stats'),)

    # ---------------------------------------------------------------------- #
    # Optimization
    # ---------------------------------------------------------------------- #
    def step_trn(state, batch, config, scheduler):
        def pred(params, logits=False):
            params_dict = dict(params=params)
            mutable = ["intermediates"]
            if state.image_stats is not None:
                params_dict["image_stats"] = state.image_stats
            if state.batch_stats is not None:
                params_dict["batch_stats"] = state.batch_stats
                mutable.append("batch_stats")

            _, new_model_state = state.apply_fn(
                params_dict, batch['images'],
                rngs=None,
                mutable=mutable,
                use_running_average=False)
            if logits:
                return new_model_state["intermediates"]["cls.logit"][0]
            return new_model_state

        def loss_fn(params):
            new_model_state = pred(params)
            # compute loss
            # [B, K,]
            logits = new_model_state['intermediates']['cls.logit'][0]
            target = common_utils.onehot(
                batch['labels'], num_classes=logits.shape[-1])  # [B, K,]
            
            predictions = jax.nn.log_softmax(logits, axis=-1)
            loss = -jnp.sum(target * predictions, axis=-1)      # [B,]
            count = jnp.sum(batch['marker'])
            loss = jnp.sum(
                jnp.where(batch['marker'], loss, jnp.zeros_like(loss))
            )
            loss = jnp.where(count!=0, loss/count, loss)

            # accuracy
            acc = evaluate_acc(
                predictions, batch['labels'], log_input=True, reduction='none')          # [B,]
            nll = evaluate_nll(
                predictions, batch['labels'], log_input=True, reduction='none')          # [B,]

            # refine and return metrics
            acc = jnp.sum(jnp.where(batch['marker'], acc, jnp.zeros_like(acc)))
            nll = jnp.sum(jnp.where(batch['marker'], nll, jnp.zeros_like(nll)))
            cnt = jnp.sum(batch['marker'])
            # log metrics
            metrics = OrderedDict(
                {'loss': loss, "acc": acc/cnt, "nll": nll/cnt})
            return loss, (metrics, new_model_state)

        aux, grads = jax.value_and_grad(
            loss_fn, has_aux=True)(state.params)
        grads = jax.lax.pmean(grads, axis_name='batch')

        # weight decay regularization in PyTorch-style
        if config.optim == "sgd":
            grads = jax.tree_util.tree_map(
                lambda g, p: g + config.optim_weight_decay * p, grads, state.params)

        # get auxiliaries
        metrics, new_model_state = aux[1]
        metrics = jax.lax.pmean(metrics, axis_name='batch')
        metrics['lr'] = scheduler(state.step)

        # update train state
        if new_model_state.get("batch_stats") is not None:
            new_state = state.apply_gradients(
                grads=grads, batch_stats=new_model_state['batch_stats'])
        else:
            new_state = state.apply_gradients(grads=grads)

        return new_state, metrics

    def step_val(state, batch):
        params_dict = dict(params=state.opt_state[1].mean)
        if state.image_stats is not None:
            params_dict["image_stats"] = state.image_stats
        if state.batch_stats is not None:
            params_dict["batch_stats"] = state.opt_state[1].swa_batch_stats

        begin = time.time()
        _, new_model_state = state.apply_fn(
            params_dict, batch['images'],
            rngs=None,
            mutable='intermediates',
            use_running_average=True)
        sec = time.time() - begin

        # compute metrics
        logits = new_model_state['intermediates']['cls.logit'][0]
        predictions = jax.nn.log_softmax(
            logits, axis=-1)  # [B, K,]
        target = common_utils.onehot(
            batch['labels'], num_classes=logits.shape[-1])  # [B, K,]
        loss = -jnp.sum(target * predictions, axis=-1)      # [B,]
        acc = evaluate_acc(
            predictions, batch['labels'], log_input=True, reduction='none')          # [B,]
        nll = evaluate_nll(
            predictions, batch['labels'], log_input=True, reduction='none')          # [B,]

        # refine and return metrics
        loss = jnp.sum(jnp.where(batch['marker'], loss, jnp.zeros_like(loss)))
        acc = jnp.sum(jnp.where(batch['marker'], acc, jnp.zeros_like(acc)))
        nll = jnp.sum(jnp.where(batch['marker'], nll, jnp.zeros_like(nll)))
        cnt = jnp.sum(batch['marker'])

        metrics = OrderedDict(
            {"loss": loss, 'acc': acc, 'nll': nll, 'cnt': cnt, "sec": sec*cnt})
        metrics = jax.lax.psum(metrics, axis_name='batch')
        return metrics

    def update_batch_stats(state, swag_state, batch):
        params_dict = dict(params=swag_state.mean)
        mutable = ["intermediates", "batch_stats"]
        params_dict["batch_stats"] = swag_state.swa_batch_stats
        if state.image_stats is not None:
            params_dict["image_stats"] = state.image_stats
                
        _, new_model_state = state.apply_fn(
            params_dict,
            batch['images'],
            mutable=mutable,
            use_running_average=False,
        )
        return swag_state._replace(
            swa_batch_stats = new_model_state['batch_stats'],
        )
    
    def step_ens_val(state, batch):
        swag_state = state.opt_state[1]
        swag_param_list = sample_swag_diag(config.num_swag_samples, rng, swag_state)
        
        logit_list = []
        sec = 0
        for swag_param in swag_param_list:
            params_dict = dict(params=swag_param)
            if state.image_stats is not None:
                params_dict["image_stats"] = state.image_stats
            if state.batch_stats is not None:
                # Update batch_stats
                trn_loader = dataloaders['dataloader'](rng=data_rng)
                trn_loader = jax_utils.prefetch_to_device(trn_loader, size=2)
                batch_stats = initial_batch_stats
                for _, batch in enumerate(trn_loader, start=1):
                    batch_stats = update_swag_batch_stats(state, params_dict, batch, batch_stats)
                params_dict["batch_stats"] = cross_replica_mean(batch_stats)

            begin = time.time()
            _, new_model_state = state.apply_fn(
                params_dict, batch['images'],
                rngs=None,
                mutable='intermediates',
                use_running_average=True)
            sec_ = time.time() - begin
            
            logits = new_model_state['intermediates']['cls.logit'][0]
            logit_list.append(logits)
            
            sec += sec_
        
        # compute metrics
        _logits = jnp.stack(logit_list)
        logprobs = jax.nn.log_softmax(_logits, axis=-1)
        ens_logprobs = jax.scipy.special.logsumexp(logprobs, axis=0) - np.log(logprobs.shape[0])
        
        target = common_utils.onehot(
            batch['labels'], num_classes=logits.shape[-1])  # [B, K,]
        loss = -jnp.sum(target * ens_logprobs, axis=-1)      # [B,]
        acc = evaluate_acc(
            ens_logprobs, batch['labels'], log_input=True, reduction='none')          # [B,]
        nll = evaluate_nll(
            ens_logprobs, batch['labels'], log_input=True, reduction='none')          # [B,]

        # refine and return metrics
        loss = jnp.sum(jnp.where(batch['marker'], loss, jnp.zeros_like(loss)))
        acc = jnp.sum(jnp.where(batch['marker'], acc, jnp.zeros_like(acc)))
        nll = jnp.sum(jnp.where(batch['marker'], nll, jnp.zeros_like(nll)))
        cnt = jnp.sum(batch['marker'])

        metrics = OrderedDict(
            {"loss": loss, 'acc': acc, 'nll': nll, 'cnt': cnt, "sec": sec*cnt})
        metrics = jax.lax.psum(metrics, axis_name='batch')
        return metrics
    
    cross_replica_mean = jax.pmap(lambda x: jax.lax.pmean(x, 'x'), 'x')
    p_step_trn = jax.pmap(partial(step_trn, config=config,
                          scheduler=scheduler), axis_name='batch')
    p_step_val = jax.pmap(step_val,
                          axis_name='batch')
    p_step_ens_val = jax.pmap(step_ens_val,
                          axis_name='batch')
    state = jax_utils.replicate(state)
    best_acc = 0.

    wandb.init(
        project="teacher",
        name=config.exp_name,
        config=vars(config),
        mode="disabled" if config.nowandb else "online"
    )
    wandb.define_metric("val/loss", summary="min")
    wandb.define_metric("val/nll", summary="min")
    wandb.define_metric("val/acc", summary="max")
    wandb.run.summary["params"] = sum(
        x.size for x in jax.tree_util.tree_leaves(variables["params"]))
    wl = WandbLogger()

    for epoch_idx, _ in enumerate(tqdm(range(config.optim_ne)), start=1):
        rng, data_rng = jax.random.split(rng)

        # ---------------------------------------------------------------------- #
        # Train
        # ---------------------------------------------------------------------- #
        trn_metric = []
        trn_loader = dataloaders['dataloader'](rng=data_rng)
        trn_loader = jax_utils.prefetch_to_device(trn_loader, size=2)
        
        for batch_idx, batch in enumerate(trn_loader, start=1):
            batch_rng = jax.random.fold_in(rng, batch_idx)
            state, metrics = p_step_trn(state, batch)
            trn_metric.append(metrics)

        trn_metric = common_utils.get_metrics(trn_metric)
        trn_summarized = {f'trn/{k}': v for k,
                          v in jax.tree_util.tree_map(lambda e: e.mean(), trn_metric).items()}
        wl.log(trn_summarized)

        if state.batch_stats is not None:
            # synchronize batch normalization statistics
            state = state.replace(
                batch_stats=cross_replica_mean(state.batch_stats))

            opt_state = state.opt_state[0]
            swag_state = state.opt_state[1]
            swag_state = swag_state._replace(swa_batch_stats=initial_batch_stats)
            
            trn_loader = dataloaders['dataloader'](rng=data_rng)
            trn_loader = jax_utils.prefetch_to_device(trn_loader, size=2)
            for batch_idx, batch in enumerate(trn_loader, start=1):
                swag_state = update_batch_stats(state, swag_state, batch)
            
            swag_state = swag_state._replace(
                swa_batch_stats=cross_replica_mean(swag_state.swa_batch_stats))
            state = state.replace(opt_state=(opt_state, swag_state))

        # ---------------------------------------------------------------------- #
        # Valid
        # ---------------------------------------------------------------------- #
        val_metric = []
        val_ens_metric = []
        val_loader = dataloaders['val_loader'](rng=None)
        val_loader = jax_utils.prefetch_to_device(val_loader, size=2)
        for batch_idx, batch in enumerate(val_loader, start=1):
            metrics = p_step_val(state, batch)
            ens_metrics = p_step_ens_val(state, batch)
            val_metric.append(metrics)
            val_ens_metric.append(ens_metrics)
        val_metric = common_utils.get_metrics(val_metric)
        val_summarized = {f'val/{k}': v for k,
                          v in jax.tree_util.tree_map(lambda e: e.sum(), val_metric).items()}
        val_summarized['val/loss'] /= val_summarized['val/cnt']
        val_summarized['val/acc'] /= val_summarized['val/cnt']
        val_summarized['val/nll'] /= val_summarized['val/cnt']
        val_summarized['val/sec'] /= val_summarized['val/cnt']
        del val_summarized['val/cnt']
        val_summarized.update(trn_summarized)
        wl.log(val_summarized)
        
        val_ens_metric = common_utils.get_metrics(val_ens_metric)
        val_ens_summarized = {f'val_ens/{k}': v for k,
                          v in jax.tree_util.tree_map(lambda e: e.sum(), val_ens_metric).items()}
        val_ens_summarized['val_ens/loss'] /= val_ens_summarized['val_ens/cnt']
        val_ens_summarized['val_ens/acc'] /= val_ens_summarized['val_ens/cnt']
        val_ens_summarized['val_ens/nll'] /= val_ens_summarized['val_ens/cnt']
        val_ens_summarized['val_ens/sec'] /= val_ens_summarized['val_ens/cnt']
        del val_ens_summarized['val_ens/cnt']
        val_ens_summarized.update(trn_summarized)
        wl.log(val_ens_summarized)

        # ---------------------------------------------------------------------- #
        # Save
        # ---------------------------------------------------------------------- #
        tst_metric = []
        tst_ens_metric = []
        tst_loader = dataloaders['tst_loader'](rng=None)
        tst_loader = jax_utils.prefetch_to_device(tst_loader, size=2)
        for batch_idx, batch in enumerate(tst_loader, start=1):
            metrics = p_step_val(state, batch)
            ens_metrics = p_step_ens_val(state, batch)
            tst_metric.append(metrics)
            tst_ens_metric.append(ens_metrics)
        tst_metric = common_utils.get_metrics(tst_metric)
        tst_summarized = {
            f'tst/{k}': v for k, v in jax.tree_util.tree_map(lambda e: e.sum(), tst_metric).items()}
        tst_summarized['tst/loss'] /= tst_summarized['tst/cnt']
        tst_summarized['tst/acc'] /= tst_summarized['tst/cnt']
        tst_summarized['tst/nll'] /= tst_summarized['tst/cnt']
        tst_summarized['tst/sec'] /= tst_summarized['tst/cnt']
        del tst_summarized["tst/cnt"]
        wl.log(tst_summarized)
        
        tst_ens_metric = common_utils.get_metrics(tst_ens_metric)
        tst_ens_summarized = {
            f'tst_ens/{k}': v for k, v in jax.tree_util.tree_map(lambda e: e.sum(), tst_ens_metric).items()}
        tst_ens_summarized['tst_ens/loss'] /= tst_ens_summarized['tst_ens/cnt']
        tst_ens_summarized['tst_ens/acc'] /= tst_ens_summarized['tst_ens/cnt']
        tst_ens_summarized['tst_ens/nll'] /= tst_ens_summarized['tst_ens/cnt']
        tst_ens_summarized['tst_ens/sec'] /= tst_ens_summarized['tst_ens/cnt']
        del tst_ens_summarized["tst_ens/cnt"]
        wl.log(tst_ens_summarized)
        
        best_acc = val_summarized["val/acc"]

        if config.save:
            save_state = jax_utils.unreplicate(state)
            ckpt = dict(model=save_state, config=vars(
                config), best_acc=best_acc)
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

    parser.add_argument("--config", default=None, type=str)
    args, argv = parser.parse_known_args(sys.argv[1:])
    config_f = args.config
    if config_f is not None:
        import yaml
        with open(config_f, 'r') as f:
            arg_defaults = yaml.safe_load(f)

    # Data
    parser.add_argument('--data_root', default='./data/', type=str,
                        help='root directory containing datasets (default: ./data/)')

    # ResNet
    parser.add_argument('--model_depth', default=44, type=int,
                        choices=[20, 32, 44, 56, 110])
    parser.add_argument('--model_width', default=2, type=int,
                        help='widen factor (default: 1)')
    parser.add_argument('--model_style', default='BN-ReLU', type=str,
                        choices=['BN-ReLU', "FRN-Swish"])
    parser.add_argument('--model_name', default='FlaxResNet', type=str,
                        choices=['FlaxResNet', "FlaxPreResNet"])
    parser.add_argument("--model_planes", default=16, type=int)
    parser.add_argument("--model_blocks", default=None, type=str)
    parser.add_argument('--model_nobias', action="store_true")
    parser.add_argument('--first_conv', nargs='+', type=int)
    parser.add_argument('--first_pool', nargs='+', type=int)

    # Training
    parser.add_argument("--optim", default="sgd", type=str,
                        choices=["sgd", "adam"])
    parser.add_argument('--optim_bs', default=64, type=int,
                        help='mini-batch size (default: 256)')
    parser.add_argument('--optim_ne', default=200, type=int,
                        help='the number of training epochs (default: 200)')
    parser.add_argument('--optim_lr', default=0.005, type=float,
                        help='base learning rate (default: 0.1)')
    parser.add_argument('--optim_momentum', default=0.9, type=float,
                        help='momentum coefficient (default: 0.9)')
    parser.add_argument('--optim_weight_decay', default=0.001, type=float,
                        help='weight decay coefficient (default: 0.0001)')
    parser.add_argument("--warmup_factor", default=0.01, type=float)
    parser.add_argument("--warmup_steps", default=0, type=int)

    # SWAG
    parser.add_argument("--freq", default=None, type=float, help='frequency of SWAG update')
    parser.add_argument("--rank", default=None, type=float, help='rank of SWAG update')

    # Others
    parser.add_argument('--save', default=None, type=str,
                        help='save the *.log and *.ckpt files if specified (default: False)')
    parser.add_argument('--seed', default=2023, type=int,
                        help='random seed for training (default: None)')
    parser.add_argument('--precision', default='fp32', type=str,
                        choices=['fp16', 'fp32'])
    parser.add_argument('--exp_name', type=str)
    parser.add_argument("--nowandb", action="store_true")

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