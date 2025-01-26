from builtins import NotImplementedError
import os
import sys
import copy
import argparse
import pickle
import random
from typing import Any
import datetime
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

from data.build import build_dataloaders
from giung2.metrics import evaluate_acc, evaluate_nll, get_optimal_temperature, temperature_scaling
from giung2.models.layers import FilterResponseNorm

from models.resnet import FlaxResNet
from models.resnet_fed import FlaxResNetforFED
from utils import WandbLogger

from swag import sample_swag_diag
from baselines.losses import MMD

random.seed(0)

PIXEL_MEAN = (0.49, 0.48, 0.44)
PIXEL_STD = (0.2, 0.2, 0.2)

class TrainState(train_state.TrainState):
    rng: Any
    image_stats: Any = None
    batch_stats: Any = None


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


def get_resnet_for_fed(config):
    if config.model_name == 'FlaxResNet':
        _ResNet = partial(
            FlaxResNetforFED,
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
            noise_std=config.noise_std,
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
    return init({'params': key}, 
                jnp.ones(shape, model.dtype), 
                jnp.ones(shape, model.dtype), 
                jax.random.PRNGKey(0))

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
    model = get_resnet_for_fed(config)

    # initialize model
    init_rng, sub_rng = jax.random.split(rng)
    variables = initialize_model(init_rng, dataloaders['image_shape'], model)
    if variables.get('batch_stats'):
        initial_batch_stats = copy.deepcopy(variables['batch_stats'])

    # define optimizer with scheduler
    trn_steps_per_epoch = dataloaders["trn_steps_per_epoch"]
    b_and_s = {}
    for epoch in config.optim_boundaries_and_scales.keys():
        b_and_s[epoch*trn_steps_per_epoch] = config.optim_boundaries_and_scales[epoch]
    scheduler = optax.piecewise_constant_schedule(
        init_value=config.optim_lr,
        # boundaries_and_scales=config.optim_boundaries_and_scales,
    )
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
    swag_state_list = []
    for s in [2, 5, 11, 17, 23, 31, 41, 47, 59, 67]:
        ckpt = {'CIFAR10_x32': 'c10', 'CIFAR100_x32': 'c100'}[config.data_name]
        ckpt = f'checkpoints_teacher/{ckpt}/{s}.pickle'
        with open(ckpt, 'rb') as fp:
            ckpt = pickle.load(fp)
            swag_state = ckpt['swag_state']
            batch_stats = ckpt['batch_stats']
            image_stats = ckpt['image_stats']
        swag_state = namedtuple('SWAGState', swag_state.keys())(*swag_state.values())
        swag_state_list.append(swag_state)

    # swag_state_list = []
    # for swag_ckpt_dir in config.swag_ckpt_dir:
    #     resnet_state, _, batch_stats, image_stats = load_resnet(swag_ckpt_dir)
    #     d = resnet_state['model']['opt_state']['1']
    #     swag_state = namedtuple('SWAGState', d.keys())(*d.values())
    #     swag_state_list.append(swag_state)

    # ------------------------------------------------------------------------
    # define and load resnet
    # ------------------------------------------------------------------------
    resnet = get_resnet(config)
    # resnet = resnet()
    @jax.jit
    def forward_resnet(params_dict, images):
        mutable = ["intermediates"]
        _, state = resnet.apply(
            params_dict, images, rngs=None,
            mutable=mutable,
            training=False, use_running_average=True)
        logits = state["intermediates"]["cls.logit"][0]
        return logits

    # forward through resnet
    def pred(params, state, images, noise, noise_rng, train):
        params_dict = dict(params=params)
        mutable = ["intermediates"]
        if state.image_stats is not None:
            params_dict["image_stats"] = state.image_stats
        if state.batch_stats is not None:
            params_dict["batch_stats"] = state.batch_stats
            mutable.append("batch_stats")

        output = state.apply_fn(
            params_dict, images, noise, noise_rng,
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
            res_params_dict = dict(params=swag_param)
            logits0 = forward_resnet(res_params_dict, batch['images'])
            logitsA.append(logits0)
        logitsB = logitsA[0]
        logitsA = jnp.stack(logitsA)

        batch["logitsB"] = logitsB
        batch["logitsA"] = logitsA
        return batch


    def loss_func(params, state, batch, train=True):
        image = jnp.repeat(batch['images'], repeats=config.num_teachers, axis=0)
        noise = jax.random.normal(state.rng, image.shape) * config.noise_std
        
        s_logits, new_model_state = pred(params, state, image, noise, state.rng, train)
        s_probs = jax.nn.softmax(s_logits)
        s_probs = s_probs.reshape(config.num_teachers, -1)
        
        t_logits = batch["logitsA"]
        t_probs = jax.nn.softmax(t_logits)
        t_probs = t_probs.reshape(config.num_teachers, -1)

        loss = MMD(bandwidth_range=config.bandwidth_range, kernel=config.kernel)(s_probs, t_probs)

        cnt = jnp.sum(batch['marker'])
        # log metrics
        metrics = OrderedDict({'loss': loss, 'cnt': cnt})

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


    @jax.jit
    def reduce_sum(loss, marker):
        loss = jnp.where(marker, loss, 0).sum()
        return loss


    def evaluate_accnll(prob, labels, marker):
        acc = evaluate_acc(
            prob, labels, log_input=False, reduction="none")
        nll = evaluate_nll(
            prob, labels, log_input=False, reduction='none')
        
        T = get_optimal_temperature(prob, labels, log_input=False)
        scaled_prob = temperature_scaling(prob, T, log_input=False)
        cnll = evaluate_nll(
            scaled_prob, labels, log_input=False, reduction='none')
        
        acc = reduce_sum(acc, marker)
        nll = reduce_sum(nll, marker)
        cnll = reduce_sum(cnll, marker)
        return acc, nll, cnll
    

    def sample_func(state, batch):
        labels = batch["labels"]
        
        images = jnp.repeat(batch['images'], repeats=config.num_ensembles, axis=0)
        noise = jax.random.normal(state.rng, images.shape) * config.noise_std
        pred_logits, _ = pred(state.params, state, images, noise, state.rng, False)
        pred_logits = pred_logits.reshape(-1, config.num_ensembles, config.num_classes)
        pred_probs = jax.nn.softmax(pred_logits)
        
        prob_ens = pred_probs.mean(axis=1)
        prob_1 = pred_probs[:, 0, :]

        acc_ens, nll_ens, cnll_ens = evaluate_accnll(prob_ens, labels, batch["marker"])
        acc_1, nll_1, cnll_1 = evaluate_accnll(prob_1, labels, batch["marker"])
        metrics = OrderedDict({
            'cnt': jnp.sum(batch["marker"]),
        })
        metrics[f"acc_ens"] = acc_ens
        metrics[f"nll_ens"] = nll_ens
        metrics[f"cnll_ens"] = cnll_ens
        metrics[f"acc_1"] = acc_1
        metrics[f"nll_1"] = nll_1
        metrics[f"cnll_1"] = cnll_1
        
        ens_logits = batch["logitsA"].swapaxes(0, 1)
        ens_var = pred_var = 0
        for e, p in zip(ens_logits, pred_logits[0:config.num_teachers]):
            ens_cov = jnp.cov(e, rowvar=False)
            pred_cov = jnp.cov(p, rowvar=False)
            ens_var += jnp.trace(jnp.abs(ens_cov))
            pred_var += jnp.trace(jnp.abs(pred_cov))
        metrics[f"ens_var"] = ens_var
        metrics[f"pred_var"] = pred_var
        
        return metrics
    

    @partial(jax.pmap, axis_name="batch")
    def step_sample(state, batch):
        metrics = sample_func(state, batch)
        metrics = jax.lax.psum(metrics, axis_name="batch")
        return metrics

    cross_replica_mean = jax.pmap(lambda x: jax.lax.pmean(x, 'x'), 'x')
    # p_step_trn = jax.pmap(partial(step_trn, config=config,
    #                       scheduler=scheduler), axis_name='batch')
    # p_step_val = jax.pmap(step_val,
    #                       axis_name='batch')
    state = jax_utils.replicate(state)
    best_acc = -float("inf")
    best_nll = float("inf")

    wandb.init(
        project="fed",
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

    for epoch_idx in tqdm(range(config.optim_ne)):
        epoch_rng = jax.random.fold_in(rng, epoch_idx)

        # ---------------------------------------------------------------------- #
        # Train
        # ---------------------------------------------------------------------- #
        trn_metric = []
        trn_loader = dataloaders['dataloader'](rng=epoch_rng)
        trn_loader = jax_utils.prefetch_to_device(trn_loader, size=2)
        for batch_idx, batch in enumerate(trn_loader):
            batch_rng = jax.random.fold_in(epoch_rng, batch_idx)
            state = state.replace(rng=jax_utils.replicate(batch_rng))
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

        # ---------------------------------------------------------------------- #
        # Save
        # ---------------------------------------------------------------------- #
        tst_metric = []
        tst_loader = dataloaders['tst_loader'](rng=None)
        tst_loader = jax_utils.prefetch_to_device(tst_loader, size=2)
        for batch_idx, batch in enumerate(tst_loader):
            batch = step_label(state, batch)
            metrics = step_sample(state, batch)
            tst_metric.append(metrics)
        tst_summarized = summarize_metrics(tst_metric, "tst")
        wl.log(tst_summarized)
        
        acc_criteria = tst_summarized[f"tst/acc_ens"]
        nll_criteria = tst_summarized[f"tst/nll_ens"]

        if (best_acc <= acc_criteria) or (best_nll >= nll_criteria):
            best_acc = tst_summarized["tst/acc_ens"]
            best_nll = tst_summarized["tst/nll_ens"]
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

        # # ---------------------------------------------------------------------- #
        # # Valid
        # # ---------------------------------------------------------------------- #
        # val_metric = []
        # val_loader = dataloaders['val_loader'](rng=None)
        # val_loader = jax_utils.prefetch_to_device(val_loader, size=2)
        # for batch_idx, batch in enumerate(val_loader):
        #     metrics = step_val(state, batch)
        #     val_metric.append(metrics)
        # val_summarized = summarize_metrics(val_metric, "val")
        # wl.log(val_summarized)

        # # ---------------------------------------------------------------------- #
        # # Save
        # # ---------------------------------------------------------------------- #
        # test_condition = best_acc < val_summarized["val/acc"]
        # if test_condition:
        #     tst_metric = []
        #     tst_loader = dataloaders['tst_loader'](rng=None)
        #     tst_loader = jax_utils.prefetch_to_device(tst_loader, size=2)
        #     for batch_idx, batch in enumerate(tst_loader):
        #         metrics = step_val(state, batch)
        #         tst_metric.append(metrics)
        #     tst_summarized = summarize_metrics(tst_metric, "tst")
        #     wl.log(tst_summarized)
        #     best_acc = val_summarized["val/acc"]

        #     if config.save:
        #         save_state = jax_utils.unreplicate(state)
        #         ckpt = dict(
        #             params=save_state.params,
        #             batch_stats=getattr(save_state, "batch_stats", None),
        #             image_stats=getattr(save_state, "image_stats", None),
        #             config=vars(config),
        #             best_acc=best_acc)
        #         orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        #         checkpoints.save_checkpoint(ckpt_dir=config.save,
        #                                     target=ckpt,
        #                                     step=epoch_idx,
        #                                     overwrite=True,
        #                                     orbax_checkpointer=orbax_checkpointer)
        wl.flush()

        # wait until computations are done
        jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()
        if jnp.isnan(trn_summarized['trn/loss']):
            print("loss has NaN")
            break

    wandb.finish()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', default=2025, type=int)
    parser.add_argument('--save', default=None, type=str)
    # parser.add_argument('--dtype', default=jnp.float32, type=jnp.dtype)
    parser.add_argument('--data_root', default='./data/', type=str,
                        help='root directory containing datasets (default: ./data/)')
    parser.add_argument('--data_augmentation', default='standard', type=str,
                        choices=['standard',])
    parser.add_argument('--data_proportional', default=1.0, type=float,
                        help='use the proportional train split if specified (default: 1.0)')
    parser.add_argument("--config", default=None, type=str)
    parser.add_argument("--nowandb", action="store_true")
    args, argv = parser.parse_known_args(sys.argv[1:])
    config_f = args.config
    if args.config is not None:
        import yaml
        with open(config_f, 'r') as f:
            arg_defaults = yaml.safe_load(f)

    if args.config is not None:
        parser.set_defaults(**arg_defaults)

    args = parser.parse_args()

    if args.seed < 0:
        args.seed = (
            os.getpid()
            + int(datetime.datetime.now().strftime('%S%f'))
            + int.from_bytes(os.urandom(2), 'big')
        )

    if args.save is not None:
        args.save = os.path.abspath(args.save)
        if os.path.exists(args.save):
            raise AssertionError(f'already existing args.save = {args.save}')
        os.makedirs(args.save, exist_ok=True)

    print_fn = partial(print, flush=True)
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

    launch(args)


if __name__ == '__main__':
    main()