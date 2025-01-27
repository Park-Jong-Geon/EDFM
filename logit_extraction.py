from builtins import NotImplementedError
import os
import sys
import copy
import math
import argparse
import pickle
import random
from typing import Any
import numpy as np
import datetime
from collections import namedtuple
from functools import partial
from tqdm import tqdm

import jax
import jax.numpy as jnp
import flax
from flax.training import train_state, checkpoints
from flax.core.frozen_dict import freeze
from flax import jax_utils
import orbax
import optax

from giung2.models.layers import FilterResponseNorm
from giung2.data import image_processing

from models.resnet import FlaxResNet
from models.mlp import Mlp
from models.flowmatching import FlowMatching

from utils import batch_mul, get_config

from swag import sample_swag_diag

random.seed(0)

PIXEL_MEAN = (0.49, 0.48, 0.44)
PIXEL_STD = (0.2, 0.2, 0.2)

def _build_dataloader(images, labels, batch_size, transform):
    # add padding to process the entire dataset...
    marker = np.ones([len(images),], dtype=bool)
    num_batches = math.ceil(len(images) / batch_size)

    padded_images = np.concatenate([
        images, np.zeros([num_batches*batch_size - len(images), *images.shape[1:]], images.dtype)])
    # padded_labels = np.concatenate([
    #     labels, np.zeros([num_batches*batch_size - len(labels), *labels.shape[1:]], labels.dtype)])
    padded_marker = np.concatenate([
        marker, np.zeros([num_batches*batch_size - len(images), *marker.shape[1:]], marker.dtype)])

    # define generator using yield...
    local_device_count = jax.local_device_count()
    batch_indices = jnp.arange(len(padded_images)).reshape(
        (num_batches, batch_size))
    for batch_idx in batch_indices:
        batch = {'images': jnp.array(padded_images[batch_idx]),
                #  'labels': jnp.array(padded_labels[batch_idx]),
                 'marker': jnp.array(padded_marker[batch_idx]), }
        batch['images'] = transform(None, batch['images'])
        batch = jax.tree_util.tree_map(
            lambda x: x.reshape((local_device_count, -1) + x.shape[1:]), batch)
        yield batch

def build_dataloaders(data_root, data_name, batch_size, image_shape, num_classes):
    """
    Args:
        data_root (str) : root directory containing datasets (e.g., ./data/).
        data_name (str) : name of the dataset (e.g., CIFAR10_x32).
        batch_size (int) : batch size.
        image_shape (tuple) : shape of the image (e.g., (1, 32, 32, 3)).
        num_classes (int) : number of classes.

    Return:
        dataloaders (dict) : contains dataloader.
    """
    
    images = np.load(os.path.join(data_root, f'{data_name}/test_images.npy'))
    # labels = np.load(os.path.join(data_root, f'{data_name}/test_labels.npy'))
    
    print(f'Input shape: {images.shape}')

    # Dataloader
    dataloaders = dict()
    val_transform = jax.jit(jax.vmap(image_processing.ToTensorTransform()))
    dataloaders['dataloader'] = partial(
        _build_dataloader,
        images=images,
        labels = None,
        # labels=labels,
        batch_size=batch_size,
        transform=val_transform)
    dataloaders['steps_per_epoch'] = math.ceil(len(images) / batch_size)
    dataloaders['image_shape'] = image_shape
    dataloaders['num_classes'] = num_classes

    return dataloaders


class TrainState(train_state.TrainState):
    rng: Any
    ema_params: Any
    batch_stats: Any = None


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


def load_saved(ckpt_dir):
    ckpt = checkpoints.restore_checkpoint(
        ckpt_dir=ckpt_dir,
        target=None
    )
    params = ckpt["params"]
    ema_params = ckpt.get("ema_params", params)
    batch_stats = ckpt.get("batch_stats")
    return params, ema_params, batch_stats


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


def pdict(params, batch_stats=None, image_stats=None):
    params_dict = dict(params=params)
    if batch_stats is not None:
        params_dict["batch_stats"] = batch_stats
    if image_stats is not None:
        params_dict["image_stats"] = image_stats
    return params_dict


def get_resnet(config, return_emb=False):
    config.model_name = 'FlaxResNet'
    config.model_style = 'FRN-Swish'
    config.model_depth = 32
    config.model_width = 2 #ResNet32x4
    config.model_planes = 16
    config.model_blocks = None
    config.dtype = jnp.float32
    config.num_classes = 10 #ResNet32x4
    config.first_conv = None
    config.first_pool = None
    config.model_nobias = None
    
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


def get_scorenet(config):
    config.hidden_size = 256 # ResNet32x4
    config.time_embed_dim = 32
    config.num_blocks = 4
    config.num_classes = 10 # ResNet32x4
    config.droprate = 0.
    config.time_scale = 1000.

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


def build_dbn(config):
    config.var = 4
    config.num_classes = 10
    config.train_timestep_truncation = 3
    config.train_timestep_alpha = 0.5
    
    resnet = get_resnet(config, return_emb=True)
    score_net = get_scorenet(config)
    dbn = FlowMatching(
        res_net=resnet,
        score_net=score_net,
        var=config.var,
        num_classes=config.num_classes,
        eps=config.train_timestep_truncation,
        train_timestep_alpha=config.train_timestep_alpha,
    )
    return dbn


# def fm_sample(score, l0, z, config, num_models):
#     batch_size = l0.shape[0]
#     steps = 200
#     timesteps = jnp.linspace(0., 1., steps+1)
#     # a = config.sample_timestep_alpha
#     # timesteps = jnp.array([(1-a**i)/(1-a**steps) for i in range(steps+1)])

#     @jax.jit
#     def body_fn(n, l_n):
#         current_t = jnp.array([timesteps[n]])
#         current_t = jnp.tile(current_t, [batch_size])

#         next_t = jnp.array([timesteps[n+1]])
#         next_t = jnp.tile(next_t, [batch_size])

#         eps = score(l_n, z, t=current_t)
#         euler_l_n = l_n + batch_mul(next_t-current_t, eps)
#         # return euler_l_n
        
#         eps2 = score(euler_l_n, z, t=next_t)
#         heun_l_n = l_n + batch_mul((next_t-current_t)/2, eps+eps2)
        
#         return heun_l_n

#     val = l0
#     for i in range(0, steps-1):
#         val = body_fn(i, val)
#     current_t = jnp.array([timesteps[steps-1]])
#     current_t = jnp.tile(current_t, [batch_size])

#     next_t = jnp.array([timesteps[steps]])
#     next_t = jnp.tile(next_t, [batch_size])

#     eps = score(val, z, t=current_t)
#     val += batch_mul(next_t-current_t, eps)
        
#     prob = jax.nn.softmax(val).reshape(-1, num_models, config.num_classes).mean(1)
#     logits = val.reshape(-1, num_models, config.num_classes)
#     return prob, logits
def fm_sample(score, l0, z, config, num_models):
    config.mid_step = 0.8
    
    batch_size = l0.shape[0]
    
    zero = jnp.array([0.])
    zero = jnp.tile(zero, [batch_size])

    mid = jnp.array([config.mid_step])
    mid = jnp.tile(mid, [batch_size])

    one = jnp.array([1.])
    one = jnp.tile(one, [batch_size])

    val = l0
    eps = score(val, z, t=zero)
    euler = val + batch_mul(mid-zero, eps)
    
    eps2 = score(euler, z, t=mid)
    val += batch_mul((mid-zero)/2, eps+eps2)

    eps = score(val, z, t=mid)
    val += batch_mul(one-mid, eps)
        
    prob = jax.nn.softmax(val).reshape(-1, num_models, config.num_classes).mean(1)
    logits = val.reshape(-1, num_models, config.num_classes)
    return prob, logits


def launch(config):
    # ------------------------------------------------------------------------
    # load image dataset (C10, C100, TinyImageNet, ImageNet)
    # ------------------------------------------------------------------------
    dataloaders = build_dataloaders(config.data_root, 
                                    config.data_name, 
                                    config.batch_size, 
                                    config.image_shape, 
                                    config.num_classes)

    # ------------------------------------------------------------------------
    # define and load resnet
    # ------------------------------------------------------------------------
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
    
    if config.mode == 'fm':
        print(f"Loading checkpoint {config.saved_model_path}...")
        saved_params, saved_ema_params, saved_batch_stats = load_saved(config.saved_model_path)
        
        rng = jax.random.PRNGKey(config.seed)
        config.dtype = jnp.float32
        config.image_stats = dict(
            m=jnp.array(PIXEL_MEAN),
            s=jnp.array(PIXEL_STD))

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

        variables = variables.unfreeze()
        variables["params"] = saved_params
        variables["ema_params"] = saved_ema_params
        if variables.get("batch_stats") is not None:
            variables["batch_stats"] = saved_batch_stats
        variables = freeze(variables)
        
        # ------------------------------------------------------------------------
        # create train state
        # ------------------------------------------------------------------------
        base_optim = partial(
            optax.adamw, learning_rate=0.1, weight_decay=0.
        )
        
        partition_optimizers = {
            "resnet": base_optim(), #optax.set_to_zero() 
            "score": base_optim(),
        }
        def tagging(path, v):
            if "resnet" in path:
                return "resnet"
            elif "score" in path:
                return "score"
            else:
                print(path)
                print(v)
                raise NotImplementedError
        partitions = flax.core.freeze(
            flax.traverse_util.path_aware_map(tagging, variables["params"]))
        optimizer = optax.multi_transform(partition_optimizers, partitions)
    
        state_rng, sub_rng = jax.random.split(sub_rng)
        state = TrainState.create(
            apply_fn=dbn.apply,
            params=variables["params"],
            ema_params=variables["ema_params"],
            tx=optimizer,
            batch_stats=variables.get("batch_stats"),
            rng=state_rng
        )

    elif config.mode == 'kd' or config.mode == 'endd':
        rng = jax.random.PRNGKey(config.seed)
        init_rng, sub_rng = jax.random.split(rng)

        config.image_stats = dict(
            m=jnp.array(PIXEL_MEAN),
            s=jnp.array(PIXEL_STD))

        resnet_state, _, _, _ = load_resnet(config.saved_model_path)
        params = resnet_state['params']
        # d = resnet_state['model']['opt_state']['1']
        # swag_state = namedtuple('SWAGState', d.keys())(*d.values())

    elif config.mode == 'teacher':
        state = None
        
        rng = jax.random.PRNGKey(config.seed)
        init_rng, sub_rng = jax.random.split(rng)
        
        swag_state_list = []
        for s in [2, 5, 11, 17, 23, 31, 41, 47, 59, 67]:
            ckpt = f'checkpoints_teacher/c10/{s}.pickle' #ResNet32x4
            with open(ckpt, 'rb') as fp:
                ckpt = pickle.load(fp)
                swag_state = ckpt['swag_state']
                batch_stats = ckpt['batch_stats']
                image_stats = ckpt['image_stats']
            swag_state = namedtuple('SWAGState', swag_state.keys())(*swag_state.values())
            swag_state_list.append(swag_state)
    else:
        raise NotImplementedError

    # ------------------------------------------------------------------------
    # collect logits
    # ------------------------------------------------------------------------    
    @partial(jax.pmap, axis_name="batch")
    def extract_from_teacher(batch, state, rng):
        swag_param_list = []  
        # MultiSWA
        for i in range(10):
            swag_param_list += [swag_state_list[i].mean]

        # # MultiSWAG          
        # for _ in range(config.num_samples):
        #     swag_state = random.choice(swag_state_list)
        #     swag_params_per_seed = sample_swag_diag(1, rng, swag_state)
        #     swag_param_list += swag_params_per_seed
        #     rng, _ = jax.random.split(rng)
            
        logitsA = []
        for swag_param in swag_param_list:
            res_params_dict = dict(params=swag_param)
            if image_stats is not None:
                res_params_dict["image_stats"] = image_stats
            if batch_stats is not None:
                # Update batch_stats
                trn_loader = dataloaders['dataloader'](rng=rng)
                trn_loader = jax_utils.prefetch_to_device(trn_loader, size=2)
                iter_batch_stats = initial_batch_stats
                for _, batch in enumerate(trn_loader, start=1):
                    iter_batch_stats = update_swag_batch_stats(resnet, res_params_dict, batch, iter_batch_stats)
                res_params_dict["batch_stats"] = cross_replica_mean(iter_batch_stats)
            
            images = batch["images"]
            logits0 = forward_resnet(res_params_dict, images)
            logitsA.append(logits0)
            
        batch["logitsA"] = jnp.stack(logitsA, axis=1)
        return batch

    @partial(jax.pmap, axis_name="batch")
    def extract_from_fm(batch, state, rng):
        drop_rng, score_rng = jax.random.split(state.rng)
        
        params_dict = pdict(
            params=state.ema_params,
            image_stats=config.image_stats,
            batch_stats=state.batch_stats)
        rngs_dict = dict(dropout=drop_rng)
        
        model_bd = dbn.bind(params_dict, rngs=rngs_dict)
        _fm_sample = partial(
            fm_sample, config=config, num_models=config.num_samples)
        _, val = model_bd.sample(
            score_rng, _fm_sample, batch["images"], config.num_samples)
        
        batch["logitsA"] = val
        return batch

    @partial(jax.pmap, axis_name="batch")
    def extract_from_kd(batch, params, rng):
        params_dict = pdict(
            params=params,
            image_stats=config.image_stats,
            batch_stats=None) # no batch stats for now        
        batch["logitsA"] = forward_resnet(params_dict, batch["images"])
        return batch

    assert config.mode in ['kd', 'endd', 'fm', 'teacher']
    if config.mode == 'teacher':
        extract = extract_from_teacher
    elif config.mode == 'fm':
        extract = extract_from_fm
    elif config.mode == 'kd' or config.mode == 'endd':
        extract = extract_from_kd
    else:
        raise NotImplementedError

    # ------------------------------------------------------------------------
    # init settings
    # ------------------------------------------------------------------------
    cross_replica_mean = jax.pmap(lambda x: jax.lax.pmean(x, 'x'), 'x')
    # state = params
    state = jax_utils.replicate(state)
        
    data_loader = dataloaders["dataloader"]()
    data_loader = jax_utils.prefetch_to_device(data_loader, size=2)
    
    extracted_logits = []
    for batch_idx, batch in tqdm(enumerate(data_loader)):
        batch_rng = jax.random.fold_in(sub_rng, batch_idx)
        batch = extract(batch, state, jax_utils.replicate(batch_rng))
        # save logits
        logits = batch["logitsA"]
        # extracted_logits.append(logits.reshape(-1, *logits.shape[-1:]))
        extracted_logits.append(logits.reshape(-1, *logits.shape[-2:]))
    extracted_logits = jnp.stack(extracted_logits)
    # extracted_logits = extracted_logits.reshape(-1, *extracted_logits.shape[-1:])
    extracted_logits = extracted_logits.reshape(-1, *extracted_logits.shape[-2:])
    
    print(f'Output shape: {extracted_logits.shape}')
    np.save(f'{config.save_path}{config.data_name}_{config.mode}_{config.name}.npy',
            np.array(extracted_logits))
    print(f'Logits saved')
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str)
    parser.add_argument('--saved_model_path', type=str)
    parser.add_argument('--data_name', type=str, help='name of the dataset (e.g., CIFAR10_x32)')
    parser.add_argument('--name', type=str, help='name of the experiment')
    
    parser.add_argument('--seed', default=2025, type=int)
    parser.add_argument('--data_root', default='./data/', type=str,
                        help='root directory containing datasets (default: ./data/)')
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--image_shape', default=(1, 32, 32, 3), type=tuple)
    parser.add_argument('--num_classes', default=10, type=int)
    parser.add_argument('--num_samples', default=30, type=int)
    parser.add_argument('--save_path', default='./logits/', type=str, 
                        help='path to save the extracted logits')
    
    args, argv = parser.parse_known_args(sys.argv[1:])
    args = parser.parse_args()

    if args.seed < 0:
        args.seed = (
            os.getpid()
            + int(datetime.datetime.now().strftime('%S%f'))
            + int.from_bytes(os.urandom(2), 'big')
        )

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