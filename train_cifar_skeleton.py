import math
import os
import pickle
from argparse import ArgumentParser
from collections import namedtuple
from datetime import datetime

import tensorflow as tf
# import tensorflow_models as tfm
# tf.config.set_visible_devices([], 'GPU')
from image_processing import RandAugment

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import image_processing


if __name__ == '__main__':

    ############################################################################
    parser = ArgumentParser()

    parser.add_argument('--data_root', type=str, default='./data/')
    parser.add_argument('--data_name', type=str)
    parser.add_argument('--data_augmentation', type=str, default='simple',
                        choices=['simple', 'rand_aug'])

    parser.add_argument('--mixup_alpha', type=float, default=0.0)

    parser.add_argument('--num_steps', type=int, default=160000)
    parser.add_argument('--num_batch', type=int, default=256)
    parser.add_argument('--num_train', type=int, default=40960)
    parser.add_argument('--num_swag_samples', type=int, default=10)

    parser.add_argument('--save', type=str, default=None)
    parser.add_argument('--seed', type=int, default=None)

    args = parser.parse_args()

    if args.seed is None:
        args.seed = (
            os.getpid()
            + int(datetime.now().strftime('%S%f'))
            + int.from_bytes(os.urandom(2), 'big'))

    ############################################################################
    augment_fn = lambda rng, img: img

    if args.data_augmentation == 'simple':
        augment_1 = jax.jit(jax.vmap(
            image_processing.TransformChain([
                image_processing.RandomCropTransform(size=32, padding=4),
                image_processing.RandomHFlipTransform(prob=0.5)])))
        augment_fn = lambda rng, img: augment_1(
            jax.random.split(rng, img.shape[0]), img)

    if args.data_augmentation == 'rand_aug':
        augment_1 = RandAugment(
            num_layers=2, magnitude=9.0,
            cutout_const=16.0, translate_const=8.0,
            magnitude_std=0.5, prob_to_apply=0.5)
        augment_2 = jax.jit(jax.vmap(
            image_processing.TransformChain([
                image_processing.RandomCropTransform(size=32, padding=4),
                image_processing.RandomHFlipTransform(prob=0.5)])))
        augment_fn = lambda rng, img: augment_2(
            jax.random.split(rng, img.shape[0]),
            np.asarray(augment_1.distort(tf.convert_to_tensor(img))))

    shard_shape = (jax.local_device_count(), -1)
    input_shape = (32, 32, 3)
    num_classes = 100 #{'cifar10': 10, 'cifar100': 100}[args.data_name]

    trn_images = np.load(os.path.join(
        args.data_root, args.data_name, 'train_images.npy'))[:args.num_train]
    trn_labels = np.load(os.path.join(
        args.data_root, args.data_name, 'train_labels.npy'))[:args.num_train]

    val_images = np.load(os.path.join(
        args.data_root, args.data_name, 'train_images.npy'))[args.num_train:]
    val_labels = np.load(os.path.join(
        args.data_root, args.data_name, 'train_labels.npy'))[args.num_train:]

    PIXEL_MEAN = np.array([0.49, 0.48, 0.44]).reshape((1, 1, 1, 3))
    PIXEL_STD = np.array([0.2, 0.2, 0.2]).reshape((1, 1, 1, 3))

    ############################################################################
    batch_rng = jax.random.PRNGKey(args.seed)
    batch_queue = np.asarray(
        jax.random.permutation(batch_rng, args.num_train))

    for step_idx in range(1, args.num_steps + 1):

        if batch_queue.shape[0] <= args.num_batch:
            batch_rng = jax.random.split(batch_rng)[0]
            batch_queue = np.concatenate((
                batch_queue, jax.random.permutation(batch_rng, args.num_train)))
        batch_index = batch_queue[:args.num_batch]
        batch_queue = batch_queue[args.num_batch:]

        batch = {}
        batch_rng, mixup_rng = jax.random.split(batch_rng)
        batch['inputs'] = augment_fn(batch_rng, trn_images[batch_index])
        if args.mixup_alpha > 0.0:
            mixup_indices = jax.random.permutation(mixup_rng, args.num_batch)
            mixup_weights = jax.random.beta(
                mixup_rng, args.mixup_alpha, args.mixup_alpha,
                shape=(args.num_batch, 1, 1, 1))
            batch['inputs'] = mixup_weights * batch['inputs'] \
                + (1. - mixup_weights)* batch['inputs'][mixup_indices]
        batch = jax.tree_util.tree_map(
            lambda e: e.reshape(shard_shape + e.shape[1:]), batch)

        print(batch['inputs'].shape) # [1, 256, 32, 32, 3]

        N = 8
        images = batch['inputs'][0].astype(np.uint8)
        fig, ax = plt.subplots(nrows=N, ncols=N, figsize=(5, 5))
        for i in range(N):
            for j in range(N):
                img = images[N*i + j]
                ax[i][j].imshow(img)
                ax[i][j].set_xticks([])
                ax[i][j].set_yticks([])
        plt.savefig(f'debug_rand_aug_mixup.png')
        break