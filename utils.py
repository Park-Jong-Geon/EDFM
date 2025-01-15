import jax.numpy as jnp
import defaults_sghmc as defaults
from jax.experimental.host_callback import call as jcall
import os
import wandb
import jax
from flax.training import common_utils
from collections import OrderedDict
from flax.training import checkpoints
from easydict import EasyDict
import defaults_sgd

debug = os.environ.get("DEBUG")
if isinstance(debug, str):
    debug = debug.lower() == "true"

pixel_mean = jnp.array(defaults.PIXEL_MEAN, dtype=jnp.float32)
pixel_std = jnp.array(defaults.PIXEL_STD, dtype=jnp.float32)

def normalize(x):
    return (x-pixel_mean)/pixel_std
    # return (x-pixel_mean)


def unnormalize(x):
    return pixel_std*x+pixel_mean
    # return x+pixel_mean


def model_list(data_name, model_style, shared_head=False, tag=""):
    if data_name == "CIFAR100_x32" and model_style == "BN-ReLU":
        if shared_head:
            return [
                "./checkpoints/bn100_sd2_shared",
                "./checkpoints/bn100_sd3_shared",
                "./checkpoints/bn100_sd5_shared",
                "./checkpoints/bn100_sd7_shared",
                "./checkpoints/bn100_sd11_shared",
                "./checkpoints/bn100_sd13_shared",
                "./checkpoints/bn100_sd17_shared",
            ]
        else:
            return [
                "./checkpoints/bn100_sd2",
                "./checkpoints/bn100_sd3",
                # "./checkpoints/bn100_sd5",
                # "./checkpoints/bn100_sd7",
                # "./checkpoints/bn100_sd11",
                # "./checkpoints/bn100_sd13",
                # "./checkpoints/bn100_sd17",
            ]
    if data_name == "CIFAR100_x32" and model_style == "FRN-Swish":
        if tag == "AtoABC":
            return [
                "./checkpoints/frn100_sd11_be",
                "./checkpoints/frn100_sd13_be",
                "./checkpoints/frn100_sd15_be",
                "./checkpoints/frn100_sd2_be",
                "./checkpoints/frn100_sd3_be",
                "./checkpoints/frn100_sd5_be",
                "./checkpoints/frn100_sd9_be",
            ]
        elif tag == "AtoABC2":
            return [
                "./checkpoints/frn100_sd11_be",
                "./checkpoints/frn100_sd2_be",
                "./checkpoints/frn100_sd3_be",
            ]
        elif tag == "AtoABC3":
            return [
                "./checkpoints/frn100_sd11_be",
                "./checkpoints/frn100_sd5_be",
                "./checkpoints/frn100_sd9_be",
            ]
    elif data_name == "TinyImageNet200_x64" and model_style == "FRN-Swish":
        if tag == "AtoABC":
            return [
                "./checkpoints/frn200_sd3_be",
                "./checkpoints/frn200_sd5_be",
                "./checkpoints/frn200_sd7_be",
                "./checkpoints/frn200_sd9_be",
                "./checkpoints/frn200_sd13_be",
                "./checkpoints/frn200_sd15_be",
                "./checkpoints/frn200_sd17_be",
            ]
        elif tag == "AtoABC2":
            return [
                "./checkpoints/frn200_sd3_be",
                "./checkpoints/frn200_sd9_be",
                "./checkpoints/frn200_sd13_be",
            ]
        elif tag == "AtoABC3":
            return [
                "./checkpoints/frn200_sd3_be",
                "./checkpoints/frn200_sd15_be",
                "./checkpoints/frn200_sd17_be",
            ]
    elif data_name == "ImageNet1k_x64" and model_style == "FRN-Swish":
        if tag == "AtoABC":
            return [
                "./checkpoints/frn1000_sd2_be",
                "./checkpoints/frn1000_sd3_be",
                "./checkpoints/frn1000_sd5_be",
                "./checkpoints/frn1000_sd7_be",
                "./checkpoints/frn1000_sd9_be",
                "./checkpoints/frn1000_sd11_be",
                "./checkpoints/frn1000_sd13_be",
                "./checkpoints/frn1000_sd15_be",
                "./checkpoints/frn1000_sd17_be",
                "./checkpoints/frn1000_sd19_be",
            ]
        elif tag == "AtoABC2":
            return [
                "./checkpoints/frn1000_sd2_be",
                "./checkpoints/frn1000_sd7_be",
                "./checkpoints/frn1000_sd9_be",
            ]
        elif tag == "AtoABC3":
            return [
                "./checkpoints/frn1000_sd2_be",
                "./checkpoints/frn1000_sd11_be",
                "./checkpoints/frn1000_sd13_be",
            ]
        elif tag == "vAtoABC":
            return [
                "./checkpoints/frnv1000_sd2_tpu",
                "./checkpoints/frnv1000_sd3_tpu",
                "./checkpoints/frnv1000_sd5_tpu",
            ]
        elif tag == "vAtoABC2":
            return [
                "./checkpoints/frnv1000_sd2_tpu",
                "./checkpoints/frnv1000_sd7_tpu",
                "./checkpoints/frnv1000_sd9_tpu",
            ]
        elif tag == "vAtoABC3":
            return [
                "./checkpoints/frnv1000_sd2_tpu",
                "./checkpoints/frnv1000_sd11_tpu",
                "./checkpoints/frnv1000_sd13_tpu",
            ]
        elif tag == "vAtoCAB":
            return [
                "./checkpoints/frnv1000_sd5_tpu",
                "./checkpoints/frnv1000_sd2_tpu",
                "./checkpoints/frnv1000_sd3_tpu",
            ]
        elif tag == "vAtoAB":
            return [
                "./checkpoints/frnv1000_sd2_tpu",
                "./checkpoints/frnv1000_sd3_tpu",
            ]
        elif tag == "vAtoAB2":
            return [
                "./checkpoints/frnv1000_sd2_tpu",
                "./checkpoints/frnv1000_sd5_tpu",
            ]
    elif data_name == "CIFAR10_x32" and model_style == "FRN-Swish":
        if tag == "bezier":
            return [
                "./checkpoints/frn_sd2_be",
                "./checkpoints/frn_sd25_bezier"
            ]
        elif tag == "distill":
            return [
                "./checkpoints/frn_sd23_distill_mean2",
                "./checkpoints/frn_sd2_be",
                "./checkpoints/frn_sd3_be"
            ]
        elif tag == "distref":
            return [
                "./checkpoints/frn_sd2_be",
                "./checkpoints/frn_sd2_be",
                "./checkpoints/frn_sd3_be"
            ]
        elif tag == "AtoB":
            return [
                "./checkpoints/frn_sd2_be",
                "./checkpoints/frn_sd3_be"
            ]
        elif tag == "AtoshB":
            return [
                "./checkpoints/frn_sd2_be",
                "./checkpoints/frn_sd23_shared3"
            ]
        elif tag == "AtoshABC":
            return [
                "./checkpoints/frn_sd2_be",
                "./checkpoints/frn_sd23_shared3",
                "./checkpoints/frn_sd25_shared3",
                "./checkpoints/frn_sd27_shared3",
                "./checkpoints/frn_sd29_shared3"
            ]
        elif tag == "AtoABC":
            return [
                "./checkpoints/frn_sd2_be",
                "./checkpoints/frn_sd3_be",
                "./checkpoints/frn_sd5_be",
                "./checkpoints/frn_sd7_be",
                "./checkpoints/frn_sd9_be",
                "./checkpoints/frn_sd11_be",
                "./checkpoints/frn_sd13_be",
                "./checkpoints/frn_sd15_be",
                "./checkpoints/frn_sd17_be",
            ]
        elif tag == "AtoABC2":
            return [
                "./checkpoints/frn_sd2_be",
                "./checkpoints/frn_sd7_be",
                "./checkpoints/frn_sd9_be",
            ]
        elif tag == "AtoABC3":
            return [
                "./checkpoints/frn_sd2_be",
                "./checkpoints/frn_sd11_be",
                "./checkpoints/frn_sd13_be",
            ]
        elif tag == "AtoABC4":
            return [
                "./checkpoints/frn_sd2_be",
                "./checkpoints/frn_sd15_be",
                "./checkpoints/frn_sd17_be",
            ]
        elif tag == "DistoABC":
            return [
                "./checkpoints/naive_ed/frn_sd235_t3",
                "./checkpoints/frn_sd2_be",
                "./checkpoints/frn_sd3_be",
                "./checkpoints/frn_sd5_be",
            ]
        elif tag == "layer2stride1_shared":
            return [
                "./checkpoints/frn_sd2_be",
                "./checkpoints/frn_sd23_layer2stride1",
                "./checkpoints/frn_sd24_layer2stride1",
                "./checkpoints/frn_sd25_layer2stride1",
                "./checkpoints/frn_sd26_layer2stride1",
                "./checkpoints/frn_sd27_layer2stride1"
            ]
        elif tag == "ABCtoindABC":
            return [
                "./checkpoints/frn_sd2_be",
                "./checkpoints/frn_sd23_shared3",
                "./checkpoints/frn_sd3_be",
                "./checkpoints/frn_sd35_shared3",
                "./checkpoints/frn_sd5_be",
                "./checkpoints/frn_sd57_shared3"
            ]
        elif tag == "ABCtoindABC_layer2stride1":
            return [
                "./checkpoints/frn_sd2_be",
                "./checkpoints/frn_sd23_layer2stride1",
                "./checkpoints/frn_sd3_be",
                "./checkpoints/frn_sd35_layer2stride1",
                "./checkpoints/frn_sd5_be",
                "./checkpoints/frn_sd57_layer2stride1"
            ]
        elif tag == "layer2stride1_nonshared":
            return [
                "./checkpoints/frn_sd2_be",
                "./checkpoints/frn_sd3_be",
                "./checkpoints/frn_sd5_be"
            ]
        elif tag == "distA":
            return [
                "./checkpoints/frn_sd23_distill_mean2",
                "./checkpoints/frn_sd2_be"
            ]
        elif tag == "distB":
            return [
                "./checkpoints/frn_sd23_distill_mean2",
                "./checkpoints/frn_sd3_be"
            ]
        return [
            "./checkpoints/frn_sd2",
            "./checkpoints/frn_sd3",
            "./checkpoints/frn_sd5",
            "./checkpoints/frn_sd7",
            "./checkpoints/frn_sd11",
            "./checkpoints/frn_sd13",
            "./checkpoints/frn_sd17",
            # "./checkpoints/frn_sd19",
        ]
    elif data_name == "CIFAR10_x32" and model_style == "BN-ReLU":
        return [
            # "./checkpoints/bn_sd2_smooth",
            "./checkpoints/bn_sd3_smooth",
            "./checkpoints/bn_sd5_smooth",
            "./checkpoints/bn_sd7_smooth",
        ]
    else:
        raise Exception("Invalid data_name and model_style.")


logit_dir_list = [
    "features", "features_fixed",
    "features_1mixup10", "features_1mixup10_fixed",
    "features_1mixupplus10",
    "features_smooth",
    "features_distill",
    "features_distill_mean",
    "features_distill_mean20",
    "features_distill_mean5e-4",
    "features_distill_mean2",
    "features_distref",
    "features100",
    "features100_ods", "features100_noise",
    "features100_2ods",
    "features100_fixed", "features100_shared",
    "features100_1mixup10", "features100_1mixupplus10",
    "features100_1mixupext10", "features100_1mixupplusext10",
    "features100_0p4mixup10_fixed", "features100_0p4mixup10_rand",
    "features100_0p4mixup10", "features100_0p4mixup10_valid",
    "features_distA", "features_distB",
    "features_AtoB", "features_AtoshB",
    "features_AtoshABC",
    "features_AtoABC"
]
feature_dir_list = [
    "features_last", "features_last_fixed",
    "features_last_1mixup10", "features_last_1mixup10_fixed",
    "features_last_1mixupplus10",
    "features_last_smooth",
    "features_last_distill",
    "features_last_distill_mean5e-4",
    "features_last_distill_mean2",
    "features_last_distref",
    "features100_last",
    "features100_last_1mixup10", "features100_last_0p4mixup10",
    "features100_last_0p4mixup10_fixed", "features100_last_0p4mixup10_rand",
    "features100_last_fixed", "features100_last_shared",
    "features_last_distA", "features_last_distB",
    "features_last_bezier25"
]
feature2_dir_list = [
    "features_last2",
    "features100_last2",
    "features100_last2_shared"
]
feature3_dir_list = [
    "features_last3_bezier",
    "features_last3_bezier_0p4mixup5",
    "features_last3_bezier_0p4mixup10",
    "features_last3_bezier25",
    "features_last3_bezier27",
    "features_last3_distref",
    "features100_last3",
    "features_last3_distA",
    "features_last3_distill_mean2",
    "features_last3_AtoB", "features_last3_AtoshB",
    "features_last3_AtoshABC",
    "features_last3_AtoABC"
]
layer2stride1_dir_list = [
    "features_layer2stride1_shared"
]


def _get_meanstd(features_dir):
    if features_dir == "features":
        mean = logits_mean
        std = logits_std
    elif features_dir in [
            "features_fixed",
            "features_1mixup10",
            "features_1mixup10_fixed",
            "features_1mixupplus10",
            "features_distill",
            "features_distill_mean",
            "features_distill_mean20",
            "features_distill_mean5e-4",
            "features_distill_mean2",
            "features_distref",
            "features_distA",
            "features_distB",
            "features_AtoB",
            "features_AtoshB"
    ]:
        mean = logits_fixed_mean
        std = logits_fixed_std
    elif features_dir in ["features_smooth"]:
        mean = logits_smooth_mean
        std = logits_smooth_std
    elif features_dir == "features_last":
        mean = features_mean
        std = features_std
    elif features_dir in ["features_last_smooth"]:
        mean = features_smooth_mean
        std = features_smooth_std
    elif features_dir in ["features_last2"]:
        mean = features_mean[None, None, ...]
        std = features_std[None, None, ...]
    elif features_dir in [
        "features_last_fixed",
        "features_last_1mixup10",
        "features_last_1mixup10_fixed",
        "features_last_1mixupplus10",
    ]:
        mean = features_fixed_mean
        std = features_fixed_std
    elif features_dir in [
        "features100",
        "features100_ods",
        "features100_2ods",
        "features100_noise",
        "features100_fixed",
        "features100_shared",
        "features100_0p4mixup10",
        "features100_1mixup10",
        "features100_0p4mixup10_valid",
        "features100_0p4mixup10_fixed",
        "features100_0p4mixup10_rand",
        "features100_1mixupplus10",
        "features100_1mixupext10",
        "features100_1mixupplusext10"
    ]:
        mean = logits100_mean
        std = logits100_std
    elif features_dir in [
        "features100_last",
        "features100_last_fixed",
        "features100_last_shared",
        "features100_last_0p4mixup10",
        "features100_last_0p4mixup10_fixed",
        "features100_last_0p4mixup10_rand",
        "features100_last_1mixup10"
    ]:
        mean = features100_mean
        std = features100_std
    elif features_dir in ["features100_last2", "features100_last2_shared"]:
        mean = features100_mean[None, None, ...]
        std = features100_std[None, None, ...]
    elif features_dir in [
        "features_last3_bezier",
        "features_last3_bezier_0p4mixup10",
        "features_last3_bezier_0p4mixup5",
        "features_last3_bezier25",
        "features_last3_bezier27",
        "features_last3_distref",
        "features100_last3",
        "features_last3_distA",
        "features_last3_distill_mean2",
        "features_last3_AtoB",
        "features_last3_AtoshB",
        "features_last3_AtoshABC",
        "features_AtoshABC",
        "features_AtoABC",
        "features_last3_AtoABC",
        "features_layer2stride1_shared",
    ]:
        mean = 0
        std = 1
    elif features_dir in [
        "features_last_distill",
        "features_last_distill_mean5e-4",
        "features_last_distill_mean2",
        "features_last_distref",
        "features_last_distA",
        "features_last_distB",
        "features_last_bezier25"
    ]:
        mean = features_distill_mean
        std = features_distill_std
    else:
        raise Exception("Calculate corresponding statistics")
    return mean, std


def normalize_logits(x, features_dir="features_last"):
    mean, std = _get_meanstd(features_dir)
    return (x-mean)/std


def unnormalize_logits(x, features_dir="features_last"):
    mean, std = _get_meanstd(features_dir)
    return mean+std*x


def pixelize(x):
    return (x*255).astype("uint8")


def jprint_fn(*args):
    fstring = ""
    arrays = []
    for i, a in enumerate(args):
        if i != 0:
            fstring += " "
        if isinstance(a, str):
            fstring += a
        else:
            fstring += '{}'
            arrays.append(a)

    jcall(lambda arrays: print(fstring.format(*arrays)), arrays)


jprint = lambda *args: ...
flprint = lambda *args,**kwargs: ...
if debug:
    print("**** DEBUG MODE ****")
    jprint = jprint_fn
    flprint = print
else:
    print("**** DEBUG MODE OFF ****")


def get_info_in_dir(dir):
    if "mixupplus" in dir:
        sep = "mixupplus"
    else:
        sep = "mixup"
    if "ext" in dir:
        sep += "ext"
    alpha = float(
        dir.split(sep)[0].split("_")[-1].replace("p", ".")) if sep in dir else -1
    repeats = int(
        dir.split(sep)[1].split("_")[0]) if sep in dir else 1

    return alpha, repeats


class WandbLogger():
    def __init__(self):
        self.summary = dict()
        self.logs = dict()

        self.to_summary = [
            # "trn/acc_ref", "trn/nll_ref", "trn/ens_acc_ref", "trn/ens_t2acc_ref", "trn/ens_nll_ref", "trn/kld_ref", "trn/rkld_ref",
            # "val/acc_ref", "val/nll_ref", "val/ens_acc_ref", "val/ens_t2acc_ref", "val/ens_nll_ref", "val/kld_ref", "val/rkld_ref",
            # "tst/acc_ref", "tst/nll_ref", "tst/ens_acc_ref", "tst/ens_t2acc_ref", "tst/ens_nll_ref", "tst/kld_ref", "tst/rkld_ref",
            "tst/loss",
            "tst/skld", "tst/rkld"
            "tst/sec", "val/sec", "trn/sec"
        ]
        self.summary_keywords = ["ref", "from"]

    def log(self, object):
        for k in self.to_summary:
            value = object.get(k)
            if value is None:
                continue
            self.summary[k] = value
            del object[k]
        for k, v in object.items():
            to_summary = False
            for kw in self.summary_keywords:
                if kw in k:
                    to_summary = True
                    self.summary[k] = v
                    break
            if not to_summary:
                self.logs[k] = v

    def flush(self):
        for k, v in self.summary.items():
            wandb.run.summary[k] = v
        wandb.log(self.logs)
        self.summary = dict()
        self.logs = dict()


def expand_to_broadcast(input, target, axis):
    len_in = len(input.shape)
    len_tar = len(target.shape)
    assert len_tar >= len_in
    expand = len_tar - len_in
    expand = list(range(axis, axis+expand))
    return jnp.expand_dims(input, axis=expand)


class FeatureBank():
    def __init__(self, num_classes, maxlen=128, disable=False, gamma=1.):
        self.bank = [jnp.array([]) for _ in range(num_classes)]
        self.len = [0 for _ in range(num_classes)]
        self.num_classes = num_classes
        self.maxlen = maxlen
        self.cached = None
        self.disable = disable
        self.gamma = gamma

    def _squeeze(self, batch):
        assert len(batch["images"].shape) == 5 or len(
            batch["images"].shape) == 3
        batch_unpack = dict()
        shapes = dict()
        for k, v in batch.items():
            shapes[k] = v.shape
            batch_unpack[k] = v.reshape(-1, *v.shape[2:])

        return batch_unpack, shapes

    def _unsqueeze(self, batch, shapes):
        assert len(batch["images"].shape) == 4 or len(
            batch["images"].shape) == 2

        new_batch = dict()
        for k, v in batch:
            new_batch[k] = v.reshape(*shapes[k])

        return new_batch

    def deposit(self, batch):
        if self.disable:
            return None
        batch, _ = self._squeeze(batch)
        self._deposit(batch)

    def _deposit(self, batch):
        f_b = batch["images"]
        f_a = batch["labels"]
        labels = batch["cls_labels"]
        marker = batch["marker"]
        f = jnp.concatenate([f_b, f_a], axis=-1)
        shape = f.shape
        f = f[marker, ...]
        if self.cached is None:  # denotes right after bank init
            self.bank = list(
                map(lambda x: x.reshape(-1, *shape[1:]), self.bank))

        def store_fn(i):
            in_class = f[labels == i, ...]
            length = len(in_class)
            exceed = len(self.bank[i]) + length - self.maxlen
            if exceed > 0:
                self.bank[i] = self.bank[i][exceed:]
            self.bank[i] = jnp.concatenate([self.bank[i], in_class], axis=0)
            self.len[i] = len(self.bank[i])
            return val
        val = map(store_fn, range(self.num_classes))
        min_len = min(self.len)

        def trunc_fn(x):
            return x[-min_len:]
        cached = list(map(trunc_fn, self.bank))
        self.cached = jnp.stack(cached)

    def withdraw(self, rng, _batch):
        if self.disable:
            return _batch
        batch, shapes = self._squeeze(_batch)
        labels = batch["cls_labels"]
        out = self._withdraw(rng, labels)
        if out is None:
            return _batch
        assert out.shape == batch["images"].shape
        f_b, f_a = jnp.split(out, 2, axis=-1)
        marker = batch["marker"]
        batch["images"] = batch["images"].at[marker, ...].set(f_b)
        batch["labels"] = batch["labels"].at[marker, ...].set(f_a)
        batch = self._unsqueeze(batch, shapes)
        return batch

    def _withdraw(self, rng, labels):
        min_len = min(self.len)
        if min_len == 0:
            return None
        indices = jax.random.randint(rng, (len(labels),), 0, min_len)
        new = self.cached[labels, indices]
        return new

    def mixup_inclass(self, rng, batch, alpha=1.0):
        if self.disable:
            return batch
        # rng = jax_utils.unreplicate(rng)

        f_b = batch["images"]
        f_a = batch["labels"]
        self.deposit(batch)

        beta_rng, perm_rng = jax.random.split(rng)
        lam = jnp.where(alpha > 0, jax.random.beta(beta_rng, alpha, alpha), 1)
        lam *= self.gamma
        ingredient_batch = self.withdraw(perm_rng, batch)
        ing_b = ingredient_batch["images"]
        ing_a = ingredient_batch["labels"]

        mixed_b = lam*f_b + (1-lam)*ing_b
        mixed_a = lam*f_a + (1-lam)*ing_a
        batch["images"] = mixed_b
        batch["labels"] = mixed_a
        return batch

    def perm_aug(self, rng, batch):
        n_cls = batch["images"].shape[-1]
        labels = batch["cls_labels"]
        ps, bs = labels.shape
        assert len(labels.shape) == 2
        order = jnp.tile(jnp.arange(0, n_cls-1)[None, None, :], [ps, bs, 1])
        order = jnp.where(order >= labels[..., None], order+1, order)
        perm_order = jax.random.permutation(rng, order, axis=-1)
        perm = perm_order.reshape(-1, *perm_order.shape[2:])
        l = labels.reshape(-1, *labels.shape[2:])
        perm_order = jax.vmap(jnp.insert, (0, 0, 0), 0)(perm, l, l)

        def mixer(value):
            assert len(value.shape) > 2
            shape = value.shape
            value = value.reshape(-1, *shape[2:])
            mix = jax.vmap(lambda v, p: v[..., p], (0, 0), 0)
            result = mix(value, perm_order)
            return result.reshape(*shape)

        for k, v in batch.items():
            if v.shape[-1] == n_cls:
                batch[k] = mixer(v)
            if k == "cls_labels":
                labels = common_utils.onehot(v, n_cls)
                batch[k] = jnp.argmax(mixer(labels), axis=-1)

        return batch


def evaluate_top2acc(confidences, true_labels, log_input=True, eps=1e-8, reduction="mean"):

    pred_labels = jnp.argmax(confidences, axis=1)
    mask = common_utils.onehot(pred_labels, confidences.shape[-1])
    temp = -mask*1e10+(1-mask)*confidences
    pred2_labels = jnp.argmax(temp, axis=1)
    raw_results = jnp.equal(pred_labels, true_labels)
    raw2_results = jnp.equal(pred2_labels, true_labels)
    raw_results = jnp.logical_or(raw_results, raw2_results)
    if reduction == "none":
        return raw_results
    elif reduction == "mean":
        return jnp.mean(raw_results)
    elif reduction == "sum":
        return jnp.sum(raw_results)
    else:
        raise NotImplementedError(f'Unknown reduction=\"{reduction}\"')


def evaluate_topNacc(confidences, true_labels, top=5, log_input=True, eps=1e-8, reduction="mean"):

    pred_labels = jnp.argmax(confidences, axis=1)
    mask = common_utils.onehot(pred_labels, confidences.shape[-1])
    temp = -mask*1e10+(1-mask)*confidences
    raw_results = jnp.equal(pred_labels, true_labels)
    for i in range(1, top):
        pred2_labels = jnp.argmax(temp, axis=1)
        mask2 = common_utils.onehot(pred2_labels, confidences.shape[-1])
        temp = -mask2*1e10+(1-mask2)*temp
        raw2_results = jnp.equal(pred2_labels, true_labels)
        raw_results = jnp.logical_or(raw_results, raw2_results)
    if reduction == "none":
        return raw_results
    elif reduction == "mean":
        return jnp.mean(raw_results)
    elif reduction == "sum":
        return jnp.sum(raw_results)
    else:
        raise NotImplementedError(f'Unknown reduction=\"{reduction}\"')


def batch_add(a, b):
    return jax.vmap(lambda a, b: a + b)(a, b)


def batch_mul(a, b):
    return jax.vmap(lambda a, b: a * b)(a, b)


def get_probs(logits, ignore=False):
    if ignore:
        return logits
    if len(logits) == 0:
        return logits
    probs = jnp.exp(jax.nn.log_softmax(logits, axis=-1))
    return probs


def get_logprobs(logits, ignore=False):
    if ignore:
        return logits
    if len(logits) == 0:
        return logits
    logprobs = jax.nn.log_softmax(logits, axis=-1)
    return logprobs


def get_ens_logits(logits, logitmean=None, mean_axis=0):
    # TODO: better way
    if len(logits) == 1:
        return logits[0]
    if logitmean is None:
        logitmean = jnp.mean(logits[mean_axis], axis=-1)[:, None]
    ens_prob = 0
    for l in logits:
        ens_prob += jax.nn.softmax(l, axis=-1)
    ens_prob /= len(logits)
    ens_logprob = jnp.log(ens_prob)
    ens_logprob_mean = jnp.mean(ens_logprob, axis=-1)[:, None]
    ens_logits = ens_logprob-ens_logprob_mean+logitmean
    return ens_logits


def get_avg_logits(logits):
    return sum(logits)/len(logits)


def get_single_batch(batch, i=0):
    new_batch = OrderedDict()
    for k, v in batch.items():
        new_batch[k] = v[i]
    return new_batch


def get_config(ckpt_config):
    for k, v in ckpt_config.items():
        if isinstance(v, dict) and v.get("0") is not None:
            l = []
            for k1, v1 in v.items():
                l.append(v1)
            ckpt_config[k] = tuple(l)
    config = EasyDict(ckpt_config)
    model_dtype = getattr(config, "dtype", None) or "float32"
    if "float32" in model_dtype:
        config.dtype = jnp.float32
    elif "float16" in model_dtype:
        config.dtype = jnp.float16
    else:
        raise NotImplementedError
    if getattr(config, "num_classes", None) is None:
        if config.data_name == "CIFAR10_x32":
            config.num_classes = 10
        elif config.data_name == "CIFAR100_x32":
            config.num_classes = 100
        elif config.data_name == "TinyImageNet200_x64":
            config.num_classes = 200
        elif config.data_name == "ImageNet1k_x64":
            config.num_classes = 1000
    if getattr(config, "image_stats", None) is None:
        config.image_stats = dict(
            m=jnp.array(defaults_sgd.PIXEL_MEAN),
            s=jnp.array(defaults_sgd.PIXEL_STD))
    if getattr(config, "model_planes", None) is None:
        if config.data_name == "CIFAR10_x32" and config.model_style == "FRN-Swish":
            config.model_planes = 16
            config.model_blocks = None
        elif config.data_name == "CIFAR100_x32" and config.model_style == "FRN-Swish":
            config.model_planes = 16
            config.model_blocks = None
        elif config.data_name == "TinyImageNet200_x64" and config.model_style == "FRN-Swish":
            config.model_planes = 64
            config.model_blocks = "3,4,6,3"
        elif config.data_name == "ImageNet1k_x64" and config.model_style == "FRN-Swish":
            config.model_planes = 64
            config.model_blocks = "3,4,6,3"
    if getattr(config, "first_conv", None) is None:
        if config.data_name == "CIFAR10_x32" and config.model_style == "FRN-Swish":
            config.first_conv = None
            config.first_pool = None
            config.model_nobias = False
        elif config.data_name == "CIFAR100_x32" and config.model_style == "FRN-Swish":
            config.first_conv = None
            config.first_pool = None
            config.model_nobias = False
        elif config.data_name == "TinyImageNet200_x64" and config.model_style == "FRN-Swish":
            config.first_conv = None
            config.first_pool = None
            config.model_nobias = False
        elif config.data_name == "ImageNet1k_x64" and config.model_style == "FRN-Swish":
            config.first_conv = None
            config.first_pool = None
            config.model_nobias = True
    if getattr(config, "dsb_continuous", None) is None:
        config.dsb_continuous = False
    if getattr(config, "centering", None) is None:
        config.centering = False
    return config


def load_ckpt(dirname):
    ckpt = checkpoints.restore_checkpoint(
        ckpt_dir=dirname,
        target=None
    )
    if ckpt is None:
        raise Exception(f"{dirname} does not exist")
    if ckpt.get("model") is not None:
        if ckpt.get("Dense_0") is not None:
            params = ckpt["model"]
            batch_stats = dict()
        else:
            params = ckpt["model"]["params"]
            batch_stats = ckpt["model"]["batch_stats"]
    else:
        params = ckpt["params"]
        batch_stats = ckpt["batch_stats"]
    config = get_config(ckpt["config"])
    return params, batch_stats, config