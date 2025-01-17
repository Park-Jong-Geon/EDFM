import jax.numpy as jnp
from jax.experimental.host_callback import call as jcall
import os
import wandb
import jax
from flax.training import common_utils
from collections import OrderedDict
from flax.training import checkpoints
from easydict import EasyDict

debug = os.environ.get("DEBUG")
if isinstance(debug, str):
    debug = debug.lower() == "true"

PIXEL_MEAN = (0.49, 0.48, 0.44)
PIXEL_STD = (0.2, 0.2, 0.2)

def normalize(x):
    return (x-pixel_mean)/pixel_std
    # return (x-pixel_mean)


def unnormalize(x):
    return pixel_std*x+pixel_mean
    # return x+pixel_mean


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