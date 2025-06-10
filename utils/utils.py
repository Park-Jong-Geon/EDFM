import jax.numpy as jnp
import wandb
import jax
from easydict import EasyDict

PIXEL_MEAN = (0.49, 0.48, 0.44)
PIXEL_STD = (0.2, 0.2, 0.2)


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


def batch_mul(a, b):
    return jax.vmap(lambda a, b: a * b)(a, b)


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