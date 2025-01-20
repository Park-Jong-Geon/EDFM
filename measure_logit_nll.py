import numpy as np
import jax
import jax.numpy as jnp
from flax.training import common_utils

@jax.jit
def ce_loss(logits, labels):
    target = common_utils.onehot(labels, num_classes=logits.shape[-1])
    pred = jax.nn.log_softmax(logits, axis=-1)
    loss = jnp.mean(-jnp.sum(target*pred, axis=-1))
    return loss

label = np.load('data/CIFAR100_x32/test_labels.npy')
print(f'label shape: {label.shape}')

postfix = ['endd_2025', 'endd_2027', 'endd_2028', 'kd_2025', 'kd_2026', 'kd_2027']
for p in postfix:
    logit = np.load(f'logits/CIFAR100_x32_{p}.npy')
    print(f'logit shape: {logit.shape}')

    print(f'NLL at {p}: {ce_loss(logit, label)}')
