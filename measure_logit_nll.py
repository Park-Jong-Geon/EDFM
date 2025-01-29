import numpy as np
from scipy.special import softmax, log_softmax

def ce_loss(logits, labels):
    target = np.eye(logits.shape[-1])[labels]
    pred = log_softmax(logits, axis=-1)
    loss = np.mean(-np.sum(target*pred, axis=-1))
    return loss

label = np.load('data/CIFAR100_x32/test_labels.npy')
print(f'label shape: {label.shape}')

postfix = ['fm_cinic40960']
# postfix = ['endd_2025', 'endd_2027', 'endd_2028', 'kd_2025', 'kd_2026', 'kd_2027']
for p in postfix:
    logit = np.load(f'logits/CIFAR100_x32_{p}.npy')
    print(f'logit shape: {logit.shape}')
    
    logit = softmax(logit, axis=-1).mean(1)
    logit = np.log(logit)
    print(f'logit shape: {logit.shape}')

    print(f'NLL at {p}: {ce_loss(logit, label)}')
