import numpy as np
from scipy.special import softmax

def compute_acc(probs, labels):
    return float(np.mean(np.equal(np.argmax(probs, axis=-1), labels)))

def compute_nll(probs, labels):
    return float(np.mean(np.negative(
        np.log(probs + 1e-05)[np.arange(labels.shape[0]), labels])))

def compute_ece(probs, labels, n_bins=15):
    bins = [[] for _ in range(n_bins)]
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    for prob, label in zip(probs, labels):
        max_prob = np.max(prob)
        for i in range(n_bins):
            if bin_boundaries[i] < max_prob <= bin_boundaries[i+1]:
                break
        bins[i].append([np.equal(np.argmax(prob), label), max_prob])
    ece = 0.0
    for i in range(n_bins):
        if len(bins[i]) == 0:
            continue
        b = np.array(bins[i]).mean(0)
        ece += np.abs(b[1] - b[0]) * len(bins[i]) / len(probs)
    return ece

def compute_amb(ens_logits, labels):
    # centroid = np.prod(np.float_power(softmax(ens_logits, axis=-1), 1.0 / ens_logits.shape[1]), axis=1)
    # centroid = centroid / centroid.sum(1, keepdims=True)
    # centroid = np.expand_dims(centroid, 1)
    # amb = centroid * (np.log(centroid) - np.log(softmax(ens_logits, axis=-1))).sum(2, keepdims=True)
    # amb = amb.mean()
    ens_nll = compute_nll(softmax(ens_logits.mean(1), axis=-1), labels)
    avg_nll = np.mean([
        compute_nll(softmax(ens_logits[:, i, :], axis=-1), labels)
        for i in range(ens_logits.shape[1])])
    # print(amb, avg_nll - ens_nll)
    return avg_nll, avg_nll - ens_nll, ens_nll

def compute_var(ens_logits, labels):
    probs = softmax(ens_logits, axis=-1)
    ens_unc = 1.0 - (probs.mean(1) ** 2).sum(1).mean()
    avg_unc = np.mean([
        1.0 - (probs[:, i, :]**2).sum(1).mean()
        for i in range(ens_logits.shape[1])])
    # var = softmax(ens_logits, axis=-1).var(1).sum(1).mean()
    return avg_unc, ens_unc - avg_unc, ens_unc