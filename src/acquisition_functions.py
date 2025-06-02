import numpy as np
from scipy.stats import entropy as ent
from scipy.special import kl_div
from scipy.spatial.distance import jensenshannon
import torch
import torch.nn.functional as F


def score_entropy(stochastic_predictions: np.ndarray) -> np.ndarray:
    """
    Given a 4D numpy array of shape (T, C, H, W) representing T stochastic predictions
    (softmax probabilities) for one image, return a 1D array of length T where each entry
    is the sum of pixelwise entropy across all pixels.

    stochastic_predictions[t, c, i, j] = p(y=c | x, W_t)
    """
    T, C, H, W = stochastic_predictions.shape
    results = np.zeros((T, H, W), dtype=np.float64)

    # For each forward pass t, compute pixelwise entropy and sum the map
    for t in range(T):
        for i in range(H):
            for j in range(W):
                # Pixelwise distribution p(·|x_i_j, W_t)
                p_vec = stochastic_predictions[t, :, i, j]
                # Entropy of that pixel
                results[t, i, j] = ent(p_vec)

    # Flatten H×W → sum over all pixels
    results = results.reshape(T, H * W)
    entropy_scores = results.sum(axis=1)  # shape (T,)
    return entropy_scores


def score_kl_divergence(
        standard_prediction: np.ndarray,
        stochastic_predictions: np.ndarray
) -> np.ndarray:
    """
    Compute a KL‐divergence score for each stochastic pass:
      standard_prediction[t, c, i, j] is the “deterministic” softmax: p(y=c | x, W_eval)
      stochastic_predictions[t, c, i, j] is p(y=c | x, W_t) for t=0..T-1.
    Returns a 1D array of length T, where each entry is
      sum_{i,j} [ KL( standard_prediction[:,i,j] || stochastic_predictions[t,:,i,j] ) ].
    """
    T, C, H, W = stochastic_predictions.shape
    results = np.zeros((T, H, W), dtype=np.float64)

    for t in range(T):
        for i in range(H):
            for j in range(W):
                p_std = standard_prediction[0, :, i, j]  # shape (C,)
                p_sto = stochastic_predictions[t, :, i, j]  # shape (C,)
                # Compute discrete KL(p_std || p_sto)
                results[t, i, j] = kl_div(p_std, p_sto).sum()

    results = results.reshape(T, H * W)
    kl_scores = results.sum(axis=1)
    return kl_scores


def score_js_divergence(
        standard_prediction: np.ndarray,
        stochastic_predictions: np.ndarray
) -> np.ndarray:
    """
    Compute a Jensen‐Shannon score for each stochastic pass:
      JSD = 0.5 * KL(p_std || M) + 0.5 * KL(p_sto || M), where
      M = 0.5 * (p_std + p_sto).
    Returns 1D array length T = sum over all pixels of JSD at each pixel.
    """
    T, C, H, W = stochastic_predictions.shape
    results = np.zeros((T, H, W), dtype=np.float64)

    for t in range(T):
        for i in range(H):
            for j in range(W):
                p_std = standard_prediction[0, :, i, j].astype(np.float64)
                p_sto = stochastic_predictions[t, :, i, j].astype(np.float64)
                # Jensen‐Shannon distance from scipy is sqrt(JSD), but we want the divergence value:
                # jensenshannon returns sqrt(JS divergence). So we square it:
                # JS = ( jensenshannon(p_std, p_sto) )**2
                js_dist = jensenshannon(p_std, p_sto)
                js_divergence = js_dist * js_dist  # convert distance → divergence
                results[t, i, j] = js_divergence

    results = results.reshape(T, H * W)
    js_scores = results.sum(axis=1)
    return js_scores


def score_bald(model: torch.nn.Module,
               image: torch.Tensor,
               T: int = 10,
               device: str = "cuda",
               num_classes: int = 2
               ) -> float:
    """
    Compute BALD (Bayesian Active Learning by Disagreement) score for ONE image:
      1) Run deterministic forward pass: p0 = softmax( model.eval()(image) )
      2) Run T MC‐Dropout forwards: p_t for t=0..T-1
      3) H_mean = pixelwise entropy of Mean_t [p_t]
         E_H   = Mean_t [ pixelwise entropy of p_t ]
         BALD  = H_mean - E_H
      4) Return average over all pixels → one scalar.

    image: shape (1, C, H, W), single unlabeled image
    T: number of MC‐Dropout samples
    """
    # 1) Deterministic pass (dropout OFF)
    model.eval()
    with torch.no_grad():
        logits0 = model(image.to(device))  # (1, num_classes, H, W)
        p0 = F.softmax(logits0, dim=1).cpu().numpy()  # (1, C, H, W)

    # 2) MC‐Dropout passes (dropout ON)
    model.train()
    all_probs = []
    softmax = nn.Softmax(dim=1)
    for _ in range(T):
        with torch.no_grad():
            logits_t = model(image.to(device))  # (1, C, H, W)
            p_t = softmax(logits_t).cpu().numpy()  # (1, C, H, W)
            all_probs.append(p_t)  # list of length T

    all_probs = np.concatenate(all_probs, axis=0)  # shape (T, C, H, W)
    mean_p = np.mean(all_probs, axis=0)  # shape (C, H, W)

    # 3) Pixelwise entropy of mean prediction
    eps = 1e-8
    H_mean = -np.sum(mean_p * np.log(mean_p + eps), axis=0)  # (H, W)

    # 4) E[H(p_t)]
    ent_each = -np.sum(all_probs * np.log(all_probs + eps), axis=1)  # shape (T, H, W)
    E_H = np.mean(ent_each, axis=0)  # shape (H, W)

    # 5) Mutual information map
    mi_map = H_mean - E_H  # shape (H, W)
    return np.mean(mi_map)  # scalar
