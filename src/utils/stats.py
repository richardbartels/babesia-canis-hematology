"""Helper functions for statistical analysis."""

import numpy as np
from scipy import optimize, stats
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_curve
from tqdm import tqdm

from config import N_BOOTSTRAP


def get_threshold(y, y_proba, ref_sensitivity=0.95):
    """Compute threshold corresponding to ref_sensitivity.

    See: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html
    """
    fpr, tpr, thresholds = roc_curve(y, y_proba)
    ind = np.argmin(np.abs(tpr[:-1] - ref_sensitivity))  # ignore perfect sensitivity
    threshold = thresholds[ind]
    return threshold


def get_binary_prediction(y_proba, threshold):
    """Compute binary predictions."""
    y_pred = (y_proba >= threshold).astype("int")
    return y_pred


def get_metrics(y, y_proba, sample_weights=None, threshold=None):
    """Compute metrics."""
    if threshold is None:
        threshold = get_threshold(y, y_proba)
    y_pred = get_binary_prediction(y_proba, threshold)
    (tn, fp), (fn, tp) = confusion_matrix(y, y_pred)
    p = tp + fn
    n = tn + fp
    assert y.sum() == p
    assert (y == 0).sum() == n
    tpr = tp / float(p)
    tnr = tn / float(n)
    fpr = fp / float(n)
    ppv = tp / float(tp + fp)  # precision at best fit point
    sensitivity = tpr
    specificity = tnr
    accuracy = (tp + tn) / (p + n)

    if sample_weights is None:
        sample_weights = np.ones(y.shape[0])
    weighted_accuracy = (sample_weights * (y == y_pred).astype("float")).mean()

    fpr_roc, tpr_roc, _ = roc_curve(y, y_proba)
    precision, recall, _ = precision_recall_curve(y, y_proba)
    auroc = auc(fpr_roc, tpr_roc)
    likelihood_ratio = sensitivity / (1.0 - specificity)

    metrics = {
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "lr+": likelihood_ratio,
        "p": p,
        "n": n,
        "tp": tp,
        "tn": tn,
        "fpr": fpr,  # default threshold
        "tpr": tpr,
        "tnr": tnr,
        "ppv": ppv,
        "weighted_accuracy": weighted_accuracy,
        "tpr_roc": tpr_roc,
        "fpr_roc": fpr_roc,
        "auroc": auroc,
        "precision": precision,
        "recall": recall,
        "threshold": threshold,
    }
    return metrics


def bootstrap_resample(y_true, y_proba):
    """Resample y_true and y_proba."""
    n = len(y_true)
    indices_bootstrap = np.array(
        [
            np.random.choice(np.arange(n), size=n, replace=True)
            for _ in range(N_BOOTSTRAP)
        ]
    )
    y_proba_bootstrap = y_proba[indices_bootstrap]
    y_true_bootstrap = y_true[indices_bootstrap]
    return y_true_bootstrap, y_proba_bootstrap


def bootstrapped_metrics(y_true, y_proba, q=(0.025, 0.5, 0.975), threshold=None):
    """Recompute metrics using bootstrap technique.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Prediction probabilities
    q : tuple, optional
        Quantiles, by default (0.025, 0.5, 0.975)
    threshold : float, optional
        Threshold for predicting positive class, by default None

    Returns
    -------
    metric_bt : dict
        Dictionary with metrics
    """
    y_true_bt, y_proba_bt = bootstrap_resample(y_true, y_proba)
    sensitivities = []
    specificities = []
    likelihood_ratios = []
    accuracies = []
    ppvs = []
    aurocs = []
    tprs = []
    precisions = []
    fpr_ref = np.linspace(0, 1, 101)
    recall_ref = np.linspace(0, 1, 101)[::-1]
    for i in tqdm(range(N_BOOTSTRAP)):
        metrics = get_metrics(y_true_bt[i], y_proba_bt[i], threshold=threshold)
        sensitivities.append(metrics["sensitivity"])
        specificities.append(metrics["specificity"])
        likelihood_ratios.append(metrics["lr+"])
        accuracies.append(metrics["accuracy"])
        ppvs.append(metrics["ppv"])
        aurocs.append(metrics["auroc"])
        tprs.append(
            np.interp(fpr_ref, metrics["fpr_roc"], metrics["tpr_roc"])
        )  # for plotting purposes
        precisions.append(
            np.interp(recall_ref, metrics["recall"][::-1], metrics["precision"][::-1])
        )  # reverse order to have increasing recall

    metrics_bt = {
        "sensitivity": stats.mstats.mquantiles(sensitivities, q),
        "specificity": stats.mstats.mquantiles(specificities, q),
        "lr+": stats.mstats.mquantiles(likelihood_ratios, q),
        "accuracy": stats.mstats.mquantiles(accuracies, q),
        "ppv": stats.mstats.mquantiles(ppvs, q),
        "auroc": stats.mstats.mquantiles(aurocs, q),
        "tpr": stats.mstats.mquantiles(tprs, prob=q, axis=0),
        "fpr": fpr_ref,
        "precision": stats.mstats.mquantiles(precisions, prob=q, axis=0),
        "recall": recall_ref,
    }
    return metrics_bt


def get_ci(k, n, interval: float = 0.95):
    """Calculate credible interval for a beta distribution.

    Calculate the credible interval chosen to be between two points
    of similar likelihood.

    Parameters
    ----------
    k : int
        number of positive observations
    n : int
        Total number of observations
    interval : float
        Interval

    Returns
    -------
    lo : float
        Lower bound of the interval
    hi : float
        Upper bound of the interval
    """
    dist = stats.beta(k + 1, n - k + 1)

    # Find two points of equal likelihood
    residual = 1 - interval

    def f(x):
        return dist.pdf(dist.ppf(x)) - dist.pdf(dist.isf(residual - x))

    try:
        pp = optimize.brentq(f, 0, residual)  # percent point

        lo = dist.ppf(pp)
        hi = dist.isf(residual - pp)
    except ValueError:
        lo = -1
        hi = -1

    return lo, hi


def get_ci_str_beta(k, n, interval_range: float = 0.95, dec=3):
    """Return a string with the credible interval."""
    interval = get_ci(k, n, interval_range)
    return f"({'-'.join([str(round(s*100, dec-2)) for s in interval])}, {int(interval_range*100)}% C.I.)"


def get_ci_str_confidence(interval, confidence_interval: int = 95, dec=3):
    """Return a string with the confidence interval."""
    return f"({'-'.join([str(round(s*100, dec-2)) for s in interval])}, {confidence_interval:.0f}% C.I.)"
