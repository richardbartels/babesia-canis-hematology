"""Functions for evaluating model performance."""

import logging
import os

import pandas as pd
from sklearn.model_selection import cross_val_predict

from utils.stats import (
    get_binary_prediction,
    get_ci_str_beta,
    get_ci_str_confidence,
    get_metrics,
    get_threshold,
    bootstrapped_metrics,
)


def evaluate_classifier(
    clf,
    X,
    y,
    sample_weights=None,
    split="train",
    cv=None,
    bootstrap=False,
    log=False,
    threshold=None,
    identifiers=None,
    tag=None,
):
    """Evaluate classifier performance.

    Parameters
    ----------
    clf : object
        Sklearn classifier
    X : np.ndarray
        features
    y : np.ndarray
        labels
    sample_weights : list-like, optional
        sample weights, by default None
    split : str, optional
        train or test, by default "train"
    cv : object, optional
        Sklearn cross validation object, by default None
    bootstrap : bool, optional
        Compute bootstrapped metrics, by default False
    log : bool, optional
        Log steps, by default False
    threshold : float, optional
        Threshold for positive predictions (inclusive), default None
    identifiers : list-like, optional
        List of patient identifier corresponding to X and y. Save predictions if specified, default None
    tag : str, optional
        Tag to be used when saving results.

    Returns
    -------
    metrics : dict
        Dictionary with computed metrics
    metrics_bt : dict
        Dictionary with bootstrapped metrics if computed, else None
    """
    if split == "train":
        assert cv is not None
        y_proba = cross_val_predict(
            clf,
            X,
            y,
            cv=cv,
            fit_params={"clf__sample_weight": sample_weights},
            method="predict_proba",
        )
    elif split == "test":
        y_proba = clf.predict_proba(X)

    if len(y_proba.shape) == 2:
        y_proba = y_proba[:, 1]
    else:
        pass
    # Get metrics
    metrics = get_metrics(y, y_proba, sample_weights, threshold=threshold)
    sensitivity = metrics["sensitivity"]
    specificity = metrics["specificity"]
    accuracy = metrics["accuracy"]
    ppv = metrics["ppv"]
    weighted_accuracy = metrics["weighted_accuracy"]
    auroc = metrics["auroc"]
    llr = metrics["lr+"]

    # Get intervals, either through bootstrap or by assuming a beta interval
    if bootstrap:
        if threshold is None:
            threshold = get_threshold(y, y_proba)
        metrics_bt = bootstrapped_metrics(
            y, y_proba, q=(0.025, 0.5, 0.975), threshold=threshold
        )
        sensitivity_interval = get_ci_str_confidence(metrics_bt["sensitivity"][[0, -1]])
        specificity_interval = get_ci_str_confidence(metrics_bt["specificity"][[0, -1]])
        accuracy_interval = get_ci_str_confidence(metrics_bt["accuracy"][[0, -1]])
        auroc_interval = get_ci_str_confidence(metrics_bt["auroc"][[0, -1]])
        ppv_interval = get_ci_str_confidence(metrics_bt["ppv"][[0, -1]])
        llr_interval = get_ci_str_confidence(
            metrics_bt["lr+"][[0, -1]] / 100.0
        )  # divide by 100 because we multiply with 100 later
    else:
        p, n = metrics["p"], metrics["n"]
        tp, tn = metrics["tp"], metrics["tn"]
        sensitivity_interval = get_ci_str_beta(tp, metrics["p"])
        specificity_interval = get_ci_str_beta(tn, n)
        accuracy_interval = get_ci_str_beta(tp + tn, p + n)
        auroc_interval = None
        metrics_bt = None

    if log:
        logging.info(f"Evaluating on the {split} set")
        logging.info(f"AUROC: {auroc*100:.1f} {auroc_interval}")
        logging.info(
            f"Sensitivity (recall): {sensitivity*100:.1f} {sensitivity_interval}"
        )
        logging.info(f"Specifity: {specificity*100:.1f} {specificity_interval}")
        logging.info(f"LR+: {llr:.1f} {llr_interval}")
        logging.info(f"Precision (PPV/PV+): {ppv*100:.1f} {ppv_interval}")
        logging.info(f"Accuracy: {accuracy*100:.1f} {accuracy_interval}")
        logging.info(f"Weighted accuracy: {weighted_accuracy*100:.1f}")
        if identifiers is not None:
            df_preds = pd.DataFrame(
                data={
                    "Identifier": identifiers,
                    "y_proba": y_proba,
                    "y_pred": get_binary_prediction(y_proba, threshold),
                }
            )
            df_preds.to_csv(
                os.path.join(f"data/processed/predictions_{tag}_{split}.csv"),
                index=False,
            )

    return metrics, metrics_bt
