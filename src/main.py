"""Main script."""

import argparse
from datetime import datetime
import logging
import time

import numpy as np
from sklearn.model_selection import StratifiedKFold

from config import N_SPLITS
from data.data import Data
from model.train import get_sample_weights, models, train_classifier
from model.evaluate import evaluate_classifier
from utils.plots import plot_roc, plot_dtree, plot_cmatrix, plot_shap, plot_prcurve

tag = datetime.now().strftime("%Y:%m:%d-%H:%M:%S")


def run(mode="train"):
    """Run full analysis for either training or testing."""
    logging.basicConfig(filename=f"logs/{mode}_{tag}.log", level=logging.DEBUG)
    t1 = time.time()
    d = Data()
    t2 = time.time()
    logging.info(f"Took: {t2 - t1:.1f}s")

    # Not grouped, since duplicate patients are not present a lot
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    sample_weights = get_sample_weights(d.y_train)
    logging.info(f"Sample_weights: {np.unique(sample_weights)}")

    for model_name in list(models.keys()):
        if mode == "train":
            pipe, _, _ = train_classifier(
                model_name,
                d.X_train,
                d.y_train,
                sample_weights=sample_weights,
                cv=skf,
                data_checksum=d.md5_checksum_train,
            )
        elif mode == "test":
            pipe, metrics_train, metrics_bt_train = train_classifier(
                model_name,
                d.X_train.values,
                d.y_train.values,
                sample_weights=sample_weights,
                cv=skf,
                use_best_fit_parameters=True,
                bootstrap=True,
                feature_columns=d.X_train.columns,
            )
            metrics_test, metrics_bt_test = evaluate_classifier(
                pipe,
                d.X_test.values,
                d.y_test.values,
                sample_weights=np.ones(d.y_test.shape[0]),
                split="test",
                cv=None,
                bootstrap=True,
                log=True,
                threshold=metrics_train["threshold"],
                identifiers=d.id_test.values,
                tag=model_name,
            )

            plot_cmatrix(
                pipe,
                d.X_test.values,
                d.y_test.values,
                tag=model_name,
                threshold=metrics_train["threshold"],
            )
            if model_name != "rule_based":
                plot_shap(pipe, d.X_train, tag=f"{model_name}_train", X_train=d.X_train)
                plot_shap(pipe, d.X_test, tag=f"{model_name}_test", X_train=d.X_train)
            plot_roc(
                [metrics_train, metrics_test],
                [metrics_bt_train, metrics_bt_test],
                tag=f"{model_name}",
            )
            plot_prcurve(
                [metrics_train, metrics_test],
                [metrics_bt_train, metrics_bt_test],
                tag=f"{model_name}",
            )
            if (model_name == "random_forest") | (model_name == "xgboost"):
                logging.info("Feature importances")
                clf = pipe.named_steps["clf"]
                feature_importances = clf.feature_importances_
                for k, i in enumerate(feature_importances.argsort()[::-1]):
                    logging.info(
                        f"{k + 1}: {d.X_train.columns[i], feature_importances[i]}"
                    )
                    if k == 9:
                        break
            elif model_name == "decision_tree":
                plot_dtree(pipe.named_steps["clf"], d.X_train.columns.values)

        else:
            logging.warning("Invalid mode, should be train or test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process string")
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        help="Train model (with mlflow tracking) or test model (after training using best parameters)",
    )
    args = parser.parse_args()
    run(mode=args.mode)
