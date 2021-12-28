"""Functions for training models."""

import logging

import hyperopt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb

from config import MAX_EVALS
from model.evaluate import evaluate_classifier
from model.hyperparameters import search_space
from model.rule_based_models import RuleBasedClassifier
from utils.hyperopt import HyperoptHPOptimizer

loggers_to_silence = [
    "hyperopt.tpe",
    "hyperopt.fmin",
    "hyperopt.pyll.base",
]
for logger in loggers_to_silence:
    logging.getLogger(logger).setLevel(logging.ERROR)

models = {
    "rule_based": {
        "model": RuleBasedClassifier(),
    },
    "decision_tree": {
        "model": DecisionTreeClassifier(),
    },
    "logistic_regression": {
        "model": LogisticRegression(solver="lbfgs"),
    },
    "random_forest": {
        "model": RandomForestClassifier(),
    },
    "xgboost": {
        "model": xgb.XGBClassifier(),
    },
}


def get_sample_weights(y_train):
    """Compute sample weights for the training set."""
    y_values = np.sort(y_train.unique())
    class_weights = dict(
        zip(y_values, compute_class_weight("balanced", y_values, y_train))
    )  # can also be done in class
    sample_weights = np.array([class_weights[y] for y in y_train])
    return sample_weights


def get_objective(model, X_train, y_train, sample_weights=None, cv=None):
    """Get objective for hyperopt."""

    def objective(params):
        model.set_params(**params)
        metrics = evaluate_classifier(
            model,
            X_train.values,
            y_train.values,
            sample_weights=sample_weights,
            split="train",
            cv=cv,
            method="predict",
        )[0]

        return metrics

    return objective


def get_pipeline(model_name):
    """Get sklearn pipeline."""
    model = models[model_name]["model"]

    if model_name == "logistic_regression":
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("feature_selection", SelectKBest()),
                ("clf", model),
            ]
        )
    else:
        pipe = Pipeline([("clf", model)])
    return pipe


def train_classifier(
    model_name,
    X_train,
    y_train,
    sample_weights=None,
    cv=None,
    use_best_fit_parameters=False,
    data_checksum=None,
    bootstrap=False,
    feature_columns=None,
):
    """Train a classifier (sklearn pipeline) using hyperopt.

    Parameters
    ----------
    model_name : str
    X_train : pd.DataFrame
        Features for training
    y_train : pd.DataFrame
        labels for training
    sample_weights : list, optional
        Sample weights, by default None
    cv : class, optional
        Class for cross validation, by default None
    use_best_fit_parameters : bool, optional
        Rather than using hyperopt, fit a pre-determined set of parameters, by default False
    data_checksum : str, optional
        Data checksum to be logged by mlflow, by default None
    feature_columns : list-like, optional
        List of features, useful when input is not a pd.DataFrame, by default None

    Returns
    -------
    clf : class
        sklearn pipeline
    metrics : dict
        Dictionary with metrics
    """
    logging.info(f"Train: {model_name}")
    pipe = get_pipeline(model_name)

    if model_name == "rule_based":
        # Add feature names to the class
        RuleBasedClassifier.features = feature_columns

    else:
        if not use_best_fit_parameters:
            objective = get_objective(
                pipe, X_train, y_train, sample_weights=sample_weights, cv=cv
            )

            HYPERPARAMETERS_SPACE = search_space[model_name]
            assert data_checksum
            hp_optimizer = HyperoptHPOptimizer(
                objective,
                hyperparameters_space=HYPERPARAMETERS_SPACE,
                max_evals=MAX_EVALS,
                model_name=model_name,
                data_checksum=data_checksum,
            )
            best = hp_optimizer.optimize()
            optimal_hyperparameters = hyperopt.space_eval(HYPERPARAMETERS_SPACE, best)

            logging.info(
                f"Best hyperparameters ({model_name}): {optimal_hyperparameters}"
            )

            pipe.set_params(**optimal_hyperparameters)
        else:
            from model.best_fit_parameters import bf_hyperparameters

            pipe.set_params(**bf_hyperparameters[model_name])

    metrics, metrics_bt = evaluate_classifier(
        pipe,
        X_train,
        y_train,
        sample_weights=sample_weights,
        split="train",
        cv=cv,
        bootstrap=bootstrap,
        log=True,
    )
    clf = pipe.fit(X_train, y_train, clf__sample_weight=sample_weights)
    return clf, metrics, metrics_bt
