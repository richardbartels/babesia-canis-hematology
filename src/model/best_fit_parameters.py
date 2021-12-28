"""Best fit parameters.

Best fit parameters are selected from the MLFlow logging. In case of equally perfoming models Occam's razor is applied.
"""

bf_hyperparameters = {
    "rule_based": {},
    "decision_tree": {
        "clf__max_depth": 3,
        "clf__min_samples_split": 98,
        "clf__min_samples_leaf": 23,
        "clf__random_state": 42,
    },
    "logistic_regression": {
        "feature_selection__k": "all",
        "clf__C": 0.5,
        "clf__random_state": 42,
    },
    "random_forest": {
        "clf__max_depth": 3,
        "clf__max_features": "log2",
        "clf__min_samples_split": 4,
        "clf__n_estimators": 80,
        "clf__random_state": 42,
    },
    "xgboost": {
        "clf__objective": "binary:logistic",
        "clf__learning_rate": 0.37,
        "clf__max_depth": 4,
        "clf__n_estimators": 48,
        "clf__random_state": 42,
    },
}
