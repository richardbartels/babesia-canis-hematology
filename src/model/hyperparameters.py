"""Configuration for hyperparameters."""
from hyperopt import hp
from hyperopt.pyll import scope
import numpy as np

search_space = {
    "decision_tree": hp.choice(
        "dtree",
        [
            {
                "clf__max_depth": scope.int(
                    hp.qloguniform("dtree_max_depth", np.log(2), np.log(1e1), 1)
                ),
                "clf__min_samples_split": scope.int(
                    hp.qloguniform("dtree_min_samples_split", np.log(2), np.log(100), 1)
                ),
                "clf__min_samples_leaf": scope.int(
                    hp.uniform("dtree_min_samples_leaf", 1, 25)
                ),
                "clf__random_state": scope.int(
                    hp.choice(
                        "dtree_random_state",
                        [
                            42,
                        ],
                    )
                ),
            }
        ],
    ),
    "logistic_regression": hp.choice(
        "logreg",
        [
            {
                # 'feature_selection__k': scope.int(hp.qloguniform('logreg_k', np.log(1),
                #                                                  np.log(100), 1)),
                "feature_selection__k": hp.choice(
                    "logreg_k", [1, 2, 3, 4, 5, 10, 15, 20, "all"]
                ),
                "clf__C": hp.qloguniform("logreg_C", np.log(0.001), np.log(10), 0.001),
                "clf__random_state": scope.int(
                    hp.choice(
                        "logreg_random_state",
                        [
                            42,
                        ],
                    )
                ),
            }
        ],
    ),
    "random_forest": hp.choice(
        "rf",
        [
            {
                "clf__max_depth": scope.int(
                    hp.qloguniform("rf_max_depth", np.log(2), np.log(1e1), 1)
                ),
                "clf__max_features": hp.choice("rf_max_features", ["sqrt", "log2"]),
                "clf__min_samples_split": scope.int(
                    hp.qloguniform("rf_min_samples_split", np.log(2), np.log(100), 1)
                ),
                "clf__n_estimators": scope.int(
                    hp.qloguniform("rf_n_estimators", 0, np.log(100), 1)
                ),
                "clf__random_state": scope.int(
                    hp.choice(
                        "rf_random_state",
                        [
                            42,
                        ],
                    )
                ),
            }
        ],
    ),
    "xgboost": hp.choice(
        "xgb",
        [
            {
                "clf__objective": hp.choice(
                    "xgb_objective",
                    [
                        "binary:logistic",
                    ],
                ),
                "clf__learning_rate": hp.qloguniform(
                    "xgb_learning_rate", np.log(0.001), np.log(0.3), 0.005
                ),
                "clf__max_depth": scope.int(
                    hp.qloguniform("xgb_max_depth", np.log(2), np.log(1e1), 1)
                ),
                "clf__n_estimators": scope.int(
                    hp.qloguniform("xgb_n_estimators", 0, np.log(100), 1)
                ),
                "clf__random_state": scope.int(
                    hp.choice(
                        "xgb_random_state",
                        [
                            42,
                        ],
                    )
                ),
            }
        ],
    ),
}
