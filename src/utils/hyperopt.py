"""Addons for hyperopt."""

from hyperopt import fmin, tpe, Trials, STATUS_OK
import mlflow


class HyperoptHPOptimizer(object):
    """Class for hyperparameter tracking.

    Class for hyperparameter tracking in mlflow as suggested in https://github.com/mlflow/mlflow/issues/326
    """

    def __init__(
        self,
        generate_loss,
        hyperparameters_space,
        max_evals,
        model_name,
        data_checksum=None,
    ):
        self.generate_loss = generate_loss
        self.trials = Trials()
        self.max_evals = max_evals
        self.hyperparameters_space = hyperparameters_space
        self.model_name = model_name
        self.data_checksum = data_checksum

    def get_loss(self, hyperparameters):
        """Compute loss and track hyperparemeters, loss and scores with mlflow."""
        experiment_id = mlflow.set_experiment(f"{self.model_name}")
        with mlflow.start_run(experiment_id=experiment_id):
            print("Training with the following hyperparameters: ")
            print(hyperparameters)
            for k, v in hyperparameters.items():
                mlflow.log_param(k, v)
            metrics = self.generate_loss(hyperparameters)
            loss = -metrics["auroc"]
            mlflow.set_tag("model", self.model_name)
            mlflow.set_tag("data checksum", self.data_checksum)
            mlflow.log_metric("loss", loss)
            mlflow.log_metric("accuracy", metrics["accuracy"])
            mlflow.log_metric("weighted accuracy", metrics["weighted_accuracy"])
            mlflow.log_metric("auroc", metrics["auroc"])
            mlflow.log_metric("sensitivity", metrics["sensitivity"])
            mlflow.log_metric("specificity", metrics["specificity"])
            return {"loss": loss, "status": STATUS_OK}

    def optimize(self):
        """
        Create optimization function.

        This is the optimization function that given a space of
        hyperparameters and a scoring function, finds the best hyperparameters.
        """
        # Use the fmin function from Hyperopt to find the best hyperparameters
        # Here we use the tree-parzen estimator method.
        best = fmin(
            self.get_loss,
            self.hyperparameters_space,
            algo=tpe.suggest,
            trials=self.trials,
            max_evals=self.max_evals,
        )
        return best
