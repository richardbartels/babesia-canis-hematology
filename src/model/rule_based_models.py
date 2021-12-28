import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.multiclass import unique_labels


class RuleBasedClassifier(BaseEstimator, TransformerMixin):
    """Sklearn wrapperr around rule based classifier."""

    def fit(self, X, y=None, **kwargs):
        """Fit method."""
        self.classes_ = unique_labels(y)
        return self

    def predict(self, X):
        """Predict method."""
        if type(X) == np.ndarray:
            X = pd.DataFrame(X, columns=self.features)
        y = (
            (X["PLT(x10E09 cells/L)"].values < 102.0) & (X["%LUC(%)"].values > 1.8)
        ).astype("float")
        return y

    def predict_proba(self, X):
        """Placedholder for the predict proba method.

        Retruns the same values as predict, but one-hot encoded.
        """
        pred = pd.get_dummies(self.predict(X)).values.astype("float")
        return pred
