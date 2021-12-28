"""Functions to make plots."""

import logging
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import pandas as pd
from sklearn.metrics import auc, ConfusionMatrixDisplay, confusion_matrix
from sklearn import tree
import shap
from subprocess import check_call

mpl_logger = logging.getLogger("matplotlib")
mpl_logger.setLevel(logging.WARNING)


linewidth = 2
rcParams["savefig.format"] = "png"


def plot_cmatrix(clf, X, y, tag, threshold):
    """Plot confusion matrix."""
    plt.figure()
    probas = clf.predict_proba(X)[:, -1]
    y_pred = (probas >= threshold).astype("int")
    disp = ConfusionMatrixDisplay(
        confusion_matrix(y, y_pred), display_labels=np.array(["-", "+"])
    )
    disp.plot(values_format="4g")
    plt.savefig(f"plots/{tag}_confusion_matrix")
    plt.close()


def plot_prcurve(metrics_list, metrics_bt_list, tag):
    """Plot receiver-operator characteristic curve."""
    plt.figure()
    colors = ["darkorange", "blue"]
    labels = ["Train", "Validation"]
    for ix, (metrics, metrics_bt) in enumerate(zip(metrics_list, metrics_bt_list)):
        plt.plot(
            metrics["recall"],
            metrics["precision"],
            color=colors[ix],
            lw=linewidth,
            linestyle="-" if ix == 0 else "--",
            label=labels[ix],
        )
        if ix == 0:
            plt.scatter(
                metrics["tpr"], metrics["ppv"], marker="*", s=100, color="k", zorder=5
            )
        if metrics_bt is not None:
            plt.fill_between(
                metrics_bt["recall"],
                metrics_bt["precision"][0],
                metrics_bt["precision"][-1],
                color=colors[ix],
                alpha=0.3,
            )

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower right")
    plt.savefig(f"plots/{tag}_PRcurve", dpi=300)
    plt.close()


def plot_roc(metrics_list, metrics_bt_list, tag):
    """Plot receiver-operator characteristic curve."""
    colors = ["darkorange", "blue"]
    labels = ["Train", "Validation"]
    plt.figure()
    for ix, (metrics, metrics_bt) in enumerate(zip(metrics_list, metrics_bt_list)):
        roc_auc = auc(metrics["fpr_roc"], metrics["tpr_roc"]) * 100
        if ix == 0:
            plt.scatter(
                metrics["fpr"], metrics["tpr"], marker="*", s=100, color="k", zorder=5
            )
        plt.plot(
            metrics["fpr_roc"],
            metrics["tpr_roc"],
            color=colors[ix],
            lw=linewidth,
            linestyle="-" if ix == 0 else "--",
            label=f"{labels[ix]}\nAUC = {roc_auc:.1f}% ({metrics_bt['auroc'][0]*100:0.1f}-{metrics_bt['auroc'][-1]*100:.1f}, 95% C.I.)",
        )
        if metrics_bt is not None:
            plt.fill_between(
                metrics_bt["fpr"],
                metrics_bt["tpr"][0],
                metrics_bt["tpr"][-1],
                color=colors[ix],
                alpha=0.3,
            )

    plt.plot([0, 1], [0, 1], color="k", lw=0.5, linestyle=":")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC")
    plt.legend(loc="lower right")
    plt.savefig(f"plots/{tag}_ROC.png", dpi=600)
    plt.close()


def plot_dtree(clf, feature_names):
    """Plot decision tree."""
    out_file = "plots/decision_tree"
    tree.export_graphviz(
        clf,
        out_file=f"{out_file}.dot",
        feature_names=feature_names,
        precision=1,
        label="all",
        impurity=False,
        proportion=False,
        filled=True,
        class_names=[r"Negative", "Positive"],
    )
    check_call(
        ["dot", "-Tpng", "-Gdpi=600", f"{out_file}.dot", "-o", f"{out_file}.png"]
    )


def plot_shap(pipe, X, tag, X_train):
    """Create shaply plots."""
    # load JS visualization code
    shap.initjs()

    clf = pipe.named_steps["clf"]
    if "logistic_regression" in tag:
        fs = pipe.named_steps["feature_selection"]
        feature_names = X.columns[fs.get_support()]  # k-best Features
        X = pipe[:-1].transform(X)
        X = pd.DataFrame(X, columns=feature_names)
        masker = shap.maskers.Independent(data=X_train)
        explainer = shap.LinearExplainer(clf, masker=masker)
    else:
        assert len(pipe.named_steps) == 1
        explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X)

    plt.figure()
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.savefig(f"plots/{tag}_shap_bar", bbox_inches="tight")
    plt.close()

    plt.figure()
    ndim_shap = np.array(shap_values).ndim

    if ndim_shap == 2:
        _shap_values = shap_values
    elif ndim_shap == 3:
        _shap_values = shap_values[1]
    shap.summary_plot(_shap_values, X, plot_type="dot", show=False)
    plt.savefig(f"plots/{tag}_shap_dot.png", bbox_inches="tight", dpi=600)
    plt.close()
