import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, \
    recall_score, f1_score, log_loss
from sklearn.base import BaseEstimator

from numpy.typing import ArrayLike
from typing import Callable


def gini_stability(y_test: ArrayLike,
                   y_score: ArrayLike,
                   weeks: pd.Series,
                   w_fallingrate: float = 88.0,
                   w_resstd: float = -0.5) -> float:
    """
    Evaluation gini stabiliti score according groups.

    A gini score is calculated for predictions corresponding to each week:
        gini=2∗AUC−1
    A linear regression, a⋅x+b, is fit through the weekly gini scores,
    and a falling_rate is calculated as min(0,a).
    This is used to penalize models that drop off in predictive ability.
    Finally, the variability of the predictions are calculated by taking
    the standard deviation of the residuals from the above linear regression,
    applying a penalty to model variablity.

    The final metric is calculated as:
    stability metric=mean(gini)+88.0⋅min(0,a)−0.5⋅std(residuals)
    """
    df_score = pd.DataFrame(y_score, columns=["score"])
    df_score["week"] = weeks.to_numpy()
    df_score["target"] = y_test.to_numpy()

    gini_in_time = df_score.loc[:, ["week", "target", "score"]] \
        .sort_values("week") \
        .groupby("week")[["target", "score"]] \
        .apply(lambda x: 2 * roc_auc_score(x["target"], x["score"]) - 1) \
        .tolist()

    x = np.arange(len(gini_in_time))
    y = gini_in_time
    a, b = np.polyfit(x, y, 1)
    y_hat = a * x + b
    residuals = y - y_hat
    res_std = np.std(residuals)
    mean_gini = np.mean(gini_in_time)
    metric = mean_gini + w_fallingrate * min(0, a) + w_resstd * res_std
    return metric


def get_metrics(y_test: pd.Series, y_pred: pd.Series, y_score: ArrayLike,
                weeks: pd.Series, name: str) -> pd.DataFrame:
    """
    Generating a dataframe with metrics for a classification task
    """
    df_metrics = pd.DataFrame()

    df_metrics['model'] = [name]
    df_metrics['Accuracy'] = accuracy_score(y_test, y_pred)
    df_metrics['ROC_AUC'] = roc_auc_score(y_test, y_score[:, 1])
    df_metrics['Precision'] = precision_score(y_test, y_pred)
    df_metrics['Recall'] = recall_score(y_test, y_pred)
    df_metrics['f1'] = f1_score(y_test, y_pred)
    df_metrics['gini_stability'] = gini_stability(y_test, y_score[:, 1], weeks)

    return df_metrics


def check_overfitting(metric_fun: Callable,
                      y_train: pd.Series,
                      y_test: pd.Series,
                      X_train: pd.DataFrame = None,
                      X_test: pd.DataFrame = None,
                      model: BaseEstimator = None,
                      y_pred_train: pd.DataFrame = None,
                      y_pred_test: pd.DataFrame = None):
    """
    Check overfitting
    """
    if model is None:
        value_train = metric_fun(y_train, y_pred_train)
        value_test = metric_fun(y_test, y_pred_test)
    else:
        if metric_fun.__name__ == 'roc_auc_score':
            y_pred_train = model.predict_proba(X_train)[:, 1]
            y_pred_test = model.predict_proba(X_test)[:, 1]
        else:
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
        value_train = metric_fun(y_train, y_pred_train)
        value_test = metric_fun(y_test, y_pred_test)

    print(f'{metric_fun.__name__} train: %.3f' % value_train)
    print(f'{metric_fun.__name__} test: %.3f' % value_test)
    print(f'delta = {(abs(value_train - value_test) / value_test * 100):.1f} %')
