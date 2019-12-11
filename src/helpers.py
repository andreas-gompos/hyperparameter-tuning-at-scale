import os

import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss


def calculate_metrics(model, x_test, y_test):
    y_pred = model.predict_proba(x_test)[:, 1]
    y_score = y_pred.copy()
    y_pred[y_pred >= 0.5] = 1
    y_pred[y_pred < 0.5] = 0
    roc_auc = roc_auc_score(y_test, y_score)
    logloss = log_loss(y_test, y_score)

    return roc_auc, logloss


def load_data(data_dir):
    x_data = pd.read_csv(os.path.join(data_dir, "x_data.csv"), index_col=0)
    y_data = pd.read_csv(
        os.path.join(data_dir, "y_data.csv"), index_col=0
    ).values.reshape(-1)
    return x_data, y_data
