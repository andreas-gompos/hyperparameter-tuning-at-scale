import os
import argparse

import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.datasets import make_classification

def base_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--penalty",
        type=str,
        choices=["l1", "l2"],
        default="l2",
        help="Choose regularisation penalty (l1, l2).",
    )

    parser.add_argument(
        "--C", type=float, default=1.0, help="Choose regularization strength.",
    )

    parser.add_argument(
        "--model_dir",
        type=str,
        help="Local directory to save the model",
    )

    parser.add_argument(
        "--train_channel",
        type=str,
        help="Local directory where training data lives",
    )

    return parser


def calculate_metrics(model, x_test, y_test):
    y_pred = model.predict_proba(x_test)[:, 1]
    y_score = y_pred.copy()
    y_pred[y_pred >= 0.5] = 1
    y_pred[y_pred < 0.5] = 0
    roc_auc = roc_auc_score(y_test, y_score)
    logloss = log_loss(y_test, y_score)

    return roc_auc, logloss


def create_data_directory(data_dir):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)


def get_data(data_dir):
    x_data, y_data = make_classification(n_samples=10000, n_features=10)

    x_data = pd.DataFrame(x_data)
    y_data = pd.DataFrame(y_data)

    create_data_directory(data_dir)
    x_data.to_csv(os.path.join(data_dir, "x_data.csv"))
    y_data.to_csv(os.path.join(data_dir, "y_data.csv"))


def load_data(data_dir):

    x_data = pd.read_csv(os.path.join(data_dir, "x_data.csv"), index_col=0)
    y_data = pd.read_csv(
        os.path.join(data_dir, "y_data.csv"), index_col=0
    ).values.reshape(-1)
    return x_data, y_data
