import os
import argparse

from joblib import dump
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from polyaxon_client.tracking import Experiment

from helpers import calculate_metrics, load_data


def build_model(param_grid, cv=5, scoring="roc_auc"):
    model = GridSearchCV(
        LogisticRegression(solver="liblinear"),
        param_grid,
        scoring=scoring,
        cv=cv,
    )
    return model


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--penalty',
        type=str,
        choices=["l1", "l2"],
        required=True,
        default="l2",
        help='Choose regularisation penalty (l1, l2).',
    )

    parser.add_argument(
        '--C',
        type=float,
        required=True,
        default=1.0,
        help='Choose regularization strength.',
    )

    parser.add_argument(
        '--is_polyaxon_env',
        type=int,
        required=False,
        default=0,
        help='Indicate if running locally (0) or in Polyaxon (1).',
    )

    return parser.parse_known_args()[0]


def main():

    # Start Polyaxon tracking
    # ======= POLYAXON =======
    if ARGS.is_polyaxon_env:
        experiment = Experiment()
    # ========================

    # load data and split train and test sets
    x_data, y_data = load_data(DATA_DIR)
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.33, random_state=42
    )

    # train model
    param_grid = [{"penalty": [ARGS.penalty], "C": [ARGS.C]}]

    model = build_model(param_grid, cv=5, scoring="roc_auc")
    model.fit(x_train, y_train)

    # generate artifacts (model, metrics, graphs etc)
    dump(model, "model.pkl")
    roc_auc, logloss = calculate_metrics(model, x_test, y_test)
    print(f"cv roc_auc: {model.best_score_:.3f}")
    print(f"test roc_auc: {roc_auc:.3f}")
    print(f"test logloss: {logloss:.3f}")

    # persist artifacts
    # ======= POLYAXON =======
    if ARGS.is_polyaxon_env:
        experiment.outputs_store.upload_file("model.pkl")
        experiment.log_metrics(
            roc_auc=roc_auc, logloss=logloss, cv_roc_auc=model.best_score_
        )
    # ======= POLYAXON =======


if __name__ == "__main__":
    ARGS = parse_arguments()
    DATA_DIR = "/data/project_1"
    main()
