import os
import argparse

from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split

from helpers.helpers import calculate_metrics, load_data, base_parser


def build_model(param_grid, cv=5, scoring="roc_auc"):
    model = GridSearchCV(
        LogisticRegression(solver="liblinear"), param_grid, scoring=scoring, cv=cv,
    )
    return model


def parse_arguments():
    parser = base_parser()
    return parser.parse_known_args()[0]


def train(args):
    print(args)
    # load data and split train and test sets
    x_data, y_data = load_data(args.train_channel)
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.33, random_state=42
    )

    # train model
    param_grid = [{"penalty": [args.penalty], "C": [args.C]}]

    model = build_model(param_grid, cv=5, scoring="roc_auc")
    model.fit(x_train, y_train)
    print(model.best_estimator_)

    # generate artifacts (model, metrics, graphs etc)
    dump(model, os.path.join(args.model_dir, "model.pkl"))
    test_roc_auc, test_logloss = calculate_metrics(model, x_test, y_test)
    print(f"cv_roc_auc: {model.best_score_:.10f};")
    print(f"test_roc_auc: {test_roc_auc:.10f};")
    print(f"test_logloss: {test_logloss:.10f};")
    return model.best_score_, test_roc_auc, test_logloss


def main():
    train(ARGS)


if __name__ == "__main__":
    ARGS = parse_arguments()
    main()
