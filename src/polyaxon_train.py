import os

from polyaxon_client.tracking import Experiment

from local_train import train
from helpers.helpers import base_parser


def parse_arguments():
    parser = base_parser()
    return parser.parse_known_args()[0]


def train_polyaxon(args):
    # Start polyaxon experiment
    experiment = Experiment()

    # Start training
    cv_roc_auc, test_roc_auc, test_logloss = train(args)

    # Save artifacts
    experiment.outputs_store.upload_file(os.path.join(ARGS.model_dir, "model.pkl"))
    experiment.log_metrics(
        test_roc_auc=test_roc_auc, test_logloss=test_logloss, cv_roc_auc=cv_roc_auc
    )


def main():
    train_polyaxon(ARGS)


if __name__ == "__main__":
    ARGS = parse_arguments()
    main()
