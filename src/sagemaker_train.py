import os
import argparse

from local_train import train
from helpers.sagemaker_helpers import create_trainer_environment


def parse_arguments():

    env = create_trainer_environment()
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--penalty",
        type=str,
        choices=["l1", "l2"],
        default=env.hyperparameters.get("penalty", object_type=str),
        help="Choose regularisation penalty (l1, l2).",
    )

    parser.add_argument(
        "--C",
        type=float,
        default=env.hyperparameters.get("C", object_type=float),
        help="Choose regularization strength.",
    )

    parser.add_argument(
        "--model_dir",
        type=str,
        default=env.model_dir,
        help="Local directory to save the model",
    )

    parser.add_argument(
        "--train_channel",
        type=str,
        default=env.channel_dirs["train"],
        help="Local directory where training data lives",
    )

    return parser.parse_known_args()[0]


def main():
    train(ARGS)


if __name__ == "__main__":

    ARGS = parse_arguments()
    main()
