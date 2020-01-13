import argparse

from helpers.helpers import get_data


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_channel",
        type=str,
        help="Local directory where training data lives",
        required=True,
    )
    return parser.parse_known_args()[0]


def main():
    get_data(ARGS.train_channel)


if __name__ == "__main__":
    ARGS = parse_arguments()
    main()
