import os
import collections

import yaml


MODEL_PATH = "/opt/ml/model"  # Any artifacts saved in this folder are uploaded to S3 for model hosting after the training job completes.
INPUT_PATH = "/opt/ml/input"
OUTPUT_PATH = "/opt/ml/output"
INPUT_DATA_PATH = "/opt/ml/input/data"  # These artifacts are compressed and uploaded to S3 to the same S3 prefix as the model artifacts.
OUTPUT_DATA_PATH = "/opt/ml/output/data"
INPUT_CONFIG_PATH = "/opt/ml/input/config"

HYPERPARAMETERS_FILE = "hyperparameters.json"
RESOURCE_CONFIG_FILE = "resourceconfig.json"
INPUT_DATA_CONFIG_FILE = "inputdataconfig.json"


class HyperParameters(collections.Mapping):
    """dict of the hyperparameters provided in the training job. Allows casting of the hyperparameters
    in the `get` method.
    """

    def __init__(self, hyperparameters_dict):
        self.hyperparameters_dict = hyperparameters_dict

    def __getitem__(self, key):
        return self.hyperparameters_dict[key]

    def __len__(self):
        return len(self.hyperparameters_dict)

    def __iter__(self):
        return iter(self.hyperparameters_dict)

    def get(self, key, default=None, object_type=None):
        """Has the same functionality of `dict.get`. Allows casting of the values using the additional attribute
        `object_type`:
        Args:
            key: hyperparameter name
            default: default hyperparameter value
            object_type: type that the hyperparameter wil be casted to.
        Returns:
        """
        try:
            value = self.hyperparameters_dict[key]
            return object_type(value) if object_type else value
        except KeyError:
            return default

    def __str__(self):
        return str(self.hyperparameters_dict)

    def __repr__(self):
        return str(self.hyperparameters_dict)


class TrainerEnvironment(
    collections.namedtuple(
        "TrainerEnvironment",
        [
            "input_dir",
            "input_config_dir",
            "model_dir",
            "output_dir",
            "hyperparameters",
            "resource_config",
            "input_data_config",
            "output_data_dir",
            "channel_dirs",
        ],
    )
):
    def __new__(
        cls,
        input_dir,
        input_config_dir,
        model_dir,
        output_dir,
        hyperparameters,
        resource_config,
        input_data_config,
        output_data_dir,
        channel_dirs,
    ):
        return super(TrainerEnvironment, cls).__new__(
            cls,
            input_dir,
            input_config_dir,
            model_dir,
            output_dir,
            hyperparameters,
            resource_config,
            input_data_config,
            output_data_dir,
            channel_dirs,
        )


def load_config(path):
    with open(path, "r") as f:
        return yaml.load(f)


def load_hyperparameters():
    return HyperParameters(
        load_config(os.path.join(INPUT_CONFIG_PATH, HYPERPARAMETERS_FILE))
    )


def load_resource_config():
    return load_config(os.path.join(INPUT_CONFIG_PATH, RESOURCE_CONFIG_FILE))


def load_input_data_config():
    return load_config(os.path.join(INPUT_CONFIG_PATH, INPUT_DATA_CONFIG_FILE))


def get_channel_dir(channel):
    return os.path.join(INPUT_DATA_PATH, channel)


def create_trainer_environment():
    """
    Returns: an instance of `TrainerEnvironment`
    """
    resource_config = load_resource_config()

    input_data_config = load_input_data_config()
    channel_dirs = {channel: get_channel_dir(channel) for channel in input_data_config}

    env = TrainerEnvironment(
        input_dir=INPUT_PATH,
        input_config_dir=INPUT_CONFIG_PATH,
        model_dir=MODEL_PATH,
        output_dir=OUTPUT_PATH,
        output_data_dir=OUTPUT_DATA_PATH,
        channel_dirs=channel_dirs,
        hyperparameters=load_hyperparameters(),
        resource_config=resource_config,
        input_data_config=load_input_data_config(),
    )
    return env
