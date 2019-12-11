import os

import pandas as pd
from sklearn.datasets import make_classification


def create_data_directory(data_dir):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)


def get_data(data_dir):
    x_data, y_data = make_classification(n_samples=1000000, n_features=10)

    pd.DataFrame(x_data).to_csv(os.path.join(data_dir, "x_data.csv"))
    pd.DataFrame(y_data).to_csv(os.path.join(data_dir, "y_data.csv"))


def main():
    create_data_directory(DATA_DIR)
    get_data(DATA_DIR)


if __name__ == "__main__":
    DATA_DIR = "/data/project_1"
    main()
