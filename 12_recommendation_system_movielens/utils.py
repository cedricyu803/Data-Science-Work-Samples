import os

import pandas as pd
import yaml

# import logging
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
# logger.addHandler(logging.StreamHandler())


def load_yaml(filepath):
    with open(filepath, "r") as f:
        return yaml.safe_load(f)


def load_dataset(data_dir):
    ratings = pd.read_csv(
        os.path.join(data_dir, "ratings.csv"),
    ).sort_values(["timestamp"])  # sort by timestamp for chronological splitting
    movies = pd.read_csv(
        os.path.join(data_dir, "movies.csv"),
    ).sort_values(["movieId"])
    # logger.info('Loaded dataset')
    return ratings, movies
