from csv import reader
from json import load
from math import sqrt, isnan
from multiprocessing import Array, Process, Queue
from glob import glob
from os import makedirs
import logging

import numpy as np
import pandas as pd

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%H:%M:%S')
    # datefmt='%Y-%m-%d %H:%M:%S')

max_rows = 10
datasets = ["allmusic", "tagtraum", "discogs", "lastfm"]
modes = ["train", "validation"]
categorical_features = ["key_key", "key_scale", "chords_key", "chords_scale"]
categorical_levels = {"key_key": ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"],
                      "key_scale": ["minor", "major"],
                      "chords_key": ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"],
                      "chords_scale": ["minor", "major"]}
data_dir = "data"
processed_dir = "processed"

with open("data/genres.txt", "r") as file:
    all_genres = file.read().split(";")


def get_nested_dict_values(d):
    if isinstance(d, dict):
        for v in d.values():
            yield from get_nested_dict_values(v)
    elif isinstance(d, list):
        for v in d:
            yield from get_nested_dict_values(v)
    else:
        yield d


def process_observation(row, mode, dataset, sum_x, sum_x2, cnt_x, queue):
    mbid = row[0]
    genres = row[2:]
    with open(f"{data_dir}/acousticbrainz-mediaeval-{mode}/{mbid[:2]}/{mbid}.json", "r") as file:
        observation = load(file)
    observation.pop("metadata", None)  # metadata contains non-numeric data which is largely not present in the test set
    observation["rhythm"].pop("beats_position", None)  # beats_position seems to be removed, probably because it is of variable length
    # observation["rhythm"]["beats_loudness"].pop("median", None) # beats_loudness*.median is missing in some observations
    observation["rhythm"]["beats_loudness_band_ratio"].pop("median", None)
    features = list(get_nested_dict_values(observation))
    # -- one hot encode all remaining categorical values inplace
    for category in categorical_levels:
        feature_value = observation["tonal"][category]
        category_length = len(categorical_levels[category])
        level_ix = categorical_levels[category].index(feature_value)
        one_hot_encoded = np.eye(1, M=category_length, k=level_ix)[0]
        feature_ix = features.index(feature_value)
        features.pop(feature_ix)
        features[feature_ix:feature_ix] = one_hot_encoded
    features.insert(0, mbid)  # insert observation id in resulting feature set
    genres_encoded = [all_genres.index(genre) for genre in genres if genre in all_genres]
    genres_encoded.insert(0, mbid)
    for i in range(len(features) - 1):
        if not np.isnan(features[i+1]):
            sum_x[i] += features[i + 1]
            sum_x2[i] += features[i + 1] ** 2
            cnt_x[i] += 1
    queue.put((mode, dataset, str(features)[1:-1] + "\n", str(genres_encoded)[1:-1] + "\n"))


def write_output(queue):
    """ Consumer which consumes processed feature and genre vectors from the queue and writes them into a file. """
    while True:
        item = queue.get()
        if item is None:
            break
        mode = item[0]
        dataset = item[1]
        features = item[2]
        genres = item[3]
        with open(f"{processed_dir}/{mode}/{dataset}.csv", "a") as features_file,\
                open(f"{processed_dir}/{mode}/{dataset}.genres.csv", "a") as genres_file:
            features_file.write(features)
            genres_file.write(genres)


def main():
    features_dim = 2669
    # -- sum_x and sum_x2 are the cumulative sum (of squares) of all features. these values are needed to compute the
    # -- mean and standard deviation in a memory-efficient manner after all observations have been passed
    sum_x = Array('d', features_dim)
    sum_x2 = Array('d', features_dim)
    cnt_x = Array('d', features_dim)
    output_queue = Queue()
    for dataset in datasets:
        for mode in modes:
            logging.info(f"Preprocessing {mode} mode of {dataset} dataset")
            row_counter = 0
            makedirs(f"{processed_dir}/{mode}", exist_ok=True)
            with open(f"{data_dir}/acousticbrainz-mediaeval-{dataset}-{mode}.tsv", 'r') as dataset_file:
                rows = reader(dataset_file, delimiter="\t")
                next(rows)
                for row in rows:
                    if row_counter > max_rows:
                        break
                    process_observation(row, mode, dataset, sum_x, sum_x2, cnt_x, output_queue)
                    row_counter += 1
                    if row_counter % 100 == 0:
                        logging.info(f"Row counter: {row_counter}")
    output_queue.put(None)
    write_output(output_queue)
    logging.info("Finished first preprocessing pass")

    means = []
    sdevs = []
    for i in range(features_dim):
        mean = sum_x[i] / cnt_x[i] 
        sd = sqrt((sum_x2[i] / cnt_x[i]) - (mean * mean))
        # sd == 0? set it to 1 for avoiding division by zero
        if sd == 0.:
            sd = 1.
        means.append(mean)
        sdevs.append(sd)
    with open(f"{processed_dir}/means.csv", 'w') as file:
        file.write(",".join(map(str, means)))
    with open(f"{processed_dir}/sdevs.csv", 'w') as file:
        file.write(",".join(map(str, sdevs)))
    logging.info("Calculated means and standard deviations")

    logging.info("Scale preprocessed datasets")
    rows_to_read = 350000
    for dataset in datasets:
        for mode in modes:
            file_path = f"{processed_dir}/{mode}/{dataset}"
            logging.info(f"Scaling {file_path}")
            row_number = 0
            while True:
                try:
                    df = pd.read_csv(f"{file_path}.csv", skiprows=row_number, nrows=rows_to_read, index_col=0, header=None)
                except pd.errors.EmptyDataError:
                    break
                row_number += rows_to_read
                df -= means
                df /= sdevs
                df.fillna(0)
                df.to_csv(f"{file_path}.features.clean.std.csv", mode="a", index=True, header=False)


if __name__ == "__main__":
    main()

