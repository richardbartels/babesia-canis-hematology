"""Script to read and preprocess data."""

import hashlib
import os
import logging

import numpy as np
import pandas as pd

from data.features import features as _features
from data.features import clean_column_names


class Data:
    """Class containing data for training and testing."""

    def __init__(self, input_root: str = "data/raw", overwrite: bool = False):
        """Initialize Data class.

        Parameters
        ----------
        input_file
            Path to data files (xlxs)
        overwrite
            Overwrite processed data files.
        """
        self.input_root = input_root
        self.output_file_train = os.path.join(
            input_root.replace("raw", "processed"), "train.csv"
        )
        self.output_file_test = os.path.join(
            input_root.replace("raw", "processed"), "test.csv"
        )

        if (
            (os.path.exists(self.output_file_train))
            & (os.path.exists(self.output_file_test))
            & (not overwrite)
        ):
            logging.info(
                f"loading preprocessed ({self.output_file_train} & {self.output_file_train})"
            )
            self.train = pd.read_csv(self.output_file_train)
            self.test = pd.read_csv(self.output_file_test)
        else:
            logging.info("Preprocess data")
            self.train, self.test = self.get_data()

        for ix, df in enumerate([self.train, self.test]):
            assert (
                df.duplicated(keep=False).sum() == 0
            ), f"Dataset {ix} has duplicate rows."
            assert (
                df.columns.duplicated(keep=False).sum() == 0
            ), f"Dataset {ix} has duplicate columns."
        self.log_dataset_composition()

        self.create_features_labels()  # create dataframes with just features and just labels

        self.md5_checksum_train = hashlib.md5(self.train.to_csv().encode()).hexdigest()
        self.md5_checksum_test = hashlib.md5(self.test.to_csv().encode()).hexdigest()
        logging.info(f"MD5 checksum train set: {self.md5_checksum_train}")
        logging.info(f"MD5 checksum test set: {self.md5_checksum_test}")

    def __clean_features(self):
        self.features = clean_column_names(_features)
        self.train = clean_column_names(self.train)
        self.test = clean_column_names(self.test)

    def log_dataset_composition(self):
        """Log dataset composition."""
        logging.info(f"Train set count: {dict(self.train['Group'].value_counts())}")
        logging.info(
            f"Train unique patient per class: {dict(self.train.groupby('Group')['Identifier'].nunique())}"
        )
        logging.info(f"\nTest set count: {dict(self.test['Group'].value_counts())}")
        logging.info(
            f"Test unique patients per class: {dict(self.test.groupby('Group')['Identifier'].nunique())}"
        )

    def create_features_labels(self):
        """Create dataframes with features and labels."""
        # Ensure no overlapping animals between train and test split
        assert not any([id_ in self.test.Identifier for id_ in self.train.Identifier])

        self.__clean_features()
        # Select features
        self.X_train = self.train[self.features]
        self.X_test = self.test[self.features]

        # Select labels
        self.y_train = self.train["Group"]
        self.y_test = self.test["Group"]

        # Select IDs
        self.id_train = self.train["Identifier"]
        self.id_test = self.test["Identifier"]

        # Dropping features with NaNs
        features_to_drop = self.X_train.columns[self.X_train.isna().sum() != 0]
        logging.info(f"Dropping these features: {features_to_drop}")
        self.X_train.drop(columns=features_to_drop, inplace=True)
        self.X_test.drop(columns=features_to_drop, inplace=True)

    def get_data(self):
        """Load datasets from 2003-2013 and 2017-2020 and merge them."""
        # Load datasets
        datasets = [
            pd.read_excel(
                os.path.join(self.input_root, "Map1.xlsx")
            ),  # Used for training
            pd.read_excel(
                os.path.join(
                    self.input_root,
                    "bab-anapl-2017-2018 plus extra bab-anap.xlsm",  # Used for testing
                ),
                sheet_name="file",
            ),
        ]

        for ix, dataset in enumerate(datasets):
            mask = dataset.duplicated(keep="first")
            logging.info(f"Dataset {ix}: dropping {mask.sum()} duplicated items")
            datasets[ix] = dataset.loc[~mask]

        assert all([dataset.duplicated(keep=False).sum() == 0 for dataset in datasets])

        # Dataset 1 has fewer columns, make them compatible
        datasets[1] = datasets[1].rename(columns={"0=geen, 1=bab, 2=anapl": "Group"})
        all_columns = np.array(datasets[0].columns)
        column_in_common = np.array([c in datasets[1].columns for c in all_columns])
        drop_columns = []
        for col in all_columns[~column_in_common]:
            logging.info(f"Dropping {col}: not in the second dataset")
            drop_columns.append(col)
        all_columns = np.setdiff1d(all_columns, np.array(drop_columns))
        assert np.setdiff1d(all_columns, np.unique(all_columns)).size == 0

        # Ensure no overlapping animals between datasets
        assert not any(
            [id_ in datasets[1].Identifier for id_ in datasets[0].Identifier]
        )

        # Concatenate columns
        for ix, _ in enumerate(datasets):
            # Remove anaplasma:
            datasets[ix]["Group"] = datasets[ix].Group.replace({2: 0})
            datasets[ix] = datasets[ix].loc[:, all_columns]

        train_data = datasets[0]
        test_data = datasets[1]
        train_data.to_csv(self.output_file_train)
        test_data.to_csv(self.output_file_test)
        return train_data, test_data
