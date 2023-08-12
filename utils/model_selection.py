import pathlib

import pandas as pd
from tqdm.notebook import tqdm
import pathlib
from copy import deepcopy

import pandas as pd


def split_train_val_test(data_dir: pathlib.Path, save_dir: pathlib.Path):
    assert data_dir.exists()
    assert save_dir.exists()

    # calc datasize
    train_path_list = sorted(list(data_dir.glob("train_*")))
    df = pd.read_csv(train_path_list[0], sep="\t", header=None)
    chunk_size = len(df)
    df = pd.read_csv(train_path_list[-1], sep="\t", header=None)
    remainder_size = len(df)
    data_size = chunk_size * (len(train_path_list) - 1) + remainder_size
    print(f"chunk size: {chunk_size}, remainder size: {remainder_size}, data_size: {data_size}")
    test_file_num = round((len(train_path_list) - 1) * 0.3)
    val_file_num = round(((len(train_path_list) - 1) - test_file_num) * 0.3)
    train_file_num = (len(train_path_list) - 1) - test_file_num - val_file_num
    print(
        f"train file num: {train_file_num}, val file num: {val_file_num}, test file num: {test_file_num}"
    )

    # load data
    label_column = "label"
    feature_columns = [
        *["count_feature_{}".format(i) for i in range(13)],
        *["category_feature_{}".format(i) for i in range(26)],
    ]
    names = [
        label_column,
        *feature_columns,
    ]

    train_file_idx = range(train_file_num)
    val_file_idx = range(train_file_num, train_file_num + val_file_num)
    test_file_idx = range(
        train_file_num + val_file_num, train_file_num + val_file_num + test_file_num
    )

    for type, file_idx in tqdm(
        zip(["train", "val", "test"], [train_file_idx, val_file_idx, test_file_idx])
    ):
        tqdm.write(f"build {type} dataset")
        for i in tqdm(file_idx):
            df = pd.read_csv(train_path_list[i], sep="\t", header=None, names=names)
            df.to_csv(save_dir / f"{type}_{i:02d}.csv", index=False)


class CategoryEncoder:
    def __init__(self, fillna_value: str = "#nan") -> None:
        self.fillna_value = fillna_value
        self.is_fitted = False

    def fit(self, category_df: pd.DataFrame) -> None:
        """Fit Category Encoder.

        Args:
            X: _description_
        """
        self.category2idx_dict = dict()
        filled_category_df = category_df.astype(str).fillna(self.fillna_value)
        for category_column in filled_category_df.columns:
            category2idx = {
                h: i
                for i, h in enumerate(
                    sorted(
                        list(
                            set(filled_category_df[category_column].tolist() + [self.fillna_value])
                        )
                    )
                )
            }
            self.category2idx_dict[category_column] = category2idx
        self.is_fitted = True

    def transform(self, category_df: pd.DataFrame) -> pd.DataFrame:
        """Transform Category Encoder.

        Args:
            category_df: category dataframe

        Returns: encoded category dataframe

        """
        category_features = []
        filled_category_df = category_df.astype(str).fillna(self.fillna_value)
        for category_column in category_df.columns:
            map_dict = deepcopy(self.category2idx_dict[category_column])
            # unknown values are mapped to self.fillna_value
            unknown_values = set(filled_category_df[category_column]) - set(map_dict.keys())
            unknown_map_dict = {v: map_dict[self.fillna_value] for v in unknown_values}
            map_dict.update(unknown_map_dict)

            category_feature = filled_category_df[category_column].map(map_dict).astype(int)
            category_features.append(category_feature)
        encoded_category_features_df = pd.concat(category_features, axis=1)
        return encoded_category_features_df
