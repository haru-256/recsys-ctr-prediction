import pathlib
import re
from typing import Literal, Optional
import math
from copy import deepcopy

import pandas as pd
import numpy as np
import numpy.typing as npt
from torch.utils.data import Dataset


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


class CriteoAdDataset(Dataset):
    def __init__(
        self,
        data_dir: pathlib.Path,
        category_encoder: CategoryEncoder,
        type: Literal["train", "valid", "test"] = "train",
        nums: int = 100000,
        chunk_size: Optional[int] = None,
    ) -> None:
        assert data_dir.exists()

        self.data_dir = data_dir
        data_path_list = sorted(list(data_dir.glob(f"{type}_*.csv")))
        self.category_encoder = category_encoder
        if chunk_size is None:
            chunk_size = len(pd.read_csv(data_path_list[0]))
        self.chunk_size = chunk_size
        self.nums = nums

        df = pd.concat(
            map(
                pd.read_csv,
                [data_path_list[i] for i in range(math.ceil(self.nums / self.chunk_size))],
            )
        )
        df = df[: self.nums]
        self.labels = df["label"]
        self.count_feature_columns = sorted(
            list(filter(lambda c: re.match(r"count_feature_\d+", c) is not None, df.columns))
        )
        self.count_features = df[self.count_feature_columns].fillna(-1)
        self.category_feature_columns = sorted(
            list(filter(lambda c: re.match(r"category_feature_\d+", c) is not None, df.columns))
        )
        self.category_features = df[self.category_feature_columns].copy()
        if type == "train":
            self.category_encoder.fit(self.category_features)
        else:
            assert self.category_encoder.is_fitted
        self.category_features = self.category_encoder.transform(self.category_features)
        self.category_cardinalities = {
            key: len(value) for key, value in self.category_encoder.category2idx_dict.items()
        }

    def __getitem__(self, index) -> tuple[int, npt.NDArray[np.float64], npt.NDArray[np.int64]]:
        label = int(self.labels[index].item())
        count_features = self.count_features.iloc[index].to_numpy(np.float32)
        category_features = self.category_features.iloc[index].to_numpy(np.int64)

        return label, count_features, category_features

    def __len__(self) -> int:
        return len(self.labels)
