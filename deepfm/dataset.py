import pathlib
import re
from typing import Literal, Optional
import math

import pandas as pd
import numpy as np
import numpy.typing as npt
from torch.utils.data import Dataset
from utils import category_encode
from sklearn.preprocessing import OrdinalEncoder


class CriteoAdDataset(Dataset):
    def __init__(
        self,
        data_dir: pathlib.Path,
        type: Literal["train", "valid", "test"] = "train",
        nums: int = 100000,
        chunk_size: Optional[int] = None,
        category_encoder: Optional[OrdinalEncoder] = None,
    ) -> None:
        assert data_dir.exists()

        self.data_dir = data_dir
        data_path_list = sorted(list(data_dir.glob(f"{type}_*.csv")))
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
        if category_encoder is None:
            self.category_encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=
            )
            self.category_encoder.fit(self.category_features)
        # FIXME: In test, there are some categories that are not in train/valid. So we need to handle unknown categories.
        self.category2idx_dict, self.category_features = category_encode(self.category_features)
        self.category_cardinalities = {
            key: len(value) for key, value in self.category2idx_dict.items()
        }

    def __getitem__(self, index) -> tuple[int, npt.NDArray[np.float64], npt.NDArray[np.int64]]:
        label = int(self.labels[index].item())
        count_features = self.count_features.iloc[index].to_numpy(np.float32)
        category_features = self.category_features.iloc[index].to_numpy(np.int64)

        return label, count_features, category_features

    def __len__(self) -> int:
        return len(self.labels)

# TODO: category encoder sklearn-like
class CategoryEncoder()