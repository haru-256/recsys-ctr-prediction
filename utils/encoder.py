from copy import deepcopy

import pandas as pd


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
