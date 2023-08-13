import torch
import torch.nn as nn
from torch.fx import wrap


class DeepFM(nn.Module):
    def __init__(
        self,
        embedding_dims: int,
        category_cardinalities: dict[str, int],
        dense_embedding_in_features: int,
        dense_embedding_hidden_features: int,
        deep_layer_out_features: int,
    ) -> None:
        """DeepFM model.

        Args:
            embedding_dims: embedding dims for each feature.
            category_cardinalities: a dict of category feature name and its cardinality.
            dense_embedding_in_features: num of input feature for dense embedding.
            dense_embedding_hidden_features: num of hidden feature for dense embedding.
            deep_layer_out_features: num of output feature for deep component.
        """
        super().__init__()
        # embedding layer
        self.sparse_embedding = SparseEmbedding(category_cardinalities, embedding_dims)
        self.dense_embedding = DenseEmbedding(
            in_features=dense_embedding_in_features,
            hidden_features=dense_embedding_hidden_features,
            embedding_dims=embedding_dims,
        )
        # deep component
        self.deep_layer = DeepLayer(
            dense_module=nn.Sequential(
                nn.Linear(
                    in_features=(1 + len(category_cardinalities)) * embedding_dims,
                    out_features=deep_layer_out_features,
                ),
                nn.ReLU(),
            ),
            out_features=deep_layer_out_features,
        )
        # factorized machine component
        self.fm_layer = FactorizedMachineLayer()
        # predict logits from fm and deep components outputs.
        self.logits_layer = LogitsLayer(
            in_features=self.dense_embedding.embedding_dims + self.deep_layer.out_features + 1
        )

    def forward(
        self,
        count_features: torch.Tensor,
        category_features: torch.LongTensor,
        category_feature_names: list[str],
    ):
        """forward process.

        Args:
            count_features: count features, shape is (`batch_size` x `num_count_features`)
            category_features: category features, shape is (`batch_size` x `num_category_features`)
            category_feature_names: category feature names. The order of this should be same as category_features

        Returns:
            logits. Its shape is (`batch_size` x 1)
        """
        dense_embedded = self.dense_embedding(count_features)
        sparse_embedded = self.sparse_embedding(category_features, category_feature_names)

        embedded = [
            dense_embedded,
            *[sparse_embedded[feature_name] for feature_name in category_feature_names],
        ]

        deep_outputs = self.deep_layer(embedded)
        fm_outputs = self.fm_layer(embedded)

        logits = self.logits_layer(dense_embedded, deep_outputs, fm_outputs)

        return logits


class SparseEmbedding(nn.Module):
    def __init__(self, cardinalities: dict[str, int], embedding_dim: int):
        """
        Embedding layer for sparse features.
        Embedding each categorical feature into same `embedding_dims` dim.
        Although each categorical feature is jagged, this means each sparse feature has different cardinality, we embedding those into same `embedding_dims` dim.

        Args:
            cardinalities: a dict of sparse feature name and its cardinality.
            embedding_dim: embedding dimension.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.cardinalities = cardinalities

        self.category_embeddings = nn.ModuleDict(
            {
                feature_name: nn.EmbeddingBag(
                    num_embeddings=num_embeddings,
                    embedding_dim=embedding_dim,
                    mode="mean",
                )
                for feature_name, num_embeddings in cardinalities.items()
            }
        )

    def forward(
        self,
        inputs: torch.LongTensor,
        feature_names: list[str],
    ) -> dict[str, torch.Tensor]:
        """forward process.

        Args:
            inputs: a category tensor of size: (`batch_size` x `num_features`).
            feature_names: a list of category feature names. The order of this should be same as inputs

        Returns:
            embedded features dict. This length is `num_features` and its element shape is (`batch_size` x `embedding_dims`)
        """
        outputs = {}
        for idx, feature_name in enumerate(feature_names):
            input_ = inputs[:, idx].reshape(-1, 1)
            output = self.category_embeddings[feature_name](input_)
            outputs[feature_name] = output

        return outputs


class DenseEmbedding(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        embedding_dims: int,
    ) -> None:
        """Embedding layer for dense features.

        Args:
            in_features: num of dense feature
            hidden_features: hidden layer size for embedding
            embedding_dims: embedded dims
        """
        super().__init__()
        self.embedding_dims = embedding_dims
        self.model: nn.Module = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, embedding_dims),
            nn.ReLU(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """forward process.

        Args:
            features (torch.Tensor): dense features, shape is (`batch_size` x `num_features`)

        Returns:
            torch.Tensor: an output tensor. Its shape is (`batch_size` x `embedding_dims`)
        """
        return self.model(features)


@wrap
def _get_flatten_input(inputs: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat(
        [input.flatten(1) for input in inputs],
        dim=1,
    )


class DeepLayer(nn.Module):
    def __init__(self, dense_module: nn.Module, out_features: int) -> None:
        """Deep component of DeepFM. This layer learn high-order interactions between features.

        Args:
            dense_module: dense module for deep component.
            out_features: output features size.
        """
        super().__init__()
        self.dense_module = dense_module
        self.out_features = out_features

    def forward(self, embeddings: list[torch.Tensor]) -> torch.Tensor:
        """forward process

        Args:
            embeddings: a list of embedded features. The each element shape is (`batch_size` x `embedding_dims`).

        Returns:
            deep component outputs. Its shape is (`batch_size` x `out_features`)
        """
        inputs = _get_flatten_input(embeddings)
        outputs = self.dense_module(inputs)
        return outputs


class FactorizedMachineLayer(nn.Module):
    def __init__(self) -> None:
        """Factorized Machine component of DeepFM. This layer learn second-order interactions between features."""
        super().__init__()

    def forward(
        self,
        embeddings: list[torch.Tensor],
    ) -> torch.Tensor:
        """forward process

        Args:
            embeddings: a list of embedded features. The each element shape is (`batch_size` x `embedding_dims`).

        Returns:
            fm component outputs. Its shape is (`batch_size` x 1)
        """
        # flatten each embedding to be [B, N, D] -> [B, N*D], then cat them all on dim=1
        inputs = _get_flatten_input(embeddings)
        sum_of_inputs = torch.sum(inputs, dim=1, keepdim=True)
        sum_of_square = torch.sum(inputs * inputs, dim=1, keepdim=True)
        square_of_sum = sum_of_inputs * sum_of_inputs
        cross_term = square_of_sum - sum_of_square
        cross_term = torch.sum(cross_term, dim=1, keepdim=True) * 0.5  # [B, 1]
        return cross_term


class LogitsLayer(nn.Module):
    def __init__(self, in_features: int) -> None:
        """Output layer for DeepFM. This layer predict logits from deep and fm components outputs.

        Args:
            in_features: feature size of input. This is sum of each dims of dense embedding, deep and fm component outputs.
        """
        super().__init__()
        self.model = nn.Linear(in_features, 1)

    def forward(
        self, dense_embedded: torch.Tensor, deep_outputs: torch.Tensor, fm_outputs: torch.Tensor
    ) -> torch.Tensor:
        """forward process

        Args:
            dense_embedded: dense embedded features. Its shape is (`batch_size` x `embedding_dims`)
            deep_outputs: deep component outputs. Its shape is (`batch_size` x `out_features`)
            fm_outputs: fm component outputs. Its shape is (`batch_size` x 1)

        Returns:
            logits. Its shape is (`batch_size` x 1)
        """
        inputs = torch.cat([dense_embedded, deep_outputs, fm_outputs], dim=1)
        return self.model(inputs)
