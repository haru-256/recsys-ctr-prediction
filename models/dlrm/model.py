import torch
import torch.nn as nn


class DLRM(nn.Module):
    def __init__(
        self,
        embedding_dims: int,
        category_cardinalities: dict[str, int],
        bottom_mlp_in_features: int,
        bottom_mlp_hidden_layer_size_list: list[int],
        top_mlp_hidden_layer_size_list: list[int],
        use_interaction_itself: bool,
    ) -> None:
        """DLRM(Deep Learning Recommendation Model).

        Args:
            embedding_dims: embedding dims for each feature.
            category_cardinalities: a dict of category feature name and its cardinality.
            bottom_mlp_in_features: num of input feature for bottom mlp.
            bottom_mlp_hidden_layer_size_list: list of hidden layer size for bottom mlp.
            top_mlp_in_features: num of input feature for top mlp.
            top_mlp_hidden_layer_size_list: list of hidden layer size for top mlp.
            use_interaction_itself: whether use feature interaction itself or not.
        """
        super().__init__()
        # embedding layer
        self.sparse_embedding = SparseEmbedding(category_cardinalities, embedding_dims)
        self.bottom_mlp = BottomMLP(
            in_features=bottom_mlp_in_features,
            hidden_layer_size_list=bottom_mlp_hidden_layer_size_list,
            embedding_dims=embedding_dims,
        )
        # feature interaction component
        self.interaction_layer = InteractionLayer(
            use_interaction_itself=use_interaction_itself,
            num_sparse_feature=len(category_cardinalities),
        )
        # predict logits from interaction component and bottom mlp outputs.
        self.top_mlp = TopMLP(
            in_features=embedding_dims + self.interaction_layer.output_dim_,
            hidden_layer_size_list=top_mlp_hidden_layer_size_list,
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
        dense_embedded = self.bottom_mlp(count_features)
        sparse_embedded = self.sparse_embedding(category_features, category_feature_names)
        embedded = [
            dense_embedded,
            *[sparse_embedded[feature_name] for feature_name in category_feature_names],
        ]

        interaction_outputs = self.interaction_layer(embedded)

        logits = self.top_mlp(dense_embedded, interaction_outputs)

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


class BottomMLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_layer_size_list: list[int],
        embedding_dims: int,
    ) -> None:
        """BottomMLP, which is Embedding layer for dense features.

        Args:
            in_features: num of dense feature
            hidden_layer_size_list: hidden layer size for embedding
            embedding_dims: embedded dims
        """
        super().__init__()

        in_features_list = [in_features, *hidden_layer_size_list]

        if len(in_features_list) >= 2:
            in_out_features_list = [
                (in_features_list[idx], in_features_list[idx + 1])
                for idx in range(len(in_features_list) - 1)
            ]
        else:
            in_out_features_list = []
        in_out_features_list.append((in_features_list[-1], embedding_dims))
        self.model = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(in_features=in_features, out_features=out_features),
                    nn.ReLU(),
                )
                for in_features, out_features in in_out_features_list
            ]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """forward process.

        Args:
            features (torch.Tensor): dense features, shape is (`batch_size` x `num_features`)

        Returns:
            torch.Tensor: an output tensor. Its shape is (`batch_size` x `embedding_dims`)
        """
        return self.model(inputs)


class InteractionLayer(nn.Module):
    """Feature Interaction of DLRM. This layer consider second-order interactions between features."""

    def __init__(self, use_interaction_itself: bool, num_sparse_feature: int) -> None:
        """Feature Interaction of DLRM. This layer consider second-order interactions between features.

        Args:
            use_interaction_itself: whether use interaction itself or not. If this is True, interaction between feature itself is used.
            num_sparse_feature: num of sparse feature
        """
        super().__init__()
        self.use_interaction_itself = use_interaction_itself
        self.num_sparse_feature = num_sparse_feature
        self.triu_indices_ = torch.triu_indices(
            self.num_sparse_feature + 1,
            self.num_sparse_feature + 1,
            offset=0 if self.use_interaction_itself else 1,
        )
        self.output_dim_ = self.triu_indices_.shape[1]

    def forward(
        self,
        embeddings: list[torch.Tensor],
    ) -> torch.Tensor:
        """forward process

        Args:
            embeddings: a list of embedded features. The each element shape is (`batch_size` x `embedding_dims`).

        Returns:
            interaction component outputs. Its shape is (`batch_size` x (`feature_num` x `feature_num` / 2))
        """
        # shape: (`batch_size` x `feature_num` x `embedding_dims`)
        inputs = torch.stack(embeddings, dim=1)

        # dense/sparse + sparse/sparse interaction
        # shape: (`batch_size` x `feature_num` x `feature_num`)
        interactions = torch.bmm(inputs, torch.transpose(inputs, 1, 2))
        cross_term = interactions[:, self.triu_indices_[0], self.triu_indices_[1]]
        return cross_term


class TopMLP(nn.Module):
    def __init__(self, in_features: int, hidden_layer_size_list: list[int]) -> None:
        """TopMLP, which output logits from dense embedding and interaction.

        Args:
            in_features: feature size of input. This is sum of each dims of dense embedding, deep and fm component outputs.
            hidden_layer_size_list: hidden layer size for top mlp
        """
        super().__init__()

        blocks = []
        in_features_list = [in_features, *hidden_layer_size_list]
        if len(in_features_list) >= 2:
            for idx in range(len(in_features_list) - 1):
                block = nn.Sequential(
                    nn.Linear(
                        in_features=in_features_list[idx],
                        out_features=in_features_list[idx + 1],
                    ),
                    nn.ReLU(),
                )
                blocks.append(block)

        self.model = nn.Sequential(
            *blocks, nn.Linear(in_features=in_features_list[-1], out_features=1)
        )

    def forward(
        self, dense_embedded: torch.Tensor, interaction_outputs: torch.Tensor
    ) -> torch.Tensor:
        """forward process

        Args:
            dense_embedded: dense embedded features. Its shape is (`batch_size` x `embedding_dims`)
            interaction_outputs: interaction component outputs. Its shape is (`batch_size` x (`feature_num` x `feature_num / 2))

        Returns:
            logits.
        """
        inputs = torch.cat([dense_embedded, interaction_outputs], dim=1)
        return self.model(inputs)
