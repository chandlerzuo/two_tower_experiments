from data import FeatureConfig
from dataclasses import dataclass
from typing import Dict, List, Tuple
from data import (
    FloatFeatureConfig,
    IdListFeatureConfig,
)
import torch


class SparseNNSingleTower(torch.nn.Module):
    def __init__(
        self,
        id_list_feature_configs: List[IdListFeatureConfig],
        float_feature_configs: List[FloatFeatureConfig],
        sparse_proj_dims: List[int],
        float_proj_dims: List[int],
        overarch_proj_dims: List[int],
        output_dim: int,
    ) -> Tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module]:
        super().__init__()

        self.embedding_tables: Dict[str, torch.nn.Module] = torch.nn.ModuleDict(
            {
                str(cfg.fid): torch.nn.EmbeddingBag(
                    num_embeddings=cfg.num_embeddings,
                    embedding_dim=cfg.embedding_dim,
                    mode=cfg.pooling_type,
                )
                for cfg in id_list_feature_configs
            }
        )

        sparse_proj_list = []
        prev_dim = sum(cfg.embedding_dim for cfg in id_list_feature_configs)
        for hidden_dim in sparse_proj_dims:
            sparse_proj_list.append(torch.nn.Linear(prev_dim, hidden_dim))
            sparse_proj_list.append(torch.nn.ReLU())
            prev_dim = hidden_dim
        self.sparse_proj: torch.nn.Module = torch.nn.Sequential(*sparse_proj_list)
        sparse_proj_dim: int = prev_dim

        float_proj_list = []
        prev_dim = len(float_feature_configs)
        for hidden_dim in float_proj_dims:
            float_proj_list.append(torch.nn.Linear(prev_dim, hidden_dim))
            float_proj_list.append(torch.nn.ReLU())
            prev_dim = hidden_dim
        self.float_proj: torch.nn.Module = torch.nn.Sequential(*float_proj_list)
        float_proj_dim: int = prev_dim

        overarch_list = []
        prev_dim = sparse_proj_dim + float_proj_dim
        for hidden_dim in overarch_proj_dims + [output_dim]:
            overarch_list.append(torch.nn.Linear(prev_dim, hidden_dim))
            overarch_list.append(torch.nn.ReLU())
            prev_dim = hidden_dim
        self.overarch: torch.nn.Module = torch.nn.Sequential(*overarch_list)

    def forward(
        self,
        id_list_features: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        float_features: torch.Tensor,
    ) -> torch.Tensor:
        sparse_embs = []
        for fid, embedding_table in self.embedding_tables.items():
            values, offsets = id_list_features[fid]
            emb = embedding_table(values, offsets)
            sparse_embs.append(emb)
        sparse_embs = torch.cat(sparse_embs, dim=1) # [B, sum(embedding_dim)]

        return self.overarch(
            torch.cat(
                [
                    self.sparse_proj(sparse_embs),
                    self.float_proj(float_features)
                ],
                dim=1
            )
        )


class SparseNNTwoTower(torch.nn.Module):
    def __init__(
        self,
        feature_config: FeatureConfig,
        user_sparse_proj_dims: List[int],
        user_float_proj_dims: List[int],
        user_overarch_proj_dims: List[int],
        item_sparse_proj_dims: List[int],
        item_float_proj_dims: List[int],
        item_overarch_proj_dims: List[int],
        output_dim: int,
    ):
        super().__init__()
        self.feature_configs: List[FeatureConfig] = feature_config

        self.user_tower: torch.nn.Module = SparseNNSingleTower(
            id_list_feature_configs=feature_config.user_id_list_features,
            float_feature_configs=feature_config.user_float_features,
            sparse_proj_dims=user_sparse_proj_dims,
            float_proj_dims=user_float_proj_dims,
            overarch_proj_dims=user_overarch_proj_dims,
            output_dim=output_dim,
        )
        self.item_tower:  torch.nn.Module = SparseNNSingleTower(
            id_list_feature_configs=feature_config.item_id_list_features,
            float_feature_configs=feature_config.item_float_features,
            sparse_proj_dims=item_sparse_proj_dims,
            float_proj_dims=item_float_proj_dims,
            overarch_proj_dims=item_overarch_proj_dims,
            output_dim=output_dim,
        )

    def forward(
        self,
        user_id_list_features: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        user_float_features: torch.Tensor,
        item_id_list_features: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        item_float_features: torch.Tensor,
    ) -> torch.Tensor:
        user_embeddings = self.user_tower(
            id_list_features=user_id_list_features,
            float_features=user_float_features,
        ) # [B, output_dim]
        item_embeddings = self.item_tower(
            id_list_features=item_id_list_features,
            float_features=item_float_features,
        ) # [B, output_dim]

        return torch.matmul(user_embeddings, item_embeddings.transpose(0, 1))
