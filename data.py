from dataclasses import dataclass
from enum import Enum
from torch.utils.data import Dataset
from typing import Dict, List
import numpy as np
import pandas as pd
import torch


class DataSetSplit(Enum):
    TRAIN = "train"
    TEST = "test"

@dataclass
class IdListFeatureConfig:
    fid: int
    num_embeddings: int = 1001
    embedding_dim: int = 64
    max_len: int = 10
    pooling_type: str = "mean"


@dataclass
class FloatFeatureConfig:
    fid: int
    padding_val: float = 0


@dataclass
class FeatureConfig:
    user_id_list_features: List[IdListFeatureConfig]
    item_id_list_features: List[IdListFeatureConfig]
    user_float_features: List[FloatFeatureConfig]
    item_float_features: List[FloatFeatureConfig]

    @property
    def id_list_features(self):
        return self.user_id_list_features + self.item_id_list_features

    @property
    def float_features(self):
        return self.user_float_features + self.item_float_features


class SyntheticDataset(Dataset):
    MAX_SPARSE_VALUE = 1_000_000_000

    def __init__(
        self,
        dataset_split: DataSetSplit,
        feature_config: FeatureConfig,
        num_rows: int = 10_000,
        max_id_list_len: int = 10,
        feature_coverage_perc: float = 0.9
    ) -> None:
        self.feature_config = feature_config
        self.max_id_list_len = max_id_list_len
        self.num_rows = num_rows
        self.feature_coverage_perc = feature_coverage_perc
        self.__gen_synthetic_data(
            num_rows = num_rows,
            feature_coverage_perc = feature_coverage_perc
        )
        self.__compute_float_feature_meta()
        self.dataset_split = dataset_split

        self.id_to_maxlen = {
            cfg.fid: cfg.max_len
            for cfg in feature_config.id_list_features
        }
        self.id_to_embed_size = {
            cfg.fid: cfg.num_embeddings
            for cfg in feature_config.id_list_features
        }

    def __len__(self):
        return self.data.shape[0]

    def __process_id_list_features(
        self,
        id_list_features: Dict[int, List[int]],
        id_list_feature_configs: List[IdListFeatureConfig],
    ):
        return {
            cfg.fid: torch.tensor(
                [
                    x % self.id_to_embed_size[cfg.fid]
                    for x in id_list_features.get(
                        cfg.fid, []
                    )[:self.id_to_maxlen[cfg.fid]]
                ],
                dtype=torch.int
            )
            for cfg in id_list_feature_configs
        }

    def __process_float_features(
        self,
        float_features: Dict[int, float],
        float_feature_configs: List[FloatFeatureConfig],
    ):
        return torch.tensor(
            [
                (
                    max(
                        min(
                            float_features.get(cfg.fid, cfg.padding_val),
                            self.float_metadata[cfg.fid]['max'],
                        ),
                        self.float_metadata[cfg.fid]['min'],
                    ) - self.float_metadata[cfg.fid]['mean']
                ) / max(self.float_metadata[cfg.fid]['std'], 0.01)
                for cfg in float_feature_configs
            ],
            dtype=torch.float32
        )

    def __getitem__(self, idx: int):
        user_id_list_features = self.__process_id_list_features(
            id_list_features=self.data.loc[idx, "id_list_features"],
            id_list_feature_configs=self.feature_config.user_id_list_features,
        )
        item_id_list_features = self.__process_id_list_features(
            id_list_features=self.data.loc[idx, "id_list_features"],
            id_list_feature_configs=self.feature_config.item_id_list_features,
        )
        user_float_features = self.__process_float_features(
            float_features=self.data.loc[idx, "float_features"],
            float_feature_configs=self.feature_config.user_float_features,
        )
        item_float_features = self.__process_float_features(
            float_features=self.data.loc[idx, "float_features"],
            float_feature_configs=self.feature_config.item_float_features,
        )
        if self.dataset_split == DataSetSplit.TRAIN:
            weight = self.data.loc[idx, "weight"]
            return (
                user_id_list_features,
                user_float_features,
                item_id_list_features,
                item_float_features,
                torch.tensor(self.data.loc[idx, "weight"], dtype = torch.float)
            )
        else:
            return (
                user_id_list_features,
                user_float_features,
                item_id_list_features,
                item_float_features,
            )

    def __gen_synthetic_data(self, num_rows: int, feature_coverage_perc: float) -> pd.DataFrame:
        data = []
        for i in range(num_rows):
            row = {}
            row["id"] = i
            row["id_list_features"] = {
                cfg.fid: np.random.randint(
                    self.MAX_SPARSE_VALUE,
                    size=np.random.randint(cfg.max_len)
                )
                for cfg in self.feature_config.id_list_features
                if np.random.rand() < feature_coverage_perc
            }
            row["float_features"] = {
                cfg.fid: np.random.uniform(0, 1 / np.random.rand())
                for cfg in self.feature_config.float_features
                if np.random.rand() < feature_coverage_perc
            }
            row["weight"] = np.random.uniform(0,1)
            data.append(row)
        self.data = pd.DataFrame(data)

    def __compute_float_feature_meta(self):
        # compute the metadata for each float feature
        metadata = {}
        for cfg in self.feature_config.float_features:
            metadata[cfg.fid] = {}
            value_list = pd.Series([v.get(cfg.fid) for v in self.data['float_features']]).dropna()
            q01 = value_list.quantile(.01)
            q99 = value_list.quantile(.99)
            value_list[value_list < q01] = q01
            value_list[value_list > q99] = q99
            metadata[cfg.fid]['min'] = value_list.min()
            metadata[cfg.fid]['max'] = value_list.max()
            metadata[cfg.fid]['mean'] = value_list.mean()
            metadata[cfg.fid]['std'] = value_list.std()
        self.float_metadata = metadata


def collate_fn(batch):
    batch_size = len(batch)
    if len(batch[0]) == 5:
        (
            user_id_list_features,
            user_float_features,
            item_id_list_features,
            item_float_features,
            weights,
        ) = zip(*batch)
        weights = torch.stack(weights).view(-1)
    elif len(batch[0]) == 4:
        (
            user_id_list_features,
            user_float_features,
            item_id_list_features,
            item_float_features,
        ) = zip(*batch)
        weights = torch.ones(batch_size)
    else:
        raise ValueError("Invalid batch")
    user_id_list_sparse_format = {}
    item_id_list_sparse_format = {}
    user_id_list_fids = set([fid for f in user_id_list_features for fid in f.keys()])
    item_id_list_fids = set([fid for f in item_id_list_features for fid in f.keys()])
    for fid in user_id_list_fids:
        raw_data = [
            x.get(fid, torch.tensor([], dtype=torch.int))
            for x in user_id_list_features
        ]
        id_list_values = torch.cat(raw_data, dim=0)
        id_list_lengths = torch.tensor(
            [x.size(0) for x in raw_data],
            dtype=torch.int,
        )
        offsets = torch.cat(
            [
                torch.zeros(1, dtype=torch.int),
                torch.cumsum(id_list_lengths, dim=0)[:-1],
            ]
        )
        user_id_list_sparse_format[str(fid)] = (id_list_values, offsets)
    for fid in item_id_list_fids:
        raw_data = [
            x.get(fid, torch.tensor([], dtype=torch.int))
            for x in item_id_list_features
        ]
        id_list_values = torch.cat(raw_data, dim=0)
        id_list_lengths = torch.tensor(
            [x.size(0) for x in raw_data],
            dtype=torch.int,
        )
        offsets = torch.cat(
            [
                torch.zeros(1, dtype=torch.int),
                torch.cumsum(id_list_lengths, dim=0)[:-1],
            ]
        )
        item_id_list_sparse_format[str(fid)] = (id_list_values, offsets)
    return (
        user_id_list_sparse_format,
        torch.stack(user_float_features).view(batch_size, -1),
        item_id_list_sparse_format,
        torch.stack(item_float_features).view(batch_size, -1),
        weights,
    )
