from data import collate_fn
from torch.utils.data import DataLoader, Dataset
import logging
import torch


def train(
    model: torch.nn.Module,
    dataset: Dataset,
    batch_size: int = 32,
    num_epoch: int = 1,
    verbose_log_every_n: int = 20,
):
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
    data_loader = DataLoader(
        dataset, batch_size = batch_size, collate_fn = collate_fn,
    )
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    for i_epoch in range(num_epoch):
        n_batches = 0
        for (
            user_id_list_features,
            user_float_features,
            item_id_list_features,
            item_float_features,
            weights,
        ) in data_loader:
            logits = model(
                user_id_list_features,
                user_float_features,
                item_id_list_features,
                item_float_features,
            ) # [B, B]
            labels = torch.arange(logits.shape[1], dtype=torch.long)
            loss = loss_fn(logits, labels) # [B]
            weighted_sum_loss = (loss * weights).sum()
            optimizer.zero_grad()
            weighted_sum_loss.backward()
            optimizer.step()
            n_batches += 1
            if verbose_log_every_n > 0 and n_batches % verbose_log_every_n == 0:
                logging.info(f"epoch {i_epoch}, batch {n_batches}, loss: {weighted_sum_loss.item()}")


def test(
    model: torch.nn.Module,
    dataset: Dataset,
    batch_size: int = 32,
    verbose_log_every_n: int = 20,
) -> torch.Tensor:
    model.eval()
    data_loader = DataLoader(
        dataset, batch_size = batch_size, collate_fn = collate_fn,
    )
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    n_batches = 0
    for (
        user_id_list_features,
        user_float_features,
        item_id_list_features,
        item_float_features,
        weights,
    ) in data_loader:
        logits = model(
            user_id_list_features,
            user_float_features,
            item_id_list_features,
            item_float_features,
        ) # [B, B]
        labels = torch.arange(logits.shape[1], dtype=torch.long)
        loss = loss_fn(logits, labels) # [B]
        weighted_sum_loss = (loss * weights).sum() / weights.sum()
        n_batches += 1
        if verbose_log_every_n > 0 and n_batches % verbose_log_every_n == 0:
            logging.info(f"batch {n_batches}, loss: {weighted_sum_loss.item()}")
