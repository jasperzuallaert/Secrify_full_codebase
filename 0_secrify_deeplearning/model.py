# input managing
from __future__ import annotations

from math import floor
import torch
import numpy as np
from typing import Any
from torch import nn

aas = 'ACDEFGHIKLMNPQRSTVWY'
aa_to_idx = {aa: idx+1 for idx, aa in enumerate(aas)}
aa_to_idx['<pad>'] = 0

def onehot_encode(seq):
    tokens = [aa_to_idx[aa] for aa in seq]
    onehot_enc = np.zeros((len(tokens), 21))
    for i, token in enumerate(tokens):
        onehot_enc[i, token] = 1
    return onehot_enc

class SecrifyDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_locs
    ) -> None:
        print(f'--- Loading data from {dataset_locs} ---')
        self.data = []
        for dataset_loc in dataset_locs:
            self.data.extend(_read_records(dataset_loc))

    def get_targets(self):
        return [o['labels'] for o in self.data]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> tuple:
        item = self.data[index]
        seq = item["seq_data"]
        onehot_enc = onehot_encode(seq)
        labels = np.asarray([item["labels"]])
        return (
            onehot_enc,
            labels,
        )

    def collate_fn(self, batch) -> dict[str, Any]:
        (
            onehot_enc,
            labels,
        ) = tuple(zip(*batch))
        onehot_enc = torch.from_numpy(pad_sequences(onehot_enc, 0)).float()
        labels = torch.from_numpy(np.asarray(labels))
        output = {
            "onehot_enc": onehot_enc,
            "labels": labels,
        }
        return output

def _read_records(input_file):
    num_pos = 0
    data = []
    for i,line in enumerate(open(input_file, "r").readlines()[1:]):
        line = line.strip().split(",")
        if len(line) == 3:
            id_, seq, lab = line
        else:
            seq, lab = line
            id_ = floor(1000000+i)
        lab = int(float(lab))
        num_pos += lab
        data.append({'fragment_id': f'{id_}', 'seq_data': seq, 'labels': lab})
    print(f"--- Loaded {input_file} data ---")
    print(f"- Number of positive samples: {num_pos}")
    print(f"- Number of negative samples: {len(data)-num_pos}")
    print()
    return data

def pad_sequences(sequences, constant_value):
    batch_size = len(sequences)
    shape = [batch_size] + np.max([seq.shape for seq in sequences], 0).tolist()

    dtype = sequences[0].dtype
    if isinstance(sequences[0], np.ndarray):
        array = np.full(shape, constant_value, dtype=dtype)
    elif isinstance(sequences[0], torch.Tensor):
        array = torch.full(shape, constant_value, dtype=dtype)

    for arr, seq in zip(array, sequences):
        arrslice = tuple(slice(dim) for dim in seq.shape)
        arr[arrslice] = seq

    return array


# create network
class ConvNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.l1 = nn.ModuleList()
        self.l1.append(nn.Conv1d(21, 260, 7, padding=3))
        self.l1.append(nn.ReLU())
        self.l1.append(nn.Dropout(0.4))
        self.l1.append(nn.MaxPool1d(3))
        self.l1.append(nn.Conv1d(260, 260, 7, padding=3))
        self.l1.append(nn.ReLU())
        self.l1.append(nn.Dropout(0.4))
        self.l1.append(nn.MaxPool1d(3))
        self.l1.append(nn.Conv1d(260, 260, 7, padding=3))
        self.l1.append(nn.ReLU())
        self.l1.append(nn.Dropout(0.4))
        self.l1.append(nn.MaxPool1d(3))
        self.global_pooling = nn.AdaptiveMaxPool1d(1)
        self.l2 = nn.ModuleList()
        self.l2.append(nn.Linear(260, 300)) # +1 for encoded length
        self.l2.append(nn.ReLU())
        self.l2.append(nn.Dropout(0.4))
        self.l2.append(nn.Linear(300, 1))

    def forward(self, x) -> torch.tensor:
        mask = x.sum(dim=-1) != 0
        lengths = mask.sum(dim=-1).to(x.device)
        lengths_after_maxpooling = lengths // 3 // 3 // 3
        max_len = lengths_after_maxpooling.max()
        mask_after_maxpooling = torch.arange(max_len).to(x.device).expand(len(lengths_after_maxpooling), max_len) < lengths_after_maxpooling.unsqueeze(1)
        mask_after_maxpooling = mask_after_maxpooling.unsqueeze(1).float()

        x = x.permute(0, 2, 1)
        for layer in self.l1:
            x = layer(x)
        x = x * mask_after_maxpooling

        x = self.global_pooling(x)
        x = x.squeeze(-1)
        for layer in self.l2:
            x = layer(x)
        return x

# pytorch lightning module 
import torch
import pytorch_lightning as pl
from torch import nn
from torch.optim import AdamW
from torchmetrics import Metric, AUROC, AveragePrecision, Accuracy
from typing import Any
from torch.optim.lr_scheduler import LinearLR

# PyTorch-Lightning Module class; takes care of training, batch organization, metrics, logging, evaluation
class LightningModule(pl.LightningModule):
    def __init__(self, dataset_name='', steps_per_training_epoch=0) -> None:
        super().__init__()
        self.model = ConvNet()
        self.dataset_name = dataset_name
        warm_up_epochs = 1.5
        self.warm_up_steps = int(warm_up_epochs * steps_per_training_epoch)

        # set loss functions for training/validation/test
        self.train_cross_entropy_loss = torch.nn.BCEWithLogitsLoss()
        self.valid_cross_entropy_loss = torch.nn.BCEWithLogitsLoss()
        self.test_cross_entropy_loss = torch.nn.BCEWithLogitsLoss()

        # initialize metrics for training/validation/test
        self.metric_names, self.train_metrics = self._init_metrics()
        _, self.valid_metrics = self._init_metrics()
        _, self.test_metrics = self._init_metrics()

        self.calibration_max_value = 0.0

    # calibration will make sure the model outputs are between 0 and 1, with 1 being the highest predicted probability during training
    def predict_calibrated(self, x): 
        y_hat = self.model(x)
        y_hat_sigm = torch.sigmoid(y_hat)
        y_hat_calibrated = torch.min(y_hat_sigm / self.calibration_max_value, torch.ones_like(y_hat_sigm))
        return y_hat_calibrated

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=2e-4)
        scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=self.warm_up_steps
        )
        scheduler = {
            'scheduler': scheduler,
            'name': 'actual_learning_rate',
            'interval': 'step',
            'frequency': 1
        }
        return [optimizer], [scheduler]

    def _init_metrics(self) -> tuple[list[str], nn.ModuleList]:
        names = ["AUPRC", "AUROC", "accuracy"]
        return (
            names,
            nn.ModuleList(
                [
                    AveragePrecision(task='binary'),#,compute_on_step=False),
                    AUROC(task='binary'),#,compute_on_step=False)
                    Accuracy(task='binary'),#,compute_on_step=False)
                ]
            ),
        )

    def training_step(self, batch):
        X, y = batch['onehot_enc'], batch['labels']
        y_hat = self.model(X)
        y_hat_sigm = torch.sigmoid(y_hat)

        loss = self.train_cross_entropy_loss(y_hat, y.float())
        self.log("training_loss", loss, on_step=False, on_epoch=True)
        for metric in self.train_metrics:
            metric(y_hat_sigm, y)
        return loss

    def on_save_checkpoint(self, checkpoint) -> None:
        checkpoint["calibration_max_value"] = self.calibration_max_value
        

    def on_load_checkpoint(self, checkpoint) -> None:
        self.calibration_max_value = checkpoint["calibration_max_value"]

    def validation_step(self, batch):
        X, y = batch['onehot_enc'], batch['labels']
        y_hat = self.model(X)
        y_hat_sigm = torch.sigmoid(y_hat)
        # calibration, only at validation set
        max_prediction = torch.max(y_hat_sigm)
        if max_prediction > self.calibration_max_value:
            self.calibration_max_value = max_prediction
        loss = self.valid_cross_entropy_loss(y_hat, y.float())
        self.log("validation_loss", loss, on_step=False, on_epoch=True)
        for metric in self.valid_metrics:
            metric(y_hat_sigm, y)
        return loss

    def on_training_epoch_end(self):
        for metric_name, metric in zip(self.metric_names, self.train_metrics):
            result = metric.compute()
            self.log(
                f"train_{metric_name}",
                result[0] if isinstance(result, tuple) else result,
            )
            metric.reset()

    def on_validation_epoch_start(self):
        self.calibration_max_value = 0.0 # reset for next checkpoint

    def on_validation_epoch_end(self):
        for metric_name, metric in zip(self.metric_names, self.valid_metrics):
            result = metric.compute()
            self.log(
                f"valid_{metric_name}",
                result[0] if isinstance(result, tuple) else result,
            )
            metric.reset()

    def test_step(self, batch):
        X, y = batch['onehot_enc'], batch['labels']
        y_hat = self.model(X)
        y_hat_sigm = torch.sigmoid(y_hat)
        loss = self.test_cross_entropy_loss(y_hat, y.float())
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        for metric in self.test_metrics:
            metric(y_hat_sigm, y)
        return loss
        

    def on_test_epoch_end(self):
        results_file = open('all_results.csv', 'a')
        for metric_name, metric in zip(self.metric_names, self.test_metrics):
            result = metric.compute()
            self.log(
                f"test_{metric_name}",
                result[0] if isinstance(result, tuple) else result,
            )
            print(f"{self.dataset_name}_maxpool2,test_{metric_name},{result[0] if isinstance(result, tuple) else result:.3f}", file=results_file)
            metric.reset()