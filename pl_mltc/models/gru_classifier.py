from typing import Any, Optional

import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn
from torch.nn import functional as F
from torchmetrics.classification import MultilabelF1Score

from pl_mltc.data import Vocab
from pl_mltc.modules import Embedder


class GruClassifier(LightningModule):
    def __init__(self, hp, vocab: Vocab) -> None:
        super().__init__()
        self.save_hyperparameters(hp, ignore=("vocab", ))
        self.vocab = vocab
        self.embedder = Embedder(
            filename=hp.embedder.filename,
            n_dim=hp.embedder.n_dim,
            vocab=vocab
        )
        self.gru = nn.GRU(
            input_size=hp.embedder.n_dim,
            hidden_size=hp.gru.n_hidden,
            num_layers=hp.gru.n_layers,
            dropout=hp.gru.dropout,
            batch_first=True,
            bidirectional=hp.gru.bidirectional,
        )
        self.linear = nn.Sequential(
            nn.Dropout(hp.dropout),
            nn.Linear(self.text_dim, hp.n_labels)
        )

        self.train_f1 = MultilabelF1Score(num_labels=hp.n_labels, average="micro", validate_args=False)
        self.val_f1 = MultilabelF1Score(num_labels=hp.n_labels, average="micro", validate_args=False)

    @property
    def text_dim(self):
        return (
            2 * self.hparams.gru.n_hidden
            if self.hparams.gru.bidirectional
            else self.hparams.gru.n_hidden
        )

    def configure_optimizers(self) -> Any:
        return torch.optim.Adam(
            self.parameters(), self.hparams.optimization.lr
        )

    def forward(self, batch):
        texts = batch.texts
        lengths = (texts != 0).sum(dim=-1)
        labels = batch.labels
        # shape: (batch_size, max_seq_len, n_dim)
        word_embeddings = self.embedder(texts)
        # (batch_size, max_seq_len, text_dim)
        hidden_states, _ = self.gru(word_embeddings)
        # shape: (batch_size, text_dim)
        sentence_embeddings = []
        for index, length in enumerate(lengths - 1):
            sentence_embeddings.append(hidden_states[index, length, :])
        sentence_embeddings = torch.stack(sentence_embeddings)
        scores = self.linear(sentence_embeddings)
        if self.trainer.training:
            loss = F.binary_cross_entropy_with_logits(scores, labels.float())
        else:
            loss = None
        return {
            "loss": loss,
            "scores": torch.sigmoid(scores.detach()) > 0.5,
            "labels": labels.detach(),
        }

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        return self(batch)

    def on_train_epoch_start(self) -> None:
        self.train_f1.reset()

    def training_step_end(self, output: STEP_OUTPUT) -> Optional[STEP_OUTPUT]:
        self.train_f1.update(preds=output["scores"], target=output["labels"])
        self.log("train_f1", self.train_f1.compute(), prog_bar=True, on_step=True)
        return output

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        return self(batch)

    def on_validation_epoch_start(self) -> None:
        self.val_f1.reset()

    def validation_step_end(self, output: STEP_OUTPUT) -> Optional[STEP_OUTPUT]:
        self.val_f1.update(preds=output["scores"], target=output["labels"])
        self.log("val_f1", self.val_f1.compute(), prog_bar=True)
        return output
