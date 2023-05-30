from typing import Any, Optional

import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn
from torch.nn import functional as F
from torchmetrics.classification import MultilabelF1Score

from pl_mltc.data import Vocab
from pl_mltc.modules import Embedder


class CnnClassifier(LightningModule):
    def __init__(self, hp, vocab: Vocab) -> None:
        super().__init__()
        self.save_hyperparameters(hp, ignore=("vocab", ))
        self.vocab = vocab
        self.embedder = Embedder(
            filename=hp.embedder.filename,
            n_dim=hp.embedder.n_dim,
            vocab=vocab,
        )
        self.cnn = nn.ModuleList([
            nn.Conv1d(in_channels=hp.embedder.n_dim, out_channels=hp.cnn.hidden_size, kernel_size=kernel_size)
            for kernel_size in hp.cnn.kernel_size
        ])
        self.linear = nn.Linear(hp.cnn.hidden_size, hp.n_labels)

        self.train_f1 = MultilabelF1Score(num_labels=hp.n_labels, threshold=0.5, average="micro", validate_args=False)
        self.val_f1 = MultilabelF1Score(num_labels=hp.n_labels, threshold=0.5, average="micro", validate_args=False)

    def forward(self, batch):
        texts = batch.texts
        labels = batch.labels
        # shape: (batch_size, max_seq_len, n_dim)
        word_embeddings = self.embedder(texts)
        # shape: (batch_size, hidden_size, max_seq_len - kernel_size + 1)
        hidden_states = []
        for cnn in self.cnn:
            hidden_states.append(cnn(word_embeddings.transpose(1, 2)))
        # (batch_size, hidden_size, sum(max_seq_len - kernel_size + 1 for kernel_size in kernel_sizes))
        hidden_state = torch.cat(hidden_states, dim=-1)
        # (batch_size, hidden_size)
        sentence_embedding = torch.max(hidden_state, dim=-1).values
        scores = self.linear(sentence_embedding)
        if self.trainer.training:
            loss = F.binary_cross_entropy_with_logits(scores, labels.float())
        else:
            loss = None
        return {
            "loss": loss,
            "scores": F.sigmoid(scores.detach()) > 0.5,
            "labels": labels.detach(),
        }

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(
            self.parameters(), self.hparams.optimization.lr,
        )
        return optimizer
        # scheduler = ReduceLROnPlateau(
        #     optimizer, mode="min",
        #     factor=0.8,
        #     patience=3,
        #     min_lr=5e-5,
        #     verbose=True,
        # )
        # return {
        #     "optimizer": optimizer,
        #     "lr_scheduler": {
        #         "scheduler": scheduler,
        #         "monitor": "train_f1_step",
        #         "frequency": 20,
        #         "interval": "step",
        #     }
        # }

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
        self.log("val_f1", self.val_f1.compute(), prog_bar=True, on_step=True)
        return output
