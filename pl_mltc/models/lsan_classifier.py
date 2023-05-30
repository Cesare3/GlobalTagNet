from typing import Any, Optional

import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn
from torch.nn import functional as F
from torchmetrics.classification import MultilabelF1Score

from pl_mltc.data.vocab import Vocab
from pl_mltc.modules import Embedder


class LSANClassifier(LightningModule):
    def __init__(self, hp, vocab: Vocab) -> None:
        super().__init__()
        self.save_hyperparameters(hp, ignore=("vocab",))
        self.vocab = vocab
        self.embedder = Embedder(
            filename=getattr(hp.embedder, "filename", None),
            n_dim=hp.embedder.n_dim,
            vocab=vocab
        )
        self.lstm = nn.LSTM(
            input_size=hp.embedder.n_dim,
            hidden_size=hp.lstm.n_hidden,
            num_layers=hp.lstm.n_layers,
            dropout=hp.lstm.dropout,
            batch_first=True,
            bidirectional=True,
        )
        text_dim = 2 * hp.lstm.n_hidden
        self.self_attn_scorer = nn.Sequential(
            nn.Linear(text_dim, hp.self_attn.n_hidden, bias=False),
            nn.Tanh(),
            nn.Linear(hp.self_attn.n_hidden, hp.n_labels, bias=False),
            nn.Softmax(dim=-1),
        )
        self.label_atte_scorer = nn.Linear(
            hp.lstm.n_hidden, hp.n_labels, bias=False,
        )
        self.adaptive_atte_scores = nn.Linear(
            text_dim, 1, bias=False
        )
        self.linear = nn.Sequential(
            nn.Dropout(hp.dropout),
            nn.Linear(text_dim, hp.output.n_hidden),
            nn.ReLU(),
            nn.Linear(hp.output.n_hidden, 1),
        )

        self.train_f1 = MultilabelF1Score(num_labels=hp.n_labels, threshold=0.5, average="micro", validate_args=False)
        self.val_f1 = MultilabelF1Score(num_labels=hp.n_labels, threshold=0.5, average="micro", validate_args=False)

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
        hidden_states, _ = self.lstm(word_embeddings)
        # 1: apply self-attention mechanism
        # (batch_size, max_seq_len, n_labels)
        self_attn_scores = self.self_attn_scorer(hidden_states)
        # (batch_size, n_labels, text_dim)
        self_attn = torch.einsum('bsl,bsd->bld', self_attn_scores, hidden_states)
        # 2: apply label-attention mechanism
        left_hidden_states = hidden_states[:, :, :self.hparams.lstm.n_hidden]
        # shape: (batch_size, max_seq_len, n_labels)
        left_label_atte_scores = self.label_atte_scorer(left_hidden_states)
        left_label_atte = torch.einsum('bsl,bsd->bld', left_label_atte_scores, left_hidden_states)
        right_hidden_states = hidden_states[:, :, self.hparams.lstm.n_hidden:]
        right_label_atte_scores = self.label_atte_scorer(right_hidden_states)
        right_label_atte = torch.einsum('bsl,bsd->bld', right_label_atte_scores, right_hidden_states)
        # (batch_size, n_labels, text_dim)
        label_atte = torch.cat((left_label_atte, right_label_atte), dim=-1)
        # 3. Adaptive Attention Fusion Strategy
        alpha = self.adaptive_atte_scores(self_attn)
        beta = 1.0 - alpha
        # shape: (batch_size, n_labels, text_dim)
        sentence_embeddings = alpha * self_attn + beta * label_atte
        scores = self.linear(sentence_embeddings).squeeze(dim=-1)
        if self.trainer.training:
            loss = F.binary_cross_entropy_with_logits(scores, labels.float())
        else:
            loss = None
        return {
            "loss": loss,
            "scores": torch.sigmoid(scores.detach()) > 0.5,
            "labels": labels.detach(),
        }

    def on_train_epoch_start(self) -> None:
        self.train_f1.reset()

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        return self(batch)

    def training_step_end(self, output: STEP_OUTPUT) -> STEP_OUTPUT:
        self.train_f1.update(preds=output["scores"], target=output["labels"])
        self.log("train_f1", self.train_f1.compute(), prog_bar=True, on_step=True, on_epoch=True)
        return output

    def on_validation_epoch_start(self) -> None:
        self.val_f1.reset()

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        return self(batch)

    def validation_step_end(self, output: STEP_OUTPUT) -> Optional[STEP_OUTPUT]:
        self.val_f1.update(preds=output["scores"], target=output["labels"])
        self.log("val_f1", self.val_f1.compute(), prog_bar=True, on_step=True, on_epoch=True)
        return output
