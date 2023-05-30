from typing import Any, Optional
import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT, EPOCH_OUTPUT
from pytorch_lightning import LightningModule
from pl_mltc.data.vocab import Vocab
from torch.nn import functional as F

from torchmetrics import F1Score

from pl_mltc.modules import Embedder
from torch import nn


class Word2VecClassifier(LightningModule):
    def __init__(self, hp, vocab: Vocab) -> None:
        super().__init__()
        self.save_hyperparameters(hp, ignore=("vocab", ))
        self.vocab = vocab
        self.embedder = Embedder(
            filename=hp.embedder.filename,
            n_dim=hp.embedder.n_dim,
            vocab=vocab
        )
        self.linear = nn.Linear(hp.embedder.n_dim, hp.n_labels)
        
        self.val_f1 = F1Score(num_classes=hp.n_labels, threshold=0.0)
    
    def forward(self, batch):
        texts = batch.texts
        labels = batch.labels
        # shape: (batch_size, max_seq_len, n_dim)
        word_embeddings = self.embedder(texts)
        # shape: (batch_size, n_dim)
        sentence_embeddings = (
            word_embeddings * texts.unsqueeze(dim=-1)
        ).sum(dim=1)
        sentence_embeddings = sentence_embeddings / (
            texts != self.vocab.pad_index
        ).sum(dim=1).unsqueeze(dim=-1)
        scores = self.linear(sentence_embeddings)
        if self.trainer.training:
            loss = F.binary_cross_entropy_with_logits(scores, labels.float())
        else:
            loss = None
        return {
            "loss": loss,
            "scores": scores.detach(),
            "labels": labels.detach(),
        }
    
    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        return self(batch)
    
    def configure_optimizers(self) -> Any:
        return torch.optim.Adam(
            self.parameters(), self.hparams.optimization.lr
        )

    def on_validation_epoch_start(self) -> None:
        self.val_f1.reset()
        
    def validation_step_end(self, output: STEP_OUTPUT) -> Optional[STEP_OUTPUT]:
        self.val_f1.update(preds=output["scores"], target=output["labels"])
        self.log("val_f1", self.val_f1.compute(), prog_bar=True)
        return output
