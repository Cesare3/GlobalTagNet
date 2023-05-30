from typing import Optional, Any

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn
from torchmetrics.classification import MultilabelF1Score
from transformers import BertModel
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions


class BertClassifier(LightningModule):
    def __init__(self, hp) -> None:
        super().__init__()
        self.save_hyperparameters(hp)
        # transformers
        self.bert = BertModel.from_pretrained(hp.bert.model_name_or_path)
        self.linear = nn.Linear(self.bert.config.hidden_size, hp.n_labels)
        self.dropout = nn.Dropout(hp.dropout)

        self.val_f1 = MultilabelF1Score(num_labels=hp.n_labels, average="micro", validate_args=False)
        self.train_f1 = MultilabelF1Score(num_labels=hp.n_labels, average="micro", validate_args=False)

    def forward(self, batch):
        # dict => three keys
        texts = batch.texts  # (batch_size, 1, max_seq_len)
        labels = batch.labels
        texts['input_ids'] = texts['input_ids'].squeeze(dim=1)
        texts['token_type_ids'] = texts['token_type_ids'].squeeze(dim=1)
        texts['attention_mask'] = texts['attention_mask'].squeeze(dim=1)
        output: BaseModelOutputWithPoolingAndCrossAttentions = self.bert(**texts)
        text_representation = output.pooler_output
        scores = self.linear(self.dropout(text_representation))

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

    def configure_optimizers(self) -> Any:
        return torch.optim.AdamW(
            self.parameters(), self.hparams.optimization.lr
        )

    def on_validation_epoch_start(self) -> None:
        self.val_f1.reset()

    def validation_step_end(self, output: STEP_OUTPUT) -> Optional[STEP_OUTPUT]:
        self.val_f1.update(preds=output["scores"], target=output["labels"])
        self.log("val_f1", self.val_f1.compute(), prog_bar=True, on_step=True)
        return output

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        return self(batch)
