from collections import namedtuple
import json
from pathlib import Path
from typing import Dict, Optional, Union
from dataclasses import dataclass

from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
import torch
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
from pl_mltc.data.vocab import Vocab

from pl_mltc.utils.console import console
from pl_mltc.utils.env import N_CPUS

# can not use `dataclass` here
# because the `default_collate` from PyTorch just support
# tensors, numpy arrays, numbers, dicts or lists
# @dataclass
# class Instance:
#     texts: str
#     labels: LongTensor
Instance = namedtuple(
    "Instance",
    ("texts", "labels",)
)

BertInstance = namedtuple(
    "BertInstance",
    ("input_ids", "token_type_ids", "attention_mask", "labels", )
)


class _Dataset(Dataset):
    def __init__(
            self,
            filename: str,
            label2index: Dict[str, int],
            vocab: Vocab,
            max_seq_len: int = 512,
    ):
        super().__init__()
        self.label2index = label2index
        self.vocab = vocab
        self.max_seq_len = max_seq_len

        texts, labels = [], []
        with open(filename) as fh:
            for index, line in enumerate(fh):
                if index % 2 == 0:
                    texts.append(line.strip())
                else:
                    labels.append(line.strip())
        self._texts = texts
        self._labels = labels

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, index) -> Instance:
        return Instance(
            texts=self._gettexts(index),
            labels=self._getlabels(index),
        )

    def _gettexts(self, index: int) -> Tensor:
        text = self._texts[index]
        if isinstance(text, str):
            # convert it to LongTensor
            words = text.strip().split()
            words = [self.vocab.word2id(word) for word in words][:self.max_seq_len]
            words = words + [0] * (self.max_seq_len - len(words))
            text = self._texts[index] = torch.LongTensor(words)
        return text

    def _getlabels(self, index: int) -> Tensor:
        labels = self._labels[index]
        if isinstance(labels, str):
            # convert it to LongTensor
            result = [0] * len(self.label2index)
            for label in labels.split():
                if label not in self.label2index:
                    continue
                result[self.label2index[label]] = 1
            labels = self._labels[index] = torch.LongTensor(result)
        return labels


class _BertDataset(Dataset):
    def __init__(
            self,
            filename: str,
            label2index: Dict[str, int],
            model_name_or_path: str,
            max_seq_len: int = 512,
    ):
        super().__init__()
        from transformers import BertTokenizer
        self.label2index = label2index
        self.tokenizer = BertTokenizer.from_pretrained(model_name_or_path)
        self.max_seq_len = max_seq_len

        texts, labels = [], []
        with open(filename) as fh:
            for index, line in enumerate(fh):
                if index % 2 == 0:
                    texts.append(line.strip())
                else:
                    labels.append(line.strip())
        self._texts = texts
        self._labels = labels

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, index) -> Instance:
        texts = self._gettexts(index)  # shape: (1, max_seq_len)
        return Instance(
            texts=texts,
            labels=self._getlabels(index),
        )

    def _gettexts(self, index: int) -> Tensor:
        text = self._texts[index]
        if isinstance(text, str):
            # convert it to Dict[str, LongTensor],
            # which contains three keys:
            # 1. 'input_ids'
            # 2. 'token_type_ids'
            # 3. 'attention_mask'
            text = self._texts[index] = self.tokenizer(
                text,
                max_length=self.max_seq_len,
                padding='max_length',
                return_token_type_ids=True,
                truncation=True,
                return_tensors="pt",
            )
        return text

    def _getlabels(self, index: int) -> Tensor:
        labels = self._labels[index]
        if isinstance(labels, str):
            # convert it to LongTensor
            result = [0] * len(self.label2index)
            for label in labels.split():
                if label not in self.label2index:
                    continue
                result[self.label2index[label]] = 1
            labels = self._labels[index] = torch.LongTensor(result)
        return labels


@dataclass
class _Datasets:
    train: _Dataset
    val: _Dataset
    test: _Dataset


class MultiLabelTextClassificationDataset(LightningDataModule):
    def __init__(
            self,
            data_dir: str,
            max_seq_len: int = 512,
            train_batch_size: int = 128,
            eval_batch_size: int = 128,
            model_name_or_path: Optional[str] = None,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.max_seq_len = max_seq_len
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

        self.model_name_or_path = model_name_or_path
        self.for_bertlike_models = model_name_or_path is not None
        self.datasets: Optional[_Datasets] = None
        self.label2index: Optional[Dict[str, int]] = None
        if not self.for_bertlike_models:
            # self.vocab: Optional[Vocab] = None
            self.vocab = Vocab(self.data_dir / 'vocab.txt')

    def setup(self, stage: Optional[str] = None) -> None:
        console.log(f"in [bold red]setup[/bold red](stage=[bold green]{stage}[/bold green])")
        with open(self.data_dir / 'label_to_index.json') as fh:
            self.label2index = json.load(fh)
        # self.vocab = Vocab(self.data_dir / 'vocab.txt')
        self.datasets = _Datasets(
            train=self._to_dataset(self.data_dir / 'train.txt'),
            val=self._to_dataset(self.data_dir / 'val.txt'),
            test=self._to_dataset(self.data_dir / 'test.txt'),
        )

    def _to_dataset(self, filename) -> Union[_BertDataset, _Dataset]:
        if self.for_bertlike_models:
            return _BertDataset(filename, self.label2index, self.model_name_or_path, self.max_seq_len)
        return _Dataset(filename, self.label2index, self.vocab, self.max_seq_len)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.datasets.train,
            batch_size=self.train_batch_size,
            num_workers=N_CPUS,
            shuffle=True,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.datasets.val,
            batch_size=self.eval_batch_size,
            num_workers=N_CPUS,
            shuffle=False,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.datasets.test,
            batch_size=self.eval_batch_size,
            num_workers=N_CPUS,
            shuffle=False,
        )
