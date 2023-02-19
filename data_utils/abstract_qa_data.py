import importlib
from collections import Counter
from functools import partial

import numpy as np
import pytorch_lightning as pl
import torch
import torchtext
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, IterableDataset
from datasets import load_dataset
from torchtext.vocab import vocab as vocab_builder
from tqdm import tqdm
import numpy as np
from typing import Union, List
import gzip
import csv
import random
import os


class QATextData(Dataset):
    def __init__(
        self, text_data_name: str, data_dir: str, partition="train", is_debug=False
    ):
        super().__init__()

        self.data_dir = data_dir
        self.debug = is_debug
        self.partition = partition
        self.train_feature_label = []
        self.data_path = os.path.join(self.data_dir, text_data_name, partition + ".tsv")
        self._load_data()

    def _load_data(self):

        print(f"loading data from {self.data_path}")

        if self.debug:
            self.data_path = self.data_path.replace("train", "dev")

        assert self.data_path.endswith(".tsv"), "data file has to be in tsv format."
        self.data = []
        with open(self.data_path, "r") as f:
            cnt = 0
            invalid_lines = 0
            for line in f:
                try:
                    question, answer = line.split("\t")
                except Exception:
                    invalid_lines += 1
                    continue
                self.data.append(
                    {
                        "id": "{}-{}".format(self.partition, cnt),
                        "question": question,
                        "answer": answer,
                    }
                )
            if invalid_lines > 0:
                print("# invalid lines: {}".format(invalid_lines))

        if self.debug:
            self.data = self.data[:40]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        return {
            "question": self.data[index]["question"],
            "answer": self.data[index]["answer"],
        }


def text_collate_fn(data):
    """
    data: is a list of tuples with (example, label, length)
          where 'example' is a tensor of arbitrary shape
          and label/length are scalars
    """
    questions = [item["question"] for item in data]
    answers = [item["answer"] for item in data]
    return {"questions": questions, "answers": answers}


class QATextDataModule(pl.LightningDataModule):
    """This data class aims to produce training and validation dataloader via function
    train_dataloader() and val_dataloader().

    dataloader is an iterator that produces a tuple of (a list of raw text, 1D tensor label list).

    """

    def __init__(
        self,
        text_data_name,
        data_dir,
        batch_size: int = 2,
        num_workers: int = 8,
        is_debug: bool = False,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.text_data_name = text_data_name
        self.data_dir = data_dir
        self.is_debug = is_debug

    def train_dataloader(self):

        self.train_data = QATextData(
            self.text_data_name, self.data_dir, "train", is_debug=self.is_debug
        )
        print(f"load train data loader with {len(self.train_data)} examples")
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=text_collate_fn,
        )

    def val_dataloader(self):
        self.val_data = QATextData(
            self.text_data_name, self.data_dir, "dev", is_debug=self.is_debug
        )
        print(f"load validation data loader with {len(self.val_data)} examples")
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=text_collate_fn,
        )


def _test_text_allnli_data_module():

    # print(data[0])

    data_module = QATextDataModule(
        "squad2", "/mnt/d/MLData/data/unifiedqa", is_debug=True
    )

    train_dataloader = data_module.train_dataloader()

    for batch in train_dataloader:
        print(batch)
        break


if __name__ == "__main__":
    _test_text_allnli_data_module()
