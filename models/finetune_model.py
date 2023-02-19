import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict
import pytorch_lightning as pl
from torchmetrics import Accuracy
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
from models.utils import batch_to_device


class Seq2SeqFinetune(pl.LightningModule):
    def __init__(
        self,
        pretrained_model_name: str,
        config: Dict = {},
    ):
        super().__init__()
        self.save_hyperparameters()
        self.pretrained_model_name = pretrained_model_name
        self.model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        self.config = config

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)

    def model_forward(self, batch):

        input_sentences, target_sentences = batch["questions"], batch["answers"]

        input_tokenized = self.tokenizer(
            list(input_sentences),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.get("question_truncate", 512),
        )

        target_tokenized = self.tokenizer(
            list(target_sentences),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.get("answer_truncate", 80),
        )
        input_tokenized = batch_to_device(input_tokenized, target_device=self.device)
        target_tokenized = batch_to_device(target_tokenized, target_device=self.device)

        # Prepare input and output
        target_length = target_tokenized["input_ids"].shape[1]
        decoder_input_ids = target_tokenized["input_ids"].clone()[
            :, : target_length - 1
        ]
        decoder_attention_masks = target_tokenized["attention_mask"][
            :, : target_length - 1
        ]
        label_ids = target_tokenized["input_ids"][:, 1:]

        outputs = self.model(
            input_ids=input_tokenized["input_ids"],
            attention_mask=input_tokenized["attention_mask"],
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_masks,
            return_dict=True,
        )
        # Calculate loss
        lm_logits = outputs.logits
        loss = self.loss_fn(
            lm_logits.view(-1, lm_logits.shape[-1]), label_ids.reshape(-1)
        )
        return loss

    def training_step(self, batch, batch_idx=0):

        loss = self.model_forward(batch)
        self.log(
            "train_loss",
            loss,
            batch_size=len(batch["questions"]),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, batch, batch_idx=0):

        loss = self.model_forward(batch)
        self.log(
            "val_loss",
            loss,
            batch_size=len(batch["questions"]),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def on_save_checkpoint(self, checkpoint):
        checkpoint["model_state_dict"] = self.model.state_dict()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), self.config.get("lr", 1e-3))
        lr_warmup_steps = self.config.get("lr_warm_up_steps", 10000)

        def lr_foo(step):
            lr_scale = min((step + 1) ** (-0.5), step / lr_warmup_steps ** (1.5))

            return lr_scale

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_foo)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


def _test_model():

    from data_utils.abstract_qa_data import QATextDataModule

    model = Seq2SeqFinetune("facebook/bart-large")
    model.to("cuda")
    dataloader = QATextDataModule(
        "squad2", "/mnt/d/MLData/data/unifiedqa", is_debug=True
    ).train_dataloader()

    for batch in dataloader:
        model.training_step(batch=batch, batch_idx=0)
        break


if __name__ == "__main__":
    _test_model()
