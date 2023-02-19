from argparse import ArgumentParser

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers.wandb import WandbLogger

from models.finetune_model import Seq2SeqFinetune
from data_utils.abstract_qa_data import QATextDataModule


def main(args):

    seed_everything(args.seed)

    data_module = QATextDataModule(
        args.dataset_name,
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    train_dataloader = data_module.train_dataloader()
    val_dataloader = data_module.val_dataloader()

    config = {}
    config["lr"] = args.lr
    config["question_truncate"] = args.question_truncate
    config["answer_truncate"] = args.answer_truncate

    model = Seq2SeqFinetune(args.pretrained_model_name)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss_epoch",
        save_top_k=2,
        mode="max",
        every_n_train_steps=args.val_step_interval,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # stsb_eval_callback = StsbEvaluationCallback(
    #     "/mnt/d/MLData/data/stsbenchmark.tsv", n_step=args.val_step_interval
    # )

    wandb_logger = WandbLogger(
        project=args.project_name,  # group runs in "MNIST" project
        log_model="all" if args.log_model else False,
        save_dir=args.default_root_dir,
        group=args.dataset_name,
        tags=[args.pretrained_model_name, args.dataset_name, "contrastive"],
    )

    trainer = Trainer(
        accelerator="gpu",
        devices=args.gpus,
        max_epochs=args.max_epochs,
        precision=args.precision,
        logger=wandb_logger,
        callbacks=[
            TQDMProgressBar(refresh_rate=args.progress_bar_refresh_rate),
            checkpoint_callback,
            lr_monitor,
            #   stsb_eval_callback,
        ],
        deterministic=True,
    )

    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )


def parse_arguments():

    parser = ArgumentParser()

    # trainer specific arguments

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)

    parser.add_argument("--gpus", type=int, required=True)
    parser.add_argument("--precision", type=int, default=16)
    parser.add_argument("--max_epochs", type=int, default=5)
    parser.add_argument("--val_step_interval", type=int, default=500)
    parser.add_argument("--progress_bar_refresh_rate", type=int, default=5)

    parser.add_argument("--project_name", type=str, required=True)
    parser.add_argument("--default_root_dir", type=str, required=True)
    parser.add_argument("--log_model", action="store_true")

    # model specific arguments
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--question_truncate", type=int, default=512)
    parser.add_argument("--answer_truncate", type=int, default=128)
    parser.add_argument("--pretrained_model_name", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=16)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()

    main(args)
