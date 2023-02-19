from argparse import ArgumentParser

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers.wandb import WandbLogger

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from models.finetune_model import Seq2SeqFinetune
from data_utils.abstract_qa_data import QATextDataModule
from models.utils import batch_to_device
from evaluation import evaluate_squad2
import tqdm

# model_name = "allenai/unifiedqa-t5-small"  # you can specify the model size here
# tokenizer = T5Tokenizer.from_pretrained(model_name)
# model = T5ForConditionalGeneration.from_pretrained(model_name)


def run_model(model, tokenizer, input_string, device, **generator_args):
    input_ids = tokenizer.encode(input_string, return_tensors="pt")
    input_ids = input_ids.to(device)
    res = model.generate(input_ids, **generator_args)
    return tokenizer.batch_decode(res, skip_special_tokens=True)[0]


def main(args):

    data_module = QATextDataModule(
        args.dataset_name,
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    val_dataloader = data_module.val_dataloader()

    config = {}
    config["question_truncate"] = args.question_truncate
    config["answer_truncate"] = args.answer_truncate

    if args.baseline_no_finetune:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.pretrained_model_name)
    else:
        model = Seq2SeqFinetune.load_from_checkpoint(args.model_ckpt).model

    device = "cuda"

    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name)

    responses = []
    references = []
    for batch in tqdm.tqdm(val_dataloader):
        questions = batch["questions"]
        references.extend(batch["answers"])
        for question in questions:
            response = run_model(model, tokenizer, question, device=device)
            responses.append(response)
    # print(responses, references)
    eval_res = evaluate_squad2.evalute_f1(responses, references)
    print(eval_res)


def parse_arguments():

    parser = ArgumentParser()

    # trainer specific arguments

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)

    parser.add_argument("--gpus", type=int, required=True)
    parser.add_argument("--precision", type=int, default=16)

    # model specific arguments
    parser.add_argument("--question_truncate", type=int, default=512)
    parser.add_argument("--answer_truncate", type=int, default=128)
    parser.add_argument("--pretrained_model_name", type=str, default="")
    parser.add_argument("--model_ckpt", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--baseline_no_finetune", action="store_true")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()

    main(args)
