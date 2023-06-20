import os 
import sys 
from argparse import Namespace
from typing import Optional

import torch 
from torch.nn import CrossEntropyLoss 
from torch.optim import AdamW, SGD, Adam 
from torch.utils.data import Dataset, DataLoader, RandomSampler
from transformers import BartTokenizerFast, PreTrainedTokenizer


from utils import LegacySeq2SeqDataset, base_arguments, mkdir
from metrics import calc_eval_metrics
from models import BartModel


DEVICE=torch.device("cuda")


class BaseTrainer:
    def __init__(self, 
                 model, 
                 tokenizer: PreTrainedTokenizer, 
                 trainset:Dataset, 
                 valset:Dataset=None, 
                 training_args: Namespace=None):
        self.model = model
        self.tokenizer = tokenizer
        self.training_args = training_args 
        self.training_args = (
                training_args if training_args is not None 
                else add_training_args(base_arguments()).parse_args()
        ) 
        if self.training_args.optim == "sgd":
            self.optim = SGD(self.model.parameters(), lr=self.training_args.lr, momentum=0.9)
        elif self.training_args.optim == "adam":
            self.optim = Adam(self.model.parameters(), lr=self.training_args.lr)
        else: 
            self.optim = AdamW(self.model.parameters(), lr=self.training_args.lr)
        self.valset = valset
        self.trainset = trainset 
        print(self.training_args)
        self.tloader = self.get_dataloader(self.trainset)
        self.vloader = self.get_dataloader(self.valset) # Option[Dataset]
    def train(self):
        mkdir(self.training_args.save_dir)
        self.warmup()
        epochs = self.training_args.epochs if self.training_args.epochs is not None else 1000000
        batch_size = self.training_args.batch_size
        for epoch in range(epochs):
            tloader_iter = iter(self.tloader)
            steps_per_epoch = len(self.trainset) // batch_size
            for step in range(steps_per_epoch):
                batch = next(tloader_iter)
                self.model.zero_grad(set_to_none=True)
                loss, logits = self.train_step(batch=batch, epoch=epoch, step=step)
                loss.backward()
                self.optim.step()
                if step % self.training_args.val_interval == 0 and step != 0:
                    self.validate()
                    self.model.model.save_pretrained(os.path.join(self.training_args.save_dir))
    def warmup(self):
        warmup_steps = self.training_args.warmup_steps 
        self.set_optim_lr(self.training_args.lr / 10) 
        for step in range(warmup_steps):
            batch = next(iter(self.tloader))
            self.model.zero_grad(set_to_none=True)
            loss, logits = self.train_step(batch=batch, epoch="WARMUP", step=step)
            loss.backward()
            self.optim.step()
        self.set_optim_lr(self.training_args.lr)
    def validate(self):
        if self.valset is None or self.vloader is None:
            return 
        gen_outputs = []
        labels = []
        vloader_iter = iter(self.vloader)
        validation_steps = len(self.valset) // self.training_args.batch_size
        for step in range(validation_steps):
            with torch.no_grad():
                batch=next(vloader_iter)
                gen_output, label = self.val_step(batch=batch, step=step)
                # print("LABEL: ", label)
                # print("GEN_OUTPUT: ", gen_output)
                # print("")
                labels += label 
                gen_outputs += gen_output 
        eval_mets = calc_eval_metrics(gen_output, labels) 
        print(eval_mets)

    def set_optim_lr(self, lr:float):
        for g in self.optim.param_groups:
            g['lr'] = lr
    def get_optim_lr(self):
        for g in self.optim.param_groups:
            return g["lr"]
    def get_dataloader(self, dataset:Dataset):
        raise NotImplementedError("You must implement get_dataloader")
    def train_step(self, **kwargs):
        raise NotImplementedError("You must implement train_step")
    def val_step(self, **kwards):
        raise NotImplementedError("You must implement val_step")


class SingleModelFineTune(BaseTrainer):
    def __init__(self, 
                 model, 
                 tokenizer: PreTrainedTokenizer, 
                 trainset:Dataset, 
                 valset:Dataset=None, 
                 training_args:Namespace=None):
        super().__init__(model, tokenizer, trainset, valset, training_args)
        if self.training_args.verbose_interval is None: 
            self.training_args.verbose_interval = 50 
        self.logger = Logger(interval=self.training_args.verbose_interval)
    def get_dataloader(self, dataset:LegacySeq2SeqDataset) -> Optional[DataLoader]:
        if dataset is None:
            return None
        sampler=RandomSampler(dataset)
        return DataLoader(
            dataset,
            batch_size=self.training_args.batch_size,
            collate_fn=dataset.collate_fn,
            shuffle=self.training_args.shuffle, 
            sampler=None,
        ) 
    def train_step(self, batch, epoch=None, step=None):
        logs = []
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        # print(input_ids)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logs.append(("loss", outputs['loss']))
        self.logger.log(metrics=logs, epoch=epoch, step=step, lr=self.get_optim_lr())
        return (outputs["loss"], outputs["logits"])
    def val_step(self, batch, step=None,):
        logs = [("completion", f"{step * 100 / (len(self.valset)//self.training_args.batch_size):.2f}%")]
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        outputs = self.model.model.generate(input_ids=input_ids, 
                                            attention_mask=attention_mask, 
                                            num_beams=1)
        sequences = self.tokenizer.batch_decode(outputs, skip_special_tokens=True) 
        labels = [[toked] for toked in self.tokenizer.batch_decode(labels, skip_special_tokens=True)]
        self.logger.log(metrics=logs, epoch="VALIDATION", step=step) 
        return sequences, labels
class Logger:
    def __init__(self, interval=None, beta=0.9):
        self.interval=interval
        self.cache = {}
        self.beta = beta
    def log(self, epoch=None, step=None, lr=None, metrics=[]):
        if step is None:
            return 
        for item in metrics:
            assert isinstance(item, tuple), "Log item must be in the form (A, B). i.e: (\"loss\", 3.1415)"
        for item in metrics:
            self.cache[item[0]] = (
                item[1] if item[0] not in self.cache
                else (
                    item[1] if isinstance(item[1], str)
                    else self.beta * self.cache[item[0]] + (1 - self.beta) * item[1]
                )
            )
        if step % self.interval != 0:
            return
        print(f"E={epoch}, ", end="") if epoch is not None else None 
        print(f"B={step}", end="")
        print(f", lr={lr}", end="") if lr is not None else None
        for item in metrics:
            print(f", {item[0]}={self.cache[item[0]]}", end="")\
                    if isinstance(self.cache[item[0]], str)\
                    else print(f", {item[0]}={self.cache[item[0]]:.4f}", end="")
        print("")
def add_training_args(parser):
    parser.add_argument("--data_dir", type=str, default="datasets/dart", help="Directory to data")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--optim", type=str, default="adamw", help="Optimizer to use. 'adamw', 'adam' or 'sgd'")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs. If none, then train forever")
    parser.add_argument("--shuffle", action="store_true", default=False, help="Whether or not to shuffle the dataset on training")
    parser.add_argument("--verbose_interval", 
                        type=int, 
                        default=50, 
                        help="Interval of printing logging info to the screen. "
                            "If none, then do not print.")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Number of steps to learn at a slower learning rate")
    parser.add_argument("--val_interval", type=int, default=5000, help="How many training steps to do before validation")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Where to save the model")
    return parser

if __name__ == "__main__":
    parser = base_arguments()
    parser = add_training_args(parser)
    args = parser.parse_args()
    model = BartModel().to(DEVICE)
    tok = BartTokenizerFast.from_pretrained("facebook/bart-large", model_max_length=512)
    tset = LegacySeq2SeqDataset(tok, 
                                data_dir=args.data_dir, 
                                max_source_length=args.max_source_length, 
                                max_target_length=args.max_target_length, 
                                type_path="train")
    vset = LegacySeq2SeqDataset(tok, 
                                data_dir=args.data_dir, 
                                max_source_length=args.max_source_length, 
                                max_target_length=args.max_target_length, 
                                type_path="val")
    testset = LegacySeq2SeqDataset(tok, 
                                data_dir=args.data_dir, 
                                max_source_length=args.max_source_length, 
                                max_target_length=args.max_target_length, 
                                type_path="test")
    trainer = SingleModelFineTune(model, tok, trainset=tset, valset=vset, training_args=None)
    trainer.train()
