import os 
import sys 
from pprint import pprint
from argparse import ArgumentParser, Namespace
import re

import torch 
import transformers
from transformers import BartTokenizerFast  
from torch.utils.data import DataLoader
from torch.optim import AdamW 
from torch.optim.lr_scheduler import StepLR
from accelerate import Accelerator


from utils import LegacySeq2SeqDataset, mkdir, post_tokenize, base_arguments 
from models import BartModel


DEVICE = torch.device("cuda")




def generate_samples(args: Namespace = None):
   
    parser = base_arguments()
    parser = add_generate_arguments(parser)

    batch_size=8
    
    args = parser.parse_args() 


    # input_fname contains the output inputs formatted for the student model
    mkdir(args.output_dir)
    input_fname_sampled = "inputs_sampled.untokenized.txt" 
    labels_fname_sampled = "labels_sampled.untokenized.txt"
    input_fname = "inputs.untokenized.txt"
    labels_fname = "labels.untokenized.txt"

    
    accelerator = Accelerator(project_dir="checkpoints")
    model = BartModel().to(DEVICE)
    tok = BartTokenizerFast.from_pretrained("facebook/bart-large")

    model.load_pretrained("checkpoints/teacher-stable/best_tfmr")
    
    dataset_kwargs = dict(
        data_dir="datasets/dart",
        max_source_length=128,
        prefix="",
    )

    trainset = get_dataset("train", tok, args=args)
    testset = get_dataset("test", tok, args=args)
    valset = get_dataset("val", tok, args=args)

    # Batch_size really does not matter in this context
    training_loader = DataLoader(trainset, 
                                 batch_size=batch_size, 
                                 sampler=None, 
                                 shuffle=True,
                                 collate_fn=trainset.collate_fn)
    

    training_set_steps = len(trainset) // batch_size 
    with open(os.path.join(args.output_dir, labels_fname_sampled), "w") as lfile_sampled,\
        open(os.path.join(args.output_dir, input_fname_sampled), "w") as infile_sampled, \
        open(os.path.join(args.output_dir, labels_fname), "w") as lfile, \
        open(os.path.join(args.output_dir, input_fname), "w") as infile:
    
        for step in range(training_set_steps):

            batch = next(iter(training_loader))

            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            print("INPUT IDS: ", input_ids.shape)
            print("ATTN IDS: ", attention_mask.shape)
            print("LABEL IDS: ", labels.shape)

            sample_outputs = model.get_n_samples(input_ids=input_ids, 
                                attention_mask=attention_mask,
                                n=args.n_samples)
            # sample_outputs = model.model.generate(
            #         input_ids,
            #         attention_mask=attention_mask,
            #         #use_cache=True,
            #         num_beams=1,
            #         #return_
            #         repetition_penalty=1.,
            #         pad_token_id=model.model.config.pad_token_id,
            #         bos_token_id=model.model.config.eos_token_id,
            #         eos_token_id=model.model.config.eos_token_id,
            #         max_length=128,
            #         length_penalty=1.)
            
            # looping for each input example 
            for item in range(batch_size):
                output_sents = [] 

                # clean up sentences 
                
                # print(f"INPUT {batch_size * step + item}: ", tok.decode(input_ids[item], skip_special_tokens=False))
                # print(f"INPUT TOKENS REMOVED {batch_size * step + item}: ", tok.decode(input_ids[item], skip_special_tokens=True))
                input_sent = tok.decode(input_ids[item], skip_special_tokens=True)

                print(f"OUTPUT {batch_size * step + item}: ", tok.batch_decode(sample_outputs[item]))
                eos_stripped = mask_after_eos(sample_outputs[item], tok.eos_token_id)
                print(eos_stripped)
                print(f"OUTPUT TOKENS REMOVED {batch_size * step + item}: ", tok.batch_decode(eos_stripped, skip_special_tokens=True))
                output_sents = tok.batch_decode(eos_stripped, skip_special_tokens=True) 
               
                label_sent = tok.decode(labels[item], skip_special_tokens=True)
                print(f"LABELS {batch_size * step + item}: ", label_sent)

                print("\n")
                
                
                [infile_sampled.write(input_sent + os.linesep) for _ in output_sents]
                [lfile_sampled.write(output_sent + os.linesep) for output_sent in output_sents] 

                infile.write(input_sent + os.linesep) 
                lfile.write(label_sent + os.linesep)

    post_tokenize(os.path.join(args.output_dir, input_fname)) 
    post_tokenize(os.path.join(args.output_dir, labels_fname)) 

    post_tokenize(os.path.join(args.output_dir, input_fname_sampled)) 
    post_tokenize(os.path.join(args.output_dir, labels_fname_sampled)) 



def mask_after_eos(inputs: torch.Tensor, eos_tok_id: int, replace_with=0):
    """
    Sets all tokens after (and including) eos_tok_id to replace_with 
    """
    mask = torch.cumsum(inputs == eos_tok_id, dim=-1)
    print("MASK: ", mask)
    outputs = inputs.clone() 
    outputs[mask > 0] = replace_with 
    return outputs 
    


    
def add_generate_arguments(parser):

    parser.add_argument("--data_dir", type=str, default="datasets/dart", help="")
    parser.add_argument("--n_samples", type=int, default=3, help="How many output samples to generate per input sample")
    parser.add_argument("--output_dir", type=str, default="pseudo_labels/", help="Where do you want the pseudo-labels to go")
    return parser 







def get_dataset(type_path: str, tok, args: Namespace):
    """
    @param type_path: "train", "test", or "val"
    """
    dataset_kwargs = dict(
        data_dir=args.data_dir,
        max_source_length=args.max_source_length,
        prefix="",
        src_lang=None if args.src_lang is None else args.src_lang,
        tgt_lang=None if args.tgt_lang is None else args.tgt_lang,
    )
    n_obs = dict(
        train=args.n_train,
        val=args.n_val,
        test=args.n_test,
    )

    n_obs = {k: v if v>=0 else None for k, v in n_obs.items()}[type_path]
    print("N_OBS:", n_obs)
    
    dataset = LegacySeq2SeqDataset(tokenizer=tok, 
                                   type_path=type_path, 
                                   n_obs=n_obs, 
                                   max_target_length=args.max_target_length,
                                   **dataset_kwargs)
    
    return dataset













    
if __name__ == "__main__":

    
    model = BartModel().to(DEVICE)
    tok = BartTokenizerFast.from_pretrained("facebook/bart-large")




    # model.load_state_dict(torch.load("checkpoints/val_BLEU=0.497883-step_count=8.ckpt")["state_dict"])
    

    tokenized = tok(["today is a good day", "some random sentence two", "times three or something like that word lol"], return_tensors="pt", max_length=100, padding="longest")

    input_ids, attention_mask = tokenized.input_ids.to(DEVICE), tokenized.attention_mask.to(DEVICE) 
    print(input_ids)
    print(tok.batch_decode(input_ids))

    print(model.model.config.decoder_start_token_id)
    print(tok.eos_token_id)


    output = model.get_n_samples(input_ids, attention_mask, n=3)
    [print(tok.batch_decode(o)) for o in output]

    # pprint(torch.load("checkpoints/val_BLEU=0.497883-step_count=8.ckpt")["state_dict"].keys())
    # model.compute_random_sample(torch.rand((3, 5, 10), device=DEVICE))


    generate_samples(None)
    # print("VOCAB SIZE", tok.vocab_size)
    # print("PROBLEM TOKENS", tok.decode([50265]))
    # print("PROBLEM TOKENS", tok.decode([12]))
    # print("PROBLEM TOKENS", tok.decode([18288]))
