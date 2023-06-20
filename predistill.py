import os 
import sys 
from argparse import ArgumentParser, Namespace

import torch 
from torch.nn import Module 
from transformers import BartTokenizerFast 

from models import BartModel, BartStudentModel 
from utils import base_arguments 

def predistill(args: Namespace):

    teacher_model = BartModel()
    teacher_model.load_pretrained(args.teacher)

    tok = BartTokenizerFast.from_pretrained("facebook/bart-large")

    config = {"encoder": args.encoder_layers, "decoder": args.decoder_layers}
    student_model = BartStudentModel(teacher_model=teacher_model, config=config)

    print(student_model)



def add_predistill_args(parser: ArgumentParser) -> ArgumentParser :
    parser.add_argument("--teacher", type=str, default="facebook/bart-large", help="String or path to teacher model.")
    parser.add_argument("--gt_coeff", type=float, default=1., help="Coefficient to weigh ground truth loss.")
    parser.add_argument("--psl_coeff", type=float, default=1., help="Coefficient to weigh pseudo-label loss.")
    parser.add_argument("--encoder_layers", type=int, nargs="?", default=[0, 6, 11], help="Encoder layers to copy over")
    parser.add_argument("--decoder_layers", type=int, nargs="?", default=[0], help="Encoder layers to copy over")

    return parser

    




if __name__ == "__main__": 
    parser = base_arguments()
    parser = add_predistill_args(parser)
    
    args = parser.parse_args() 

    predistill(args)
