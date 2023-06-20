import os 
import sys
from copy import deepcopy
from typing import Optional, List, Tuple, Union

import torch 
from torch.nn import Module, Softmax, ModuleList 
from transformers import BartForConditionalGeneration, LogitsProcessorList, BartTokenizerFast 


DEVICE=torch.device("cuda")


COPY_LAYERS = {
    "encoder": [0, 6, 11],
    "decoder": [0]
}




class BartModel(Module):
    def __init__(self):
        super().__init__()
        self.model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")
    
    def get_n_samples(self,
                      input_ids: torch.Tensor, 
                      attention_mask: torch.Tensor = None, 
                      n: int = 3,
                      max_len: int = 100):
        sequences = []
        for seq_id in range(input_ids.shape[0]):
            input_ids_sentence_repeated = torch.cat([input_ids[seq_id][None, :] for _ in range(n)], dim=0)
            attention_mask_sentence_repeated = (
                    None if attention_mask is None 
                    else torch.cat([attention_mask[seq_id][None, :] for _ in range(n)], dim=0)
            )
            gen_outputs = self.model.generate(inputs=input_ids_sentence_repeated, 
                                              attention_mask=attention_mask_sentence_repeated, 
                                              use_cache=True, 
                                              decoder_start_token_id=self.model.config.decoder_start_token_id, 
                                              max_length=max_len, 
                                              num_beams=1, 
                                              do_sample=False)
            print("GEN OUTPUTS", gen_outputs)
            sequences.append(gen_outputs) 
        return sequences
    def forward(self, 
                input_ids: torch.LongTensor = None,
                attention_mask: Optional[torch.Tensor] = None,
                decoder_input_ids: Optional[torch.LongTensor] = None,
                decoder_attention_mask: Optional[torch.LongTensor] = None,
                head_mask: Optional[torch.Tensor] = None,
                decoder_head_mask: Optional[torch.Tensor] = None,
                cross_attn_head_mask: Optional[torch.Tensor] = None,
                encoder_outputs: Optional[List[torch.FloatTensor]] = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,):
        return self.model(input_ids, 
                          attention_mask=attention_mask, 
                          decoder_input_ids=decoder_input_ids, 
                          decoder_attention_mask=decoder_attention_mask, 
                          head_mask=head_mask, 
                          decoder_head_mask=decoder_head_mask, 
                          cross_attn_head_mask=cross_attn_head_mask, 
                          encoder_outputs=encoder_outputs, 
                          past_key_values=past_key_values, 
                          inputs_embeds=inputs_embeds, 
                          decoder_inputs_embeds=decoder_inputs_embeds, 
                          labels=labels, 
                          use_cache=use_cache, 
                          output_attentions=output_attentions, 
                          output_hidden_states=output_hidden_states, 
                          return_dict=return_dict)



    def get_n_samples_2(self, 
                      input_ids: torch.Tensor, 
                      attention_mask: torch.Tensor=None, 
                      n: int=3, 
                      max_len:int=100):
        """
        for every input sentence, generate n output samples 

        @param input_ids: (N, S)
        @param attention_mask: (N, S)

        @returns sequences: list(torch.Tensor(n, s)): List of size N, containing tensors 
                            with n samples and sequence length of s

        """
        encoder = self.model.get_encoder()
        decoder = self.model.get_decoder()
       
        sequences = []
        logits_processor = LogitsProcessorList()
        
        for seq_id in range(input_ids.shape[0]):
            

            input_ids_sentence_repeated = torch.cat([input_ids[seq_id][None, :] for _ in range(n)], dim=0)
            attention_mask_sentence_repeated = None if attention_mask is None else torch.cat([attention_mask[seq_id][None, :] for _ in range(n)], dim=0)
            
            # print("INPUT SHAPE:", input_ids_sentence_repeated.shape)
            # print("ATTN SHAPE: ", attention_mask_sentence_repeated.shape)
            enc_outputs = encoder(input_ids_sentence_repeated, 
                                  attention_mask=attention_mask_sentence_repeated, 
                                  output_attentions=True, 
                                  output_hidden_states=True)
            enc_hidden_state, all_hidden_states, enc_attentions = enc_outputs.last_hidden_state, enc_outputs.hidden_states, enc_outputs.attentions 
        


            past_key_values = None 
            dec_input_ids = torch.ones((input_ids_sentence_repeated.shape[0], 1), dtype=torch.long, device=DEVICE) * self.model.config.decoder_start_token_id
            sequence = None

            for i in range(max_len):
                # print("ITERATION:", i)
                # print("DECODER INPUT IDS", dec_input_ids)

                # print("DEC_INPUT_IDS",  dec_input_ids)
                dec_outputs = decoder(dec_input_ids, 
                                      encoder_hidden_states=enc_hidden_state, 
                                      encoder_attention_mask=attention_mask_sentence_repeated, 
                                      past_key_values=past_key_values,
                                      output_hidden_states=True,
                                      use_cache=True)

                last_hidden_state = dec_outputs.last_hidden_state 
                past_key_values = dec_outputs.past_key_values 

                lm_logits = self.model.lm_head(last_hidden_state)
                lm_logits = lm_logits + self.model.final_logits_bias.to(lm_logits.device)
                lm_logits[:, -1, self.model.config.bos_token_id] -= 100 if sequence is not None else 0 


                # greedy 
                # last_val = torch.argmax(Softmax(dim=-1)(lm_logits[:, -1:, :]), dim=-1)
                # print(last_val.shape)
                # dec_input_ids = last_val

                
                # Sampled 
                last_val = self.compute_random_sample(lm_logits[:, -1:, :])
                # print(last_val.shape)
                dec_input_ids = last_val 




                sequence = last_val if sequence is None else torch.cat([sequence, last_val], dim=-1)
                if self.all_eos(sequence):
                    break
            
            sequences.append(sequence) 

        return sequences
    
    def compute_random_sample(self, logits):
        """
        @params logits: (N, S, C)
        @returns (N, S)
        """

        probs = Softmax(dim=-1)(logits) 
        cdf = torch.cumsum(probs, dim=-1)
        cdf[:, :, -1] += 0.1 # Final bin of cdf is 1.1, to account for floating point imprecision 
        thresh = torch.rand((logits.shape[0], logits.shape[1]), device=DEVICE)[:, :, None]
        thresh_mask = cdf < thresh
        sample_output = torch.sum(thresh_mask, dim=-1)

        if False:
            print(probs)
            print(cdf)
            print(thresh)
            print(thresh_mask)
            print(torch.sum(thresh_mask, dim=-1))
        return sample_output 




    def all_eos(self, sequences):
        """
        Checks if all sequences in the batch has EOS token
        @param sequence: (N, S)
        """
        return (torch.sum(torch.sum(sequences == self.model.config.eos_token_id, dim=-1) == 0) == 0)

    def load_pretrained(self, path: str):
    
        # self.model.load_state_dict(torch.load(os.path.join(path, "pytorch_model.bin")))
        self.model = BartForConditionalGeneration.from_pretrained(path).to(DEVICE)




class BartStudentModel(BartModel):
    def __init__(self, teacher_model:BartModel=None, config:dict=None):
        super().__init__()
            
        if teacher_model is not None:
            assert config is not None, "If teacher model is defined, so should the config argument"
        
        self.config = config


        self.model = self.copy_from_teacher(teacher_model, config)
        
    def copy_from_teacher(self, teacher_model:BartModel=None, config:dict=None):

        if teacher_model is None: 
            teacher_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")
            config = {"decoder":[0], "encoder":[0, 6, 11]}

        teacher_encoder = teacher_model.model.get_encoder()
        teacher_decoder = teacher_model.model.get_decoder() 

        student_encoder = deepcopy(teacher_encoder)
        student_encoder.layers = ModuleList([teacher_encoder.layers[i] for i in config["encoder"]])
        student_decoder = deepcopy(teacher_decoder)
        student_decoder.layers = ModuleList([teacher_decoder.layers[i] for i in config["decoder"]])

        student_model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
        student_model.model.decoder = student_decoder 
        student_model.model.encoder = student_encoder

        return student_model
    



if __name__ == "__main__":
   
    teacher_model = BartModel()
    student_model = BartStudentModel(teacher_model, config=COPY_LAYERS)
    
    print(teacher_model)
    print("\n\n")
    print(student_model)


    tok = BartTokenizerFast.from_pretrained("facebook/bart-large")
    print(tok.pad_token_id)
    print(tok.bos_token_id)
    print(tok.eos_token_id)
    
    tok = BartTokenizerFast.from_pretrained("facebook/bart-base")
    print(tok.pad_token_id)
    print(tok.bos_token_id)
    print(tok.eos_token_id)

