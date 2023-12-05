from transformers import AutoModelForMaskedLM, AutoTokenizer,AutoModelForCausalLM
import torch
import numpy as np


class perplexity_score:
    '''
    Input : String of text or sentence
    Output : perplexity score
    '''
    def __init__(self):
        self.model_name = 'cointegrated/rubert-tiny'
        self.model = AutoModelForMaskedLM.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def calculate(self,sentence):
        tensor_input = self.tokenizer.encode(sentence, return_tensors='pt')
        repeat_input = tensor_input.repeat(tensor_input.size(-1)-2, 1)
        mask = torch.ones(tensor_input.size(-1) - 1).diag(1)[:-2]
        masked_input = repeat_input.masked_fill(mask == 1, self.tokenizer.mask_token_id)
        labels = repeat_input.masked_fill( masked_input != self.tokenizer.mask_token_id, -100)
        with torch.inference_mode():
            loss = self.model(masked_input, labels=labels).loss
        return np.exp(loss.item())