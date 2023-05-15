import torch
import torch.nn.functional as F
import numpy as np
from typing import Union, List, Dict
from transformers import T5Tokenizer, T5ForConditionalGeneration
from utils.constants import NEGATIVE_INF
from utils.utils import logits_to_entropy, mask_pad


class Policy:
    def __init__(self, model_name, temperature, device, reward_cond=False, tree_tokens=None):
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.device = device

        self.tokenizer = T5Tokenizer.from_pretrained('t5-base', pad_token="<|endoftext|>")
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        if reward_cond:
            self.tokenizer.add_tokens(tree_tokens, special_tokens=True)

            weights = self.model.get_input_embeddings().weight.detach().numpy()
            mean_weights, std_weights = np.mean(weights, axis=0), np.std(weights, axis=0)
            new_inits = np.vstack([np.random.normal(loc=mean_weights, scale=std_weights) for _ in tree_tokens])

        self.model.resize_token_embeddings(len(self.tokenizer))

        if reward_cond:
            with torch.no_grad():
                new_inits = torch.tensor(new_inits)
                self.model.get_input_embeddings().weight[-len(tree_tokens):, :] = new_inits

        self.model = self.model.to(self.device)
        self.model.parallelize()

        self.temperature = temperature

    def sample(self,
               prompts: Union[str, List[str]] = None,
               input_ids: torch.Tensor = None,
               attention_mask: torch.Tensor = None,
               max_len: int = 20,
               min_len: int = 3,
               sample: bool = True,
               top_k: int = None,
               top_p: float = None,
               temperature: float = None,
               sample_num: int = 1) -> Dict[str, Union[torch.Tensor, List[str]]]:

        if temperature is None:
            temperature = self.temperature

        if prompts is not None:
            assert input_ids is None and attention_mask is None, 'repeated input'
            if isinstance(prompts, str):
                prompts = [prompts]

            encodings_dict = self.tokenizer(prompts, return_tensors="pt", padding=True)
            input_ids = encodings_dict['input_ids'].to(self.device)
            attention_mask = encodings_dict['attention_mask'].to(self.device)

        else:
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
        
        num_return_sequences = sample_num * 2
        self.model.eval()
        with torch.no_grad():
            response_ids = self.model.generate(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    max_length=512,
                                    num_beams=sample_num*2, 
                                    no_repeat_ngram_size=3, 
                                    num_return_sequences=sample_num, 
                                    early_stopping=True
                            )

        response_text = [self.tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                         for output in response_ids]
        response_text = self.select_distractor(response_text, num_return_sequences, sample_num)
        return {
            'response/text': response_text,
        }

    def select_distractor(self, distractor_text, num_return_sequences, sample_num):
        final_distractor = []
        distractor_set = set()
        for i, r in enumerate(distractor_text):
            if len(distractor_set) < sample_num:
                distractor_set.add(r)
            
            if (i + 1) % num_return_sequences == 0:
                final_distractor.append(list(distractor_set))
                distractor_set = set()
        
        return final_distractor

    def forward_pass(self,
                     input_ids: torch.Tensor,
                     attention_mask: torch.Tensor,
                     labels: torch.Tensor,
                     labels_mask: torch.Tensor):

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        labels = labels.to(self.device)
        labels_mask = labels_mask.to(self.device)

        # forward pass to get next token
        outputs = self.model(input_ids = input_ids,
                            attention_mask = attention_mask,
                            labels = labels,
        )
        logits = outputs.logits[:, :, :]

        log_prob = F.log_softmax(logits, dim=-1)
        output_logprob = torch.gather(log_prob, 2, labels[:, :, None]).squeeze(2)
        output_entropy = logits_to_entropy(logits)
        lm_loss = -1. * output_logprob

        return {
            'response/log_prob': mask_pad(output_logprob, labels_mask),
            'response/lm_loss': mask_pad(lm_loss, labels_mask),
            'response/entropy': mask_pad(output_entropy, labels_mask),
            'response/logits': logits,
        }
