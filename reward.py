import json
from pathlib import Path
import os

import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
from torch.utils.data import Dataset, DataLoader
from typing import Optional, List, Iterable, Dict, Any

import torch
import torch.nn as nn
from transformers import DebertaV2Config, DebertaV2Model, DebertaV2Tokenizer

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
log = logging.getLogger(__name__)

class DistractorRewardModelDataset(Dataset):
    def __init__(self, questions, distractors, answer_rewards, tokenizer): 
        input_datas = [HighlightDistractor(q, d) for q, d in zip(questions, distractors)]
        self.input_encodings = tokenizer(input_datas,
                                            max_length=512,
                                            padding="max_length",
                                            truncation=True,
                                            return_tensors="pt")
        self.answer_rewards = answer_rewards

    def __len__(self):
        return len(self.input_encodings['input_ids'])

    def __getitem__(self, idx):
        return {'input_ids': self.input_encodings['input_ids'][idx],
                'attention_mask': self.input_encodings['attention_mask'][idx],
                'answer_reward': self.answer_rewards[idx]}

class DebertaRM(nn.Module):
    def __init__(self,
                 pretrained: str = None,
                 config: Optional[DebertaV2Config] = None,
                 checkpoint: bool = False,) -> None:
        super().__init__()
        if pretrained is not None:
            model = DebertaV2Model.from_pretrained(pretrained)
        elif config is not None:
            model = DebertaV2Model(config)
        else:
            model = DebertaV2Model(DebertaV2Config())
        if checkpoint:
            model.gradient_checkpointing_enable()
        value_head = nn.Linear(model.config.hidden_size, 1)
        value_head.weight.data.normal_(mean=0.0, std=1 / (model.config.hidden_size + 1))
        
        self.model = model
        self.value_head = value_head
        
    def forward(self, input_ids: torch.LongTensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        outputs = self.model(input_ids, attention_mask=attention_mask)
        last_hidden_states = outputs['last_hidden_state']
        values = self.value_head(last_hidden_states)[:, :-1]
        value = values.mean(dim=1).squeeze(1)    # ensure shape is (B)
        return value

class Reward:
    def __init__(self, model_path, device, batch_size: int):
        self.device = device
        self.model = DebertaRM(pretrained='microsoft/deberta-v3-large')
        self.model = self.model.to(torch.float16)

        self.tokenizer = DebertaV2Tokenizer.from_pretrained('microsoft/deberta-v3-large')
        self.tokenizer.pad_token = self.tokenizer.eos_token

        ## add distractor token
        self.D_TOKEN = "[D]"
        self.tokenizer.add_tokens([self.D_TOKEN], special_tokens=True)
        self.model.model.resize_token_embeddings(len(self.tokenizer))

        state_dict = torch.load(model_path)
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)

        self.batch_size = batch_size

    def get_reward(self, questions: List[str], distractors: List[str], answer_rewards: List[float]) -> List[float]:
        reward_dataset = DistractorRewardModelDataset(questions = questions, 
                                                    distractors = distractors, 
                                                    answer_rewards = answer_rewards, 
                                                    tokenizer = self.tokenizer)
        reward_dataloader = DataLoader(reward_dataset, batch_size=self.batch_size, shuffle=False)
        rewards = []
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(reward_dataloader, desc='get reward score'):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                answer_reward = batch['answer_reward']
                reward = self.model(input_ids = input_ids, attention_mask=attention_mask)
                rewards.extend([r - a for r, a in zip(reward, answer_reward)])
        return [reward.cpu() for reward in rewards]
def HighlightDistractor(question, distractor):
    D_TOKEN = "[D]"
    distractor = D_TOKEN + " " + distractor + " " + D_TOKEN
    question = question.replace('_', distractor, 1)
    return question
