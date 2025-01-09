from argparse import ArgumentParser
from time import sleep
import torch
from torch import nn
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader
from transformers import (
    GPT2TokenizerFast,
    GPT2Tokenizer
)
from torch.optim import Adam

from datetime import datetime
from .data.knowledge import KnowledgeDataset
import numpy as np
from openai import OpenAI
import os
import asyncio
import time
import sys
import re
import json
import copy

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
api_key = "sk-e39450eb1e1d4a8e825df0a7e4f5f411"

class ExampleSelection(LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--dirpath", type=str, default="models"
        )
        parser.add_argument(
            "--knowledge_data_path",
            type=str,
            default="datasets/gsm8k_/deepseek/gsm8k_pool_20.json",
        )
        parser.add_argument(
            "--train_data_path",
            type=str,
            default="datasets/gsm8k_/gsm8k_train_100.json",
        )
        parser.add_argument(
            "--valid_data_path",
            type=str,
            default="datasets/gsm8k_/gsm8k_val_100.json",
        )
        parser.add_argument("--batch_size", type=int, default=10)
        parser.add_argument("--lr", type=float, default=1e-2)
        parser.add_argument("--num_workers", type=int, default=0)

        parser.add_argument("--model_name", type=str, default="deepseek")
        parser.add_argument("--sample_size", type=int, default=8)
        parser.add_argument("--pge_avg_samples", type=int, default=5)

        parser.add_argument("--task", type=str, help="Indicate Task Type",default="gsm8k")
        
        return parser

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        
        self.knowledge_dataset = KnowledgeDataset(self.hparams.knowledge_data_path)
        self.knowledge_dataset_train = KnowledgeDataset(self.hparams.train_data_path)
        self.knowledge_dataset_val = KnowledgeDataset(self.hparams.valid_data_path)

        train_size = len(self.knowledge_dataset_train)
    
        pool_size = len(self.knowledge_dataset)
        
        # 原始代码初始的概率设置有问题，应该是每个位置上选取pool中每个exemplar的概率都是1/pool_size
        # self.sample_probs = torch.FloatTensor([[1 / len(self.knowledge_dataset) * 8] * int(len(self.knowledge_dataset)/8)] * self.hparams.sample_size) 
        self.sample_probs = torch.FloatTensor([[1 / pool_size] * pool_size] * self.hparams.sample_size) 
        
        self.sample_probs.requires_grad = True
        
        # Activates manual optimization
        self.automatic_optimization = False
        
        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = True
        self.sample_probs.requires_grad = True

        self.ce_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mse_loss = torch.nn.MSELoss(reduction='none')
        self.count = 0
        print('The following parameters are trained:')
        for n, p in self.named_parameters():
            if p.requires_grad:
                print(n)
                
        self.model_filename = self.hparams.dirpath + '/model' + datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S') + "_" + str(self.count) + '.pt'
        

    def train_dataloader(self, shuffle=False):
        return DataLoader(
            self.knowledge_dataset_train,
            batch_size=self.hparams.batch_size,
            collate_fn=self.knowledge_dataset_train.collate_fn,
            shuffle=shuffle,
            num_workers = 10,
            drop_last=True
        )
        
    def val_dataloader(self, shuffle=False):
        return DataLoader(
            self.knowledge_dataset_val,
            batch_size=self.hparams.batch_size,
            collate_fn=self.knowledge_dataset_val.collate_fn,
            shuffle=shuffle,
            num_workers = 10,
            drop_last=True
        )

    def complete_gpt3(self, *args, **kwargs):
        # call GPT-3 API until result is provided and then return it
        response = None
        received = False

        while not received:
            # 设置deepseek的api_key
            client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
            response = client.chat.completions.create(*args, **kwargs)
            received = True
        return response

    # assuming only one target for one prompt, added to the end of the prompt
    def forward(self, prompts, targets):
        # deepseek不支持多个prompt同时输入，需要循环一次生成prompt的回复
        rsp = [self.complete_gpt3(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"{prompt}"}
            ] ,
            max_tokens=256,
            temperature = 0,
            logprobs = True
        ) for prompt in prompts]

        match = 0
        
        for i , target in enumerate(targets):
            # print(rsp[i])
            text = rsp[i].choices[0].message.content

            if(self.hparams.task == "gsm8k"):
                pattern = "The answer is \d{1,}\."
                sttr = re.search(pattern, text.replace("$","").replace(",","").replace("%",""))
                if (sttr is not None):
                    # 对照ground truth，记录正确的次数
                    if(sttr.group(0)[14:-1] == target.replace(",","")):
                        match += 1                  
            
        # print(f"match:{match}")
        # print(f"len: {len(targets)}")
        # print(f"acc: {match/len(targets)}")
        return match/len(targets)
        
    def training_step(self, batch, batch_idx=None):
        
        training_data = batch
        opt = self.optimizers()
        opt.zero_grad()
        with torch.no_grad():
            # For each k, we prepend a prompt and calculate loss
            prompts_dist = torch.distributions.Categorical(self.sample_probs)
            #print(prompts_dist)
            prompts_indices_list = []
            acc_list = []
            
            question, rationale, answer, ground_truth = training_data['Question'],training_data['Rationale'],training_data['Answer'],training_data['Ground_truth']

            prompts = []
            targets = []
            
            for i in range(0,len(question)):
                prompts.append("Q: " + question[i] + "\n" + "A:")
                targets.append(ground_truth[i])
            
            # 我们采样pge_avg_samples种prompt，以减少训练过程中的方差
            # 每种prompt中包含sample_size个exemplar
            for _ in range(self.hparams.pge_avg_samples):
                new_prompts = prompts
                # Sample exemplars for each prompt
                prompt_idx = prompts_dist.sample()
                prompts_indices_list.append(copy.deepcopy(prompt_idx))

                # Construct the prompt with the sampled exemplars
                for k in range(0,self.hparams.sample_size):
                    if(self.hparams.task == "gsm8k"):
                        prompt = "Question: " + self.knowledge_dataset[prompt_idx[k]]['Question'] + "\nAnswer:" + self.knowledge_dataset[prompt_idx[k]]['Rationale'] + " The answer is " + self.knowledge_dataset[prompt_idx[k]]['Ground_truth'] + ".\n\n"
                    
                    # 将选取的exemplar依次添加batch中的每个question后面
                    new_prompts = [(prompt + x).strip() for x in new_prompts]

                # Compute loss
                acc = self(new_prompts, targets)
                self.count = self.count + 1

                acc_list.append(acc)
            
            # Compute average loss
            acc_avg = sum(acc_list)/len(acc_list)
            self.log('acc_avg', acc_avg, prog_bar=True, on_step=True, on_epoch=True, batch_size=len(batch['id']))
            
            with open("./log.txt", "a") as file:
                file.write(f"acc: {acc_list}\n")
                file.write(f"avg_acc: {acc_avg}\n")
                file.write(f"prompts indices: {prompts_indices_list}\n")
                file.write(f"Prompts:\n {prompts}\n")
                file.write(f"len: {len(question)}\n")
            
            # Update the policy
            derivative = [-1 / self.sample_probs] * self.hparams.pge_avg_samples   
            for k, indice in enumerate(prompts_indices_list):
                for i in range(0,self.hparams.sample_size):
                    derivative[k][i][indice[i]] *= -1

            self.sample_probs.grad = torch.zeros_like(self.sample_probs)
            for k in range(self.hparams.pge_avg_samples):
                self.sample_probs.grad -= 1 / (self.hparams.pge_avg_samples - 1) * (acc_list[k] - acc_avg) * derivative[k]

            torch.nn.utils.clip_grad_norm_(self.sample_probs, 3)

            opt.step()

            self.constrain_score_by_whole_exact(self.sample_probs)


    def validation_step(self, batch, batch_idx=None):
        testing_data = batch
        
        question, rationale, answer, ground_truth = testing_data['Question'],testing_data['Rationale'],testing_data['Answer'],testing_data['Ground_truth']
        prompts = []
        targets = []
        for i in range(0,len(question)):
            prompts.append("Q: " + question[i] + "\n" + "A:")            
            targets.append(ground_truth[i])
        
        # 选取当前policy中每个位置概率最高的exemplar，作为测试的prompt
        # Create a tensor of size [prompt_length]
        prompt_idx = torch.zeros(len(self.sample_probs),dtype=torch.int64) 
        # Fill it with argmax for each dimension
        for i in range(0,len(self.sample_probs)):
            # Retrieve argmax index
            idx = (self.sample_probs[i]==max(self.sample_probs[i])).nonzero().squeeze()
            # If only one index are retrieved 
            if(idx.size() == torch.Size([])):
                prompt_idx[i] = idx
            else:
                prompt_idx[i] = idx[0]

        for k in range(0,self.hparams.sample_size):
            # Construct the prompt
            if(self.hparams.task == "gsm8k"):
                prompt = "Question: " + self.knowledge_dataset[prompt_idx[k]]['Question'] + "\nAnswer: Let's think step by step." + self.knowledge_dataset[prompt_idx[k]]['Rationale'] + " The answer is " + self.knowledge_dataset[prompt_idx[k]]['Ground_truth'] + ".\n\n"
            prompts = [(prompt + x).strip() for x in prompts]

        # Compute accuracy
        acc = self(prompts, targets)
        with open("./log.txt", "a") as file:
            file.write("One Validation Step End. Evaluating Result : \n")
            file.write(f"Validation Acc{acc}\n")
            
        return torch.tensor(acc)

    def configure_optimizers(self):
        optimizer = Adam([self.sample_probs], lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.max_epochs, eta_min=0)
        return [optimizer], [scheduler]

    
    def on_train_epoch_end(self):
        print(self.sample_probs)
        print("Trian epoch end, saving Model")
        with open("./log.txt", "a") as file:
            file.write("Trian epoch end, saving Model\n")
            file.write(f"Sample probs:\n {self.sample_probs}\n")
    
    def solve_v_total_exact(self, prompt_emb):
        k = 1
        a, b = 0, 0

        b = prompt_emb.max()
        def f(v):
            s = (prompt_emb - v).clamp(0.0001, 1).sum()
            return s - k
        itr = 0

        v = 0
        while True:
            itr += 1
            v = (a + b) / 2
            obj = f(v)
            if abs(obj) < 1e-3 or itr > 30:
                break
            if obj < 0:
                b = v
            else:
                a = v
        return v, itr

    def constrain_score_by_whole_exact(self, prompt_embeds):
        for i in range(len(prompt_embeds)):
            v, itr = self.solve_v_total_exact(prompt_embeds[i])
            prompt_embeds[i].sub_(v).clamp_(0.0001, 1)