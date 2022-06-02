import sys
import os
from sklearn.utils import shuffle
import random
from collections import defaultdict
import tqdm
import copy
import torch

import sklearn
from sklearn.linear_model import LogisticRegression
import random
import pickle
import numpy as np
import argparse
from transformers import BertModel, BertTokenizer

def load_bios(group):

    with open("bios_data/{}.pickle".format(group), "rb") as f:
        bios_data = pickle.load(f)
        txts = [d["hard_text_untokenized"] for d in bios_data]
        
    return txts


def encode(bert, texts):
    
    all_H = []
    bert.eval()
    with torch.no_grad():
        
        print("Encoding...")
        batch_size = 100
        pbar = tqdm.tqdm(range(len(texts)), ascii=True)
        
        for i in range(0, len(texts)-batch_size, batch_size):
            
                batch_texts = texts[i: i + batch_size]
                
                batch_encoding = tokenizer.batch_encode_plus(batch_texts, padding=True, max_length=512,truncation=True)
                input_ids, token_type_ids, attention_mask = batch_encoding["input_ids"], batch_encoding["token_type_ids"], batch_encoding["attention_mask"]
                input_ids = torch.tensor(input_ids).to(device)
                token_type_ids = torch.tensor(token_type_ids).to(device)
                attention_mask = torch.tensor(attention_mask).to(device)
                H = bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)["pooler_output"]
                assert len(H.shape) == 2
                all_H.append(H.detach().cpu().numpy())
                
                pbar.update(batch_size)
    
        remaining = texts[(len(texts)//batch_size)*batch_size:]
        print(len(remaining))
        if len(remaining) > 0:
            batch_encoding = tokenizer.batch_encode_plus(remaining, padding=True, max_length=512,truncation=True)
            input_ids, token_type_ids, attention_mask = batch_encoding["input_ids"], batch_encoding["token_type_ids"], batch_encoding["attention_mask"]
            input_ids = torch.tensor(input_ids).to(device)
            token_type_ids = torch.tensor(token_type_ids).to(device)
            attention_mask = torch.tensor(attention_mask).to(device)
            H = bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)["pooler_output"]
            assert len(H.shape) == 2
            all_H.append(H.detach().cpu().numpy())    
    
    H_np = np.concatenate(all_H)
    assert len(H_np.shape) == 2
    assert len(H_np) == len(texts)
    return H_np


if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    
    parser = argparse.ArgumentParser(description="An argparse example")
    parser.add_argument('--device', type=int, default=-1, required=False)
    parser.add_argument('--run_id', type=int, default=-1, required=True)
    args = parser.parse_args()
    device = "cpu" if args.device == -1 else "cuda:{}".format(args.device)
    print(device)
    
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert = BertModel.from_pretrained("bert-base-uncased")
    bert.to(device)
    bert.eval()
    rand_seed = args.run_id

    if not os.path.exists("encodings"):
        os.makedirs("encodings")
    if not os.path.exists("encodings/mlp-adv"):
        os.makedirs("encodings/mlp-adv")
    if not os.path.exists("encodings/linear-adv"):
        os.makedirs("encodings/linear-adv")
    if not os.path.exists("encodings/no-adv"):
        os.makedirs("encodings/no-adv")

    for finetuning_type in ["not-adv"]: #["adv", "mlp_adv", "not-adv"]:
        for rand_seed in range(5):
            for mode in ["train", "dev", "test"]:
        
                txts =  load_bios(mode)
                txts = txts[:]
                
                if finetuning_type == "adv":
                    print("Loading adv")
                    bert_params =  torch.load("models/linear-adv/bert_{}.pt".format(rand_seed))
                elif finetuning_type == "mlp_adv":
                    print("Loading MLP adv")
                    bert_params =  torch.load("models/mlp-adv/bert_{}.pt".format(rand_seed))
                else:
                    bert_params =  torch.load("models/no-adv/bert_{}.pt".format(rand_seed))
                    
                bert.load_state_dict(bert_params)
                H = encode(bert, txts)
                path = "encodings/linear-adv/{}_{}_cls.npy".format(mode,rand_seed) if finetuning_type == "adv" else "encodings/mlp-adv/{}_{}_cls.npy".format(mode,rand_seed) if finetuning_type == "mlp_adv" else "encodings/no-adv/{}_{}_cls.npy".format(mode,rand_seed)
                np.save(path, H)