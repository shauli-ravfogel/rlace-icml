import sys
import os

from sklearn.linear_model import SGDClassifier, LinearRegression, Lasso, Ridge
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
import seaborn as sn
import random
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from sklearn.manifold import TSNE
import tqdm
import copy
from sklearn.svm import LinearSVC 

from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import TruncatedSVD
import torch
from sklearn.linear_model import SGDClassifier

from sklearn.svm import LinearSVC
from pytorch_revgrad import RevGrad

import sklearn
from sklearn.linear_model import LogisticRegression
import random
import pickle
import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn import neural_network
import numpy as np
from transformers import BertModel, BertTokenizer
import argparse
import time
import wandb




def load_bios(group):

    X = np.load("bios_data/{}_cls.npy".format(group))
    with open("bios_data/{}.pickle".format(group), "rb") as f:
        bios_data = pickle.load(f)
        Y = np.array([1 if d["g"]=="f" else 0 for d in bios_data])
        professions = np.array([d["p"] for d in bios_data])
        txts = [d["hard_text_untokenized"] for d in bios_data]
        random.seed(0)
        np.random.seed(0)
        X,Y,professions,txts,bios_data = sklearn.utils.shuffle(X,Y,professions,txts,bios_data)
        X = X[:]
        Y = Y[:]
        
    return X,Y,txts,professions,bios_data



parser = argparse.ArgumentParser(description="An argparse example")

parser.add_argument('--adv', type=int, default=0)
parser.add_argument('--mlp_adv', type=int, default=0)
parser.add_argument('--run_id', type=int, default=0)
parser.add_argument('--device', type=int, default=-1, required=False)
parser.add_argument('--opt', type=str, default="sgd", required=False)
parser.add_argument('--iters', type=int, default=30000, required=False)



args = parser.parse_args()
print(args)

if not os.path.exists("models"):
    os.makedirs("models")
if not os.path.exists("models/mlp-adv"):
    os.makedirs("models/mlp-adv")
if not os.path.exists("models/linear-adv"):
    os.makedirs("models/linear-adv")
if not os.path.exists("models/no-adv"):
    os.makedirs("models/no-adv")

adv = args.adv == 1
device="cuda:{}".format(args.run_id % 4) if args.device == -1 else "cuda:{}".format(args.device)

X,y_gender,txts,professions,bios_data =  load_bios("train")
X_dev,y_dev_gender,txts_dev,professions_dev,bios_data_dev =  load_bios("dev")


random.seed(args.run_id)
np.random.seed(args.run_id)
X,y_gender,txts,professions = X[:], y_gender[:], txts[:], professions[:]
X_dev,y_dev_gender,txts_dev,professions_dev = X_dev[:N],y_dev_gender[:N],txts_dev[:N],professions_dev[:N]

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert = BertModel.from_pretrained("bert-base-uncased")
prof2ind = {p:i for i,p in enumerate(sorted(set(professions)))}
ind2prof = {i:p for i,p in prof2ind.items()}
y_prof = np.array([prof2ind[p] for p in professions])
y_dev_prof = np.array([prof2ind[p] for p in professions_dev])

W = torch.nn.Linear(768, len(prof2ind))
if adv:
    if not args.mlp_adv:
        adv_clf = torch.nn.Linear(768, 2)
        adv_clf = torch.nn.Sequential(*[RevGrad(), adv_clf]) 
        adv_clf.to(device)
    if args.mlp_adv:
        adv_clf = torch.nn.Sequential(*[RevGrad(), torch.nn.Linear(768,300), torch.nn.ReLU(), torch.nn.LayerNorm(300), torch.nn.Linear(300,2)]).to(device)
        
loss_fn = torch.nn.CrossEntropyLoss()

if args.opt == "sgd":
    if not adv:
        lr,momentum,decay  =  0.5*1e-3, 0.9, 1e-6
        optimizer = torch.optim.SGD(list(bert.parameters()) + list(W.parameters(())), lr = lr, momentum = momentum, weight_decay = decay)
    else:
        lr, momentum, decay = 0.5*1e-3, 0.8, 1e-6
        optimizer = torch.optim.SGD(list(bert.parameters())+ list(W.parameters()) +  list(adv_clf.parameters()), lr = lr, momentum = momentum, weight_decay = decay)
else:
    lr,momentum,decay = None, None, None
    if not adv:
        optimizer = torch.optim.Adam(list(bert.parameters()) + list(W.parameters()), lr = 1e-4)
    else:
        optimizer = torch.optim.Adam(list(bert.parameters())+ list(W.parameters()) +  list(adv_clf.parameters()), lr = 1e-4)

W.to(device)
bert.to(device)

loss_vals = []
bert.train()
best_score = 10000
best_model = None

def eval_dev(bert, W, texts_dev, y_dev, device, adv=None, y_dev_gender=None):
    
    loss_vals = []
    bert.eval()
    
    accs = []
    if adv is not None:
        accs_adv = []
        
                
    with torch.no_grad():
        
        print("Evaluating...")
        batch_size = 150
        pbar = tqdm.tqdm(range(len(texts_dev)), ascii=True)
        
        for i in range(0, len(texts_dev)-batch_size, batch_size):
            
                batch_texts = texts_dev[i: i + batch_size]
                batch_y = y_dev[i: i + batch_size]
                
                batch_encoding = tokenizer.batch_encode_plus(batch_texts, padding=True, max_length=512,truncation=True)
                input_ids, token_type_ids, attention_mask = batch_encoding["input_ids"], batch_encoding["token_type_ids"], batch_encoding["attention_mask"]
                input_ids = torch.tensor(input_ids).to(device)
                token_type_ids = torch.tensor(token_type_ids).to(device)
                attention_mask = torch.tensor(attention_mask).to(device)
                H = bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)["pooler_output"]
                logits = W(H)
                loss = loss_fn(logits, torch.tensor(batch_y).to(device))
                loss_vals.append(loss.detach().cpu().numpy())
                
                preds = logits.argmax(dim=1)
                acc = (preds==torch.tensor(batch_y).to(device)).float().mean().detach().cpu().numpy()
                accs.append(acc)
                
                if adv:
                    logits_adv = adv(H)
                    batch_y_gender= y_dev_gender[i: i + batch_size]
                    preds_adv = logits_adv.argmax(dim=1)
                    acc = (preds_adv==torch.tensor(batch_y_gender).to(device)).float().mean().detach().cpu().numpy()
                    accs_adv.append(acc)
                
                pbar.update(batch_size)
    bert.train()
    
    return_dict = {"loss": np.mean(loss_vals), "acc": np.mean(accs)}
    if adv:
        return_dict["adv-acc"] =  np.mean(accs_adv)
    return return_dict


pbar = tqdm.tqdm(range(args.iters), ascii=True)
d, d2 = args.__dict__, {"lr": lr, "momentum": momentum, "decay": decay}
new_d = {**d, **d2}
run = wandb.init(reinit=True, project="rlace-finetune-bios", config=new_d, tags=["no-adv" if args.adv==0 else "mlp-adv" if args.mlp_adv else "linear-adv"])


for i in pbar:
    
    optimizer.zero_grad()
    
    idx = np.arange(len(X))
    random.shuffle(idx)
    batch_idx = idx[:10]
    batch_texts, batch_prof = [txts[i] for i in batch_idx], y_prof[batch_idx]
    
    batch_encoding = tokenizer.batch_encode_plus(batch_texts, padding=True, max_length=512,truncation=True)
    input_ids, token_type_ids, attention_mask = batch_encoding["input_ids"], batch_encoding["token_type_ids"], batch_encoding["attention_mask"]
    input_ids = torch.tensor(input_ids).to(device)
    token_type_ids = torch.tensor(token_type_ids).to(device)
    attention_mask = torch.tensor(attention_mask).to(device)
    H = bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)["pooler_output"]
    logits = W(H)
    loss = loss_fn(logits, torch.tensor(batch_prof).to(device))
    if adv:
        batch_g = y_gender[batch_idx]
        loss += loss_fn(adv_clf(H), torch.tensor(batch_g).to(device))
    loss.backward()
    optimizer.step()
    
    loss_vals.append(loss.detach().cpu().numpy().item())
    
    if i % 1000 == 0: #and i > 0:
        return_dict = eval_dev(bert, W, txts_dev, y_dev_prof, device, adv=adv_clf if adv else None, y_dev_gender=y_dev_gender)
        dev_loss, dev_acc = return_dict["loss"], return_dict["acc"]
        if adv:
            adv_acc = return_dict["adv-acc"]
        train_loss = np.mean(loss_vals)
        if dev_loss < best_score:
            best_score = dev_loss.copy()
            
            path = "models/mlp-adv" if args.mlp_adv else "models/linear-adv" if args.adv else "models/no-adv"
            torch.save(W.state_dict(), "{}/W_{}.pt".format(path,args.run_id))
            torch.save(bert.state_dict(), "{}/bert_{}.pt".format(path,args.run_id))
            if adv:
                torch.save(adv_clf.state_dict(), "{}/adv_{}.pt".format(path, args.run_id))

        wandb.log({"train_loss": train_loss, "dev_loss": dev_loss, "best_score": best_score, "dev_acc": dev_acc, "adv_acc": adv_acc if adv else -1})
        #pbar.set_description("Train loss: {:.3f}; Dev loss: {:.3f}; Best Dev loss: {:.3f}; Dev acc: {:.3f}; Dev adv-acc: {:.3f}".format(train_loss, dev_loss, best_score, dev_acc, adv_acc if adv else -1))
        #pbar.refresh() # to show immediately the update
        #time.sleep(0.01)
            
        #print(i, train_loss, dev_loss, best_score)
        loss_vals = []
        
run.finish()