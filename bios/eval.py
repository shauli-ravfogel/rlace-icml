
import sys
import os
sys.path.append("../../..")
from debias import get_debiasing_projection, get_rowspace_projection

from classifier import CovMaximizer
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
from relaxed_inlp import solve_fantope_relaxation,solve_fantope_relaxation_fr

from sklearn.svm import LinearSVC
from pytorch_revgrad import RevGrad

import sklearn
from sklearn.linear_model import LogisticRegression
import random
import pickle
import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn import neural_network
from gensim.models.keyedvectors import Word2VecKeyedVectors
from gensim.models import KeyedVectors
import numpy as np
from transformers import BertModel, BertTokenizer














def load_bios(group):

    X = np.load("../../../{}_cls.npy".format(group))
    with open("../../../{}.pickle".format(group), "rb") as f:
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





device="cuda:2"

X,y_gender,txts,professions,bios_data =  load_bios("train")
X_dev,y_dev_gender,txts_dev,professions_dev,bios_data_dev =  load_bios("dev")

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert = BertModel.from_pretrained("bert-base-uncased")
prof2ind = {p:i for i,p in enumerate(sorted(set(professions)))}
ind2prof = {i:p for i,p in prof2ind.items()}
y_prof = np.array([prof2ind[p] for p in professions])
y_dev_prof = np.array([prof2ind[p] for p in professions_dev])

W = torch.nn.Linear(768, len(prof2ind))
loss_fn = torch.nn.CrossEntropyLoss()

W.to(device)
bert.to(device)

with open("bert-finetuned-bios.pickle", "rb") as f:
    W_state, bert_state, prof2ind, train_loss, dev_loss = pickle.load(f)
    
print(W_state)
print(bert_state)

torch.save(W_state, "W.pt")
torch.save(bert_state, "bert.pt")

bert.load_state_dict("bert.pt")
bert.eval()
W.load_state_dict("W.pt")
W.eval()

accs = []

for i in tqdm.tqdm(range(0, len(texts_dev)-batch_size, batch_size)):
            
                batch_texts = texts_dev[i: i + batch_size]
                batch_y = y_dev_prof[i: i + batch_size]
                
                batch_encoding = tokenizer.batch_encode_plus(batch_texts, padding=True, max_length=512,truncation=True)
                input_ids, token_type_ids, attention_mask = batch_encoding["input_ids"], batch_encoding["token_type_ids"], batch_encoding["attention_mask"]
                input_ids = torch.tensor(input_ids).to(device)
                token_type_ids = torch.tensor(token_type_ids).to(device)
                attention_mask = torch.tensor(attention_mask).to(device)
                H = bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)["pooler_output"]
                logits = W(H)
                
                preds = logits.argmax(dim=1)
                acc = (preds==batch_y).float().mean().detach().cpu().numpy()
                accs.append(acc)
                
print(np.mean(accs))