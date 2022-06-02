import sys
import os

sys.path.append("../../")
sys.path.append("../")
sys.path.append("/../")

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
from rlace import solve_adv_game

from sklearn.svm import LinearSVC

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
import warnings
import argparse


def set_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def load_bios(group, finetune_mode, seed=None):
    if finetune_mode not in ["no-adv", "mlp-adv", "linear-adv"]:
        X = np.load("bios_data/{}_cls.npy".format(group))
    else:
        X = np.load("encodings/{}/{}_{}_cls.npy".format(finetune_mode, group, seed))
    with open("bios_data/{}.pickle".format(group), "rb") as f:
        bios_data = pickle.load(f)
        Y = np.array([1 if d["g"] == "f" else 0 for d in bios_data])
        professions = np.array([d["p"] for d in bios_data])
        txts = [d["hard_text_untokenized"] for d in bios_data]
        random.seed(0)
        np.random.seed(0)
        X, Y, professions, txts, bios_data = sklearn.utils.shuffle(X, Y, professions, txts, bios_data)
        X = X[:]
        Y = Y[:]

    return X, Y, txts, professions, bios_data


def run_inlp(X, y, X_dev, y_dev, num_iters=25):
    clf = SGDClassifier
    LOSS = "log"
    ALPHA = 1e-5
    TOL = 1e-4
    ITER_NO_CHANGE = 25
    params = {"loss": LOSS, "fit_intercept": True, "max_iter": 3000000, "tol": TOL, "n_iter_no_change": ITER_NO_CHANGE,
              "alpha": ALPHA, "n_jobs": 64}

    input_dim = X_dev.shape[1]

    P_inlp, accs_inlp, ws_inlp_normalized = get_debiasing_projection(clf, params, num_iters, X_dev.shape[1], True, -1,
                                                                     X, y, X_dev, y_dev, by_class=False,
                                                                     Y_train_main=None, Y_dev_main=None, dropout_rate=0)

    Ps_nullsapce = []
    for i in range(1, num_iters):
        P = np.eye(X.shape[1]) - (ws_inlp_normalized[:i]).T @ ws_inlp_normalized[:i]
        Ps_nullsapce.append(P)

    for P, w in zip(Ps_nullsapce, ws_inlp_normalized):
        assert np.allclose(X @ P @ w, np.zeros(X.shape[0]))

    return Ps_nullsapce, ws_inlp_normalized, accs_inlp


def plot_pca(X, y, path, title, method="pca"):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        import matplotlib.pyplot as plt
        plt.rcParams['font.family'] = 'Serif'

        if method == "pca":
            pca = PCA(n_components=2)
            M = 6000
        elif method == "tsne":
            pca = TSNE(n_components=2, learning_rate="auto", init="pca")
            M = 1500

        X_proj = pca.fit_transform(X[:M])
        ax = plt.axes()
        ax.set_axisbelow(True)
        ax.yaxis.grid(color='gray', linestyle='dashed')
        ax.xaxis.grid(color='gray', linestyle='dashed')

        y_text = ["Female-biased" if yy == 1 else "Male-biased" for yy in y]
        plt1 = sn.scatterplot(X_proj[:, 0], X_proj[:, 1], hue=y_text[:M])
        plt.legend(fontsize=19)
        ax.set_title('{}'.format(title), fontsize=25)
        ax.figure.savefig("{}/{}.pdf".format(path, title), dpi=400)
        plt.clf()


def get_svd(X):
    D, U = np.linalg.eigh(X)
    return U, D


if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)

    parser = argparse.ArgumentParser(description="An argparse example")
    parser.add_argument('--device', type=int, default=-1, required=False)
    parser.add_argument('--run_id', type=int, default=-1, required=True)
    parser.add_argument('--do_inlp', type=int, default=1, required=True)
    parser.add_argument('--do_rlace', type=int, default=1, required=True)
    parser.add_argument('--ranks', type=str, default="[1,50,100]", required=False)
    parser.add_argument('--finetune_mode', type=str, default="none", required=True)

    args = parser.parse_args()
    ranks = eval(args.ranks)

    rlace_projs = defaultdict(dict)
    inlp_projs = defaultdict(dict)
    device = "cpu" if args.device == -1 else "cuda:{}".format(args.device)
    finetune_mode = args.finetune_mode

    X, y, txts, professions, bios_data = load_bios("train", finetune_mode, args.run_id)
    X, y = X[:100000], y[:100000]
    X_dev, y_dev, txts_dev, professions_dev, bios_data_dev = load_bios("dev", finetune_mode, args.run_id)

    for random_run in [args.run_id]:
        os.makedirs("plots/original/pca/run={}".format(random_run), exist_ok=True)
        os.makedirs("plots/original/tsne/run={}".format(random_run), exist_ok=True)
        os.makedirs("plots/inlp/pca/run={}".format(random_run), exist_ok=True)
        os.makedirs("plots/inlp/tsne/run={}".format(random_run), exist_ok=True)
        os.makedirs("plots/rlace/pca/run={}".format(random_run), exist_ok=True)
        os.makedirs("plots/rlace/tsne/run={}".format(random_run), exist_ok=True)
        os.makedirs("interim/rlace/run={}".format(random_run), exist_ok=True)
        os.makedirs("interim/inlp/run={}".format(random_run), exist_ok=True)

        os.makedirs("plots/{}/original/pca/run={}".format(finetune_mode, random_run), exist_ok=True)
        os.makedirs("plots/{}/original/tsne/run={}".format(finetune_mode, random_run), exist_ok=True)
        os.makedirs("plots/{}/inlp/pca/run={}".format(finetune_mode, random_run), exist_ok=True)
        os.makedirs("plots/{}/inlp/tsne/run={}".format(finetune_mode, random_run), exist_ok=True)
        os.makedirs("plots/{}/rlace/pca/run={}".format(finetune_mode, random_run), exist_ok=True)
        os.makedirs("plots/{}/rlace/tsne/run={}".format(finetune_mode, random_run), exist_ok=True)
        os.makedirs("interim/{}/rlace/run={}".format(finetune_mode, random_run), exist_ok=True)
        os.makedirs("interim/{}/inlp/run={}".format(finetune_mode, random_run), exist_ok=True)

        set_seeds(random_run)

        # run inlp

        if args.do_inlp == 1:
            Ps_nullsapce_inlp, ws_inlp_normalized, accs_inlp = run_inlp(X, y, X_dev, y_dev, num_iters=101)

            with open("interim/{}/inlp/run={}/Ps_inlp.pickle".format(finetune_mode, random_run), "wb") as f:
                pickle.dump((Ps_nullsapce_inlp, accs_inlp), f)

        # run adversarial game

        Ps_rlace, accs_rlace = [], [1.0]
        if not args.do_rlace == 1:
            exit()

        optimizer_class = torch.optim.SGD
        optimizer_params_P = {"lr": 0.005, "weight_decay": 1e-4, "momentum": 0.0}
        optimizer_params_predictor = {"lr": 0.005, "weight_decay": 1e-4, "momentum": 0.9}

        for rank in ranks:

            output = solve_adv_game(X, y, X, y, rank=rank, device=DEVICE, out_iters=60000,
                                        optimizer_class=optimizer_class, optimizer_params_P=optimizer_params_P,
                                        optimizer_params_predictor=optimizer_params_predictor, epsilon=0.002,
                                        batch_size=64)

            P = output["P"]
            Ps_rlace.append(P)
            accs_rlace.append(output["score"])

            with open("interim/{}/rlace/run={}/Ps_rlace.pickle".format(finetune_mode, random_run), "wb") as f:
                pickle.dump((Ps_rlace, accs_rlace), f)

                # rlace_projs[random_run] = {"Ps": Ps_rlace, "accs": accs_rlace,  "ranks": ranks}
        # inlp_projs[random_run] = {"Ps": Ps_nullsapce_inlp, "accs": accs_inlp}

        # with open("interim/rlace/projs.pickle", "wb") as f:
        #        pickle.dump(rlace_projs, f)

        # with open("interim/inlp/projs.pickle", "wb") as f:
        #        pickle.dump(inlp_projs, f)