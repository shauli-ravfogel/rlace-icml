import sys
import os

sys.path.append("..")
sys.path.append("../")

from debias import get_debiasing_projection, get_rowspace_projection

from sklearn.decomposition import PCA
import seaborn as sn

from collections import defaultdict
from sklearn.manifold import TSNE
import torch
from sklearn.linear_model import SGDClassifier
from rlace import solve_adv_game

import random
import pickle
import numpy as np


def set_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def load_glove():
    with open("glove/glove_data/data.pickle", "rb") as f:
        data_dict = pickle.load(f)
        X, y, words_train = data_dict["train"]
        X_dev, y_dev, words_dev = data_dict["dev"]
        X_test, y_test, words_test = data_dict["test"]
        X, y = X[y > -1], y[y > -1]
        X_dev, y_dev = X_dev[y_dev > -1], y_dev[y_dev > -1]
        X_test, y_test = X_test[y_test > -1], y_test[y_test > -1]
        return (X, y, words_train), (X_dev, y_dev, words_dev), (X_test, y_test, words_test)


def run_inlp(X, y, X_dev, y_dev, num_iters=25):
    clf = SGDClassifier
    LOSS = "log"
    ALPHA = 1e-5
    TOL = 1e-4
    ITER_NO_CHANGE = 50
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
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'Serif'

    if method == "pca":
        pca = PCA(n_components=2)
        M = 6000
    elif method == "tsne":
        pca = TSNE(n_components=2, learning_rate="auto", init="pca")
        M = 2000

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
    print("here")
    (X, y, words_train), (X_dev, y_dev, words_dev), (X_test, y_test, words_test) = load_glove()

    rlace_projs = defaultdict(dict)
    inlp_projs = defaultdict(dict)


    for random_run in range(5):
        os.makedirs("glove/plots/original/pca/run={}".format(random_run), exist_ok=True)
        os.makedirs("glove/plots/original/tsne/run={}".format(random_run), exist_ok=True)
        os.makedirs("glove/plots/inlp/pca/run={}".format(random_run), exist_ok=True)
        os.makedirs("glove/plots/inlp/tsne/run={}".format(random_run), exist_ok=True)
        os.makedirs("glove/plots/rlace/pca/run={}".format(random_run), exist_ok=True)
        os.makedirs("glove/plots/rlace/tsne/run={}".format(random_run), exist_ok=True)
        os.makedirs("glove/interim/rlace/run={}".format(random_run), exist_ok=True)
        os.makedirs("glove/interim/inlp/run={}".format(random_run), exist_ok=True)

        set_seeds(random_run)

        # run inlp

        Ps_nullsapce_inlp, ws_inlp_normalized, accs_inlp = run_inlp(X, y, X_dev, y_dev, num_iters=21)

        with open("glove/interim/inlp/run={}/Ps_inlp.pickle".format(random_run), "wb") as f:
            pickle.dump((Ps_nullsapce_inlp, accs_inlp), f)

        plot_pca(X_dev, y_dev, "glove/plots/original/pca/run={}".format(random_run), "original", method="pca")
        plot_pca(X_dev, y_dev, "glove/plots/original/tsne/run={}".format(random_run), "original", method="tsne")

        for i, P in enumerate(Ps_nullsapce_inlp):
            plot_pca(X_dev @ P, y_dev, "glove/plots/inlp/pca/run={}".format(random_run), "Projected, Rank={}".format(i + 1),
                     method="pca")
            plot_pca(X_dev @ P, y_dev, "glove/plots/inlp/tsne/run={}".format(random_run), "Projected, Rank={}".format(i + 1),
                     method="tsne")

        # run adversarial game

        ranks = [1, 2, 4, 8, 12, 16, 20]
        DEVICE = "cpu"

        Ps_rlace, accs_rlace = [], [1.0]
        optimizer_class = torch.optim.SGD
        optimizer_params_P = {"lr": 0.001, "weight_decay": 1e-5, "momentum": 0.9}
        optimizer_params_predictor = {"lr": 0.001, "weight_decay": 1e-5, "momentum": 0.9}
        for rank in ranks:
            output = solve_adv_game(X, y, X, y, rank=rank, device=DEVICE, out_iters=60000,
                                    optimizer_class=optimizer_class, optimizer_params_P=optimizer_params_P,
                                    optimizer_params_predictor=optimizer_params_predictor, epsilon=0.002,
                                    batch_size=64)
        P = output["P"]
        Ps_rlace.append(P)
        accs_rlace.append(output["score"])

        plot_pca(X_dev @ P, y_dev, "glove/plots/rlace/pca/run={}".format(random_run), "Projected, Rank={}".format(rank),
                 method="pca")
        plot_pca(X_dev @ P, y_dev, "glove/plots/rlace/tsne/run={}".format(random_run), "Projected, Rank={}".format(rank),
                 method="tsne")

    rlace_projs[random_run] = {"Ps": Ps_rlace, "accs": accs_rlace, "ranks": ranks}
    inlp_projs[random_run] = {"Ps": Ps_nullsapce_inlp, "accs": accs_inlp}

    with open("glove/interim/rlace/projs.pickle", "wb") as f:
        pickle.dump(rlace_projs, f)

    with open("glove/interim/inlp/projs.pickle", "wb") as f:
        pickle.dump(inlp_projs, f)