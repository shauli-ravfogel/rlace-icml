This repository contains the code for the experiments included in the paper [Linear Adversarial Concept Erasure](https://arxiv.org/abs/2201.12091), accepted as a long paper in ICML 2022.
In the paper, we formulate the problem of identifying and neutralizing *concept subspaces* -- linear subspaces within the representation space that capture a given concept, such as gender.

# Algorithm

the file `rlace.py` contains an implementation of Relaxed Linear Adversarial Concept Erasure (R-LACE). 
Given a dataset `X` of dense representations and labels `z` for some concept (e.g. gender), the method identifies a rank-`k` subsapce whose neutralization (suing an othogonal projection matrix) prevents linear classifiers from recovering the concept from the representations. 

The method relies on a relaxed and constrained version of a minimax game between a predictor that aims to predict `z` and a projection matrix `P` that is optimized to prevent the prediction.
### How to run
A simple running example is provided within `rlace.py`.

#### Parameters
The main method, `solve_adv_game`, receives several arguments, among them:

- `rank`: the rank of the neutralized subspace. `rank=1` is emperically enough to prevent linear prediction in binary classification problem.

- `epsilon`: stopping criterion for the adversarial game. Stops if abs(acc - majority_acc) < epsilon.

- `optimizer_class`: torch.optim optimizer

- `optimizer_params_predictor / optimizer_params_P`: parameters for the optimziers of the predictor and the projection matrix, respectively.


#### Running example:

```
num_iters = 50000
rank=1
optimizer_class = torch.optim.SGD
optimizer_params_P = {"lr": 0.003, "weight_decay": 1e-4}
optimizer_params_predictor = {"lr": 0.003,"weight_decay": 1e-4}
epsilon = 0.001 # stop 0.1% from majority acc
batch_size = 256

output = solve_adv_game(X_train, y_train, X_dev, y_dev, rank=rank, device="cpu", out_iters=num_iters, optimizer_class=optimizer_class,
optimizer_params_P =optimizer_params_P, optimizer_params_predictor=optimizer_params_predictor, epsilon=epsilon,batch_size=batch_size)
```

**Optimization**: Even though we run a concave-convex minimax game, which is generallly "well-behaved", optimziation with alternate SGD is still not completely straightforward, and may require some tuning of the optimizers. Accuracy is also not expected to monotonously decrease in optimization; we return the projection matrix which performed best along the entire game. In all experiments on binary classification problems, we identified a projection matrix that neutralizes a rank-1 subspace and decreases classification accuracy to near-random (50%).

#### Using the projection:


`output` that is returned from `solve_adv_game` is a dictionary, that contains the following keys:

1. `score`: final accuracy of the predictor on the projected data.

2. `P_before_svd`: the final approximate projection matrix, before SVD that guarantees it's a proper orthogonal projection matrix.

3. `P`: a proper orthogonal matrix that neutralizes a rank-`k` subspace. 

The ``clean" vectors are given by `X.dot(output["P"])`.


# Experiments

The directories `glove` and `bios` contain the experiments on neutralization gender information form GloVe embeddings and from BERT representations of the Bias in Bios dataset, respectively.

To run:

```
python3 glove/run_glove.py 
sh bios/finetune.sh
sh bios/run_rlace.sh
```

And then run the analysis notebooks to replicate the experiments reported in the paper.

### Data and Models
The datasets used in the experiments, as well as the trained models and projection matrices, are available [here](https://nlp.biu.ac.il/~ravfogs/rlace-cr/).
