
import numpy as np
from sklearn.utils.extmath import randomized_svd
from numpy import linalg

# an abstract class for linear classifiers

class Classifier(object):

    def __init__(self):

        pass

    def train(self, X_train: np.ndarray, Y_train: np.ndarray, X_dev: np.ndarray, Y_dev: np.ndarray) -> float:
        """
        :param X_train:
        :param Y_train:
        :param X_dev:
        :param Y_dev:
        :return: accuracy score on the dev set
        """
        raise NotImplementedError

    def get_weights(self) -> np.ndarray:
        """
        :return: final weights of the model, as np array
        """

        raise NotImplementedError


class CovMaximizer(object):

    def __init__(self):

        pass

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray) -> float:
        """
        :param X_train:
        :param Y_train:
        :param X_dev:
        :param Y_dev:
        :return: accuracy score on the dev set
        """
        
        eigvals, eigvecs =  np.linalg.eigh((1./len(X_train)**2) * X_train.T@Y_train@Y_train.T@X_train)
        """
        U, Sigma, VT = randomized_svd((1/np.sqrt(len(X_train)))*cov,
                              n_components=2,
                              n_iter=15,
                              random_state=None)
        """
        self.w = eigvecs[:,-1]
        self.coef_ = self.w
        print(self.w.shape)
        
    def score(self, X_dev, Y_dev):
    
        
        cov = (1./len(X_dev)**2) * (X_dev.T@Y_dev@Y_dev.T@X_dev)
        return self.w.T@cov.T@cov@self.w
        
    def get_weights(self) -> np.ndarray:
        """
        :return: final weights of the model, as np array
        """

        return self.w

class SKlearnClassifier(Classifier):

    def __init__(self, m):

        self.model = m

    def train_network(self, X_train: np.ndarray, Y_train: np.ndarray, X_dev: np.ndarray, Y_dev: np.ndarray) -> float:

        """
        :param X_train:
        :param Y_train:
        :param X_dev:
        :param Y_dev:
        :return: accuracy score on the dev set / Person's R in the case of regression
        """

        self.model.fit(X_train, Y_train)
        score = self.model.score(X_dev, Y_dev)
        return score

    def get_weights(self) -> np.ndarray:
        """
        :return: final weights of the model, as np array
        """

        w = self.model.coef_
        if len(w.shape) == 1:
                w = np.expand_dims(w, 0)

        return w
