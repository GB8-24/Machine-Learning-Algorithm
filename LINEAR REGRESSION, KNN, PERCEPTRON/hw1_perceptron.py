from __future__ import division, print_function

from typing import List, Tuple, Callable

import numpy as np
import scipy
import matplotlib.pyplot as plt


class Perceptron:
    def __init__(self, nb_features=2, max_iteration=10, margin=1e-4):
        """
            Args :
            nb_features : Number of features
            max_iteration : maximum iterations. You algorithm should terminate after this
            many iterations even if it is not converged
            margin is the min value, we use this instead of comparing with 0 in the algorithm
        """

        self.nb_features = 2
        self.w = [0 for i in range(0, nb_features + 1)]
        self.margin = margin
        self.max_iteration = max_iteration

    def train(self, features: List[List[float]], labels: List[int]) -> bool:
        """
            Args  :
            features : List of features. First element of each feature vector is 1
            to account for bias
            labels : label of each feature [-1,1]


            Returns :
                True/ False : return True if the algorithm converges else False.
        """
        ############################################################################
        # TODO : complete this function.
        # This should take a list of features and labels [-1,1] and should update
        # to correct weights w. Note that w[0] is the bias term. and first term is
        # expected to be 1 --- accounting for the bias
        ############################################################################
        for y in range(0, self.max_iteration):
            wold = self.w
            for i in range(0, len(features)):
                iteration = np.dot(self.w, features[i])
                if iteration <= 0:
                    iteration = -1
                else:
                    iteration = 1
                if labels[i] != iteration:
                    temp = [x * labels[i] for x in features[i]]
                    temp = temp / np.linalg.norm(np.asarray(features[i]))
                    self.w = np.add(self.w, temp)
            if np.array_equal(self.w, wold):
                return True

        return False

    def reset(self):
        self.w = [0 for i in range(0, self.nb_features + 1)]

    def predict(self, features: List[List[float]]) -> List[int]:
        '''
            Args  :
            features : List of features. First element of each feature vector is 1
            to account for bias
            iteration=np.dot(self.w,feature[i])

            Returns :
                labels : List of integers of [-1,1]
        '''
        ############################################################################
        # TODO : complete this function.
        # This should take a list of features and labels [-1,1] and use the learned
        # weights to predict the label
        ############################################################################

        predicted_labels = []
        for i in range(0, len(features)):
            iteration = np.dot(self.w, features[i])
            if iteration <= 0:
                iteration = -1
            else:
                iteration = 1
            predicted_labels.extend([iteration])
        return predicted_labels

    def get_weights(self) -> Tuple[List[float], float]:
        return self.w
