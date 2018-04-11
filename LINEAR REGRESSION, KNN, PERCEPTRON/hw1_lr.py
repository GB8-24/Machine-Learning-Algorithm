from __future__ import division, print_function

from typing import List

import numpy
import scipy


############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################

class LinearRegression:
    weights = []

    def __init__(self, nb_features: int):
        self.nb_features = nb_features

    def train(self, features: List[List[float]], values: List[float]):
        """TODO : Complete this function"""

        features = numpy.asarray(features)
        transpose_features = numpy.transpose(features)

        product = numpy.matmul(transpose_features, features)
        inverse = numpy.linalg.pinv(product)
        with_transpose = numpy.matmul(inverse, transpose_features)
        self.weights = numpy.matmul(with_transpose, numpy.asarray(values)).tolist()

    def predict(self, features: List[List[float]]) -> List[float]:
        """TODO : Complete this function"""
        weights_array = numpy.asarray(self.weights)
        features_array = numpy.asarray(features)
        num_of_samples, num_of_features = features_array.shape
        predicted_value_array = numpy.matmul(features_array, weights_array.reshape(num_of_features, 1))
        return list(predicted_value_array)

    def get_weights(self) -> List[float]:
        return self.weights


class LinearRegressionWithL2Loss:
    weights = []

    def __init__(self, nb_features: int, alpha: float):
        self.alpha = alpha
        self.nb_features = nb_features

    def train(self, features: List[List[float]], values: List[float]):
        """TODO : Complete this function"""
        features = numpy.asarray(features)
        transpose_features = numpy.transpose(features)

        product = numpy.matmul(transpose_features, features)

        rows, col = features.shape

        bias = numpy.identity(col)
        bias = bias * self.alpha

        withalpha = numpy.add(product, bias)

        inverse = numpy.linalg.pinv(withalpha)
        with_transpose = numpy.matmul(inverse, transpose_features)
        self.weights = numpy.matmul(with_transpose, numpy.asarray(values)).tolist()

    def predict(self, features: List[List[float]]) -> List[float]:
        """TODO : Complete this function"""
        weights_array = numpy.asarray(self.weights)
        features_array = numpy.asarray(features)
        num_of_samples, num_of_features = features_array.shape
        predicted_value_array = numpy.matmul(features_array, weights_array.reshape(num_of_features, 1))
        return list(predicted_value_array)

    def get_weights(self) -> List[float]:
        return self.weights


if __name__ == '__main__':
    print(numpy.__version__)
    print(scipy.__version__)
