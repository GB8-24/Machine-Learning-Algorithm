import numpy as np
from typing import List
from classifier import Classifier


class DecisionStump(Classifier):
    def __init__(self, s: int, b: float, d: float):
        self.clf_name = "Decision_stump"
        self.s = s
        self.b = b
        self.d = d

    def train(self, features: List[List[float]], labels: List[int]):
        pass

    def predict(self, features: List[float]) -> List[int]:
        ##################################################
        # TODO: implement "predict"
        ##################################################
        features_array = np.asarray(features)
        num_of_samples, dimensions = features_array.shape
        result = np.zeros(num_of_samples)

        for i in range(num_of_samples):
            if features_array[i][self.d] > self.b:
                result[i] = self.s
            else:
                result[i] = self.s * -1

        return result
