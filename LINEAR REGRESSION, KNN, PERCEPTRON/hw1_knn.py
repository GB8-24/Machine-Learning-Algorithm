from __future__ import division, print_function

from typing import List, Callable

import numpy
import scipy


############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################

class KNN:
    features = []
    labels = []

    def __init__(self, k: int, distance_function) -> float:
        self.k = k
        self.distance_function = distance_function

    def train(self, features: List[List[float]], labels: List[int]):
        # combine both the lists
        self.features = features
        self.labels = labels

    def predict(self, features: List[List[float]]) -> List[int]:
        # how knn is working
        l = []
        c = numpy.column_stack((self.features, self.labels))
        print(c)
        for x in features:
            dist = []
            temp2=[]
            for y in c:
                dist = numpy.append(self.distance_function(x, y[0:self.k]), [y[len(c[0]) - 1]])

                temp2.append(dist)

            temp2=sorted(temp2, key=lambda x: x[0])
            temp2 = numpy.array(temp2[0:self.k])
            for i in temp2:
                count_one = 0
                count_zero = 0
                temp = list(temp2[:, 1])
                for x in temp:
                    if x == 1:
                        count_one += 1
                    else:
                        count_zero += 1
                if count_one >= count_zero:
                    l.extend([1])
                else:
                    l.extend([0])
        return l


if __name__ == '__main__':
    print(numpy.__version__)
    print(scipy.__version__)
