from typing import List

import numpy as np


def mean_squared_error(y_true: List[float], y_pred: List[float]) -> float:
    assert len(y_true) == len(y_pred)
    mse = np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)
    return mse


def f1_score(real_labels: List[int], predicted_labels: List[int]) -> float:
    assert len(real_labels) == len(predicted_labels)
    # real_labels = set(real_labels)
    # predicted_labels = set(predicted_labels)

    real_labels_array = np.asarray(real_labels)
    predicted_labels_array = np.asarray(predicted_labels)

    tp = np.sum(np.logical_and(real_labels_array == 1, predicted_labels_array == 1))
    fp = np.sum(np.logical_and(real_labels_array == 0, predicted_labels_array == 1))
    fn = np.sum(np.logical_and(real_labels_array == 1, predicted_labels_array == 0))

    if tp > 0:
        precision = float(tp) / (tp + fp)
        recall = float(tp) / (tp + fn)
        f1score = 2 * (precision * recall) / (precision + recall)
        return f1score
    else:
        return 0


def polynomial_features(features: List[List[float]], k: int) -> List[List[float]]:
    rows, columns = np.asarray(features).shape
    for i in range(1, columns):
        temp_k = []
        for j in range(rows):
            temp_k.append(features[j][i] ** k)
        temp_array = np.asarray(temp_k).reshape(1, rows)
        temp_array_transpose = np.transpose(temp_array)
        features = np.append(features, temp_array_transpose, 1)
    return features


def euclidean_distance(point1: List[float], point2: List[float]) -> float:
    return np.sqrt(np.sum(np.subtract(np.asarray(point1), np.asarray(point2)) ** 2))


def inner_product_distance(point1: List[float], point2: List[float]) -> float:
    return np.inner(np.asarray(point1), np.asarray(point2))


def gaussian_kernel_distance(
        point1: List[float], point2: List[float]
) -> float:
    euc = -0.5 * (np.sum(np.subtract(np.asarray(point1), np.asarray(point2)) ** 2))
    gkd = -1 * np.exp(euc)
    return gkd


class NormalizationScaler:
    def __init__(self):
        pass

    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        a = np.asarray(features)
        norm = np.nan_to_num(np.divide(a, np.diagonal(np.sqrt(np.inner(a, a))).reshape(len(a), 1))).tolist()
        return norm


class MinMaxScaler:
    def __init__(self):
        pass

    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        features = np.asarray(features)
        features = ((features - features.min(axis=0)) / (features.max(axis=0) - features.min(axis=0))).tolist()
        return features
