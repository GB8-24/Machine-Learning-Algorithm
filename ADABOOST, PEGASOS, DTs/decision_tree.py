import numpy as np
from typing import List
from classifier import Classifier

class DecisionTree(Classifier):
    def __init__(self):
        self.clf_name = "DecisionTree"
        self.root_node = None

    def train(self, features: List[List[float]], labels: List[int]):
        # init.
        assert(len(features) > 0)
        self.feautre_dim = len(features[0])
        num_cls = np.max(labels)+1

        # build the tree
        self.root_node = TreeNode(features, labels, num_cls)
        if self.root_node.splittable:
            self.root_node.split()

        return

    def predict(self, features: List[List[float]]) -> List[int]:
        y_pred = []
        for feature in features:
            y_pred.append(self.root_node.predict(feature))
        return y_pred

    def print_tree(self, node=None, name='node 0', indent=''):
        if node is None:
            node = self.root_node
        print(name + '{')
        if node.splittable:
            print(indent + '  split by dim {:d}'.format(node.dim_split))
            for idx_child, child in enumerate(node.children):
                self.print_tree(node=child, name= '  '+name+'/'+str(idx_child), indent=indent+'  ')
        else:
            print(indent + '  cls', node.cls_max)
        print(indent+'}')


class TreeNode(object):
    def __init__(self, features: List[List[float]], labels: List[int], num_cls: int):
        self.features = features
        self.labels = labels
        self.children = []
        self.num_cls = num_cls

        count_max = 0
        for label in np.unique(labels):
            if self.labels.count(label) > count_max:
                count_max = labels.count(label)
                self.cls_max = label # majority of current node

        if len(np.unique(labels)) < 2:
            self.splittable = False
        else:
            self.splittable = True

        self.dim_split = None # the dim of feature to be splitted

        self.feature_uniq_split = None # the feature to be splitted


    def split(self):
        def conditional_entropy(branches: List[List[int]]) -> float:
            '''
                branches: C x B array,
                          C is the number of classes,
                          B is the number of branches
                          it stores the number of
                '''
            ########################################################
            # TODO: compute the conditional entropy
            ########################################################
            branches_array = np.asarray(branches)
            C, B = branches.shape
            total_sum = np.sum(branches_array)
            cond_entropy = 0
            weights = np.zeros(C)
            for i in range(C):
                sum_on_columns = np.sum(branches_array[i])
                for j in range(B):
                    prob = branches_array[i][j] / sum_on_columns
                    log_prob = np.log(prob)
                    weights[i] += prob * log_prob

                cond_entropy += (sum_on_columns / total_sum) * weights[i] * -1
            return cond_entropy

        m, n = np.asarray(self.features).shape
        min_entropy = 1000

        for idx_dim in range(len(self.features[0])):
            ############################################################
            # TODO: compare each split using conditional entropy
            #       find the best split
            ############################################################
            features_array = np.asarray(self.features).T
            branches = np.unique(features_array[idx_dim])
            count_trues = np.zeros(len(branches))
            count_falses = np.zeros(len(branches))
            for j in range(m):
                for k in range(len(branches)):
                    if self.labels[j] == 1 and features_array[0][j] == branches[k]:
                        count_trues[k] += 1
                    elif self.labels[j] == 0 and features_array[0][j] == branches[k]:
                        count_falses[k] += 1
            branches_matrix = np.zeros((2, len(branches)))
            branches_matrix[0] = count_trues
            branches_matrix[1] = count_falses
            entropy = conditional_entropy(branches_matrix)
            if min_entropy > entropy:
                min_entropy = entropy
                self.dim_split = idx_dim

                ############################################################
                # TODO: split the node, add child nodes
                ############################################################
                self.split()

        # split the child nodes
        for child in self.children:
            if child.splittable:
                child.split()

        return

    def predict(self, feature: List[int]) -> int:
        if self.splittable:
            # print(feature)
            idx_child = self.feature_uniq_split.index(feature[self.dim_split])
            return self.children[idx_child].predict(feature)
        else:
            return self.cls_max



