import pandas as pd
import numpy as np
import operator

class KNN():
    
    def __init__(self, K):
        self.K  = K
        
    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        
    def square_diff(self,x1,x2):    
        assert len(x1) == len(x2)
        sq_diff = np.square(x2-x1)
        return sq_diff
    
    def root_sum_squared(self,sq_diff):
        return np.sqrt(np.square(sq_diff.sum()))
    
    def evaluate(self, y_hat, y_true):
        correct = 0
        for i in range(len(y_true)):
            if y_true[i] == y_hat[i]:
                correct += 1
        return (correct / float(len(y_true))) * 100.00

    def euclidean_distances(self,x1, x2):
        return np.linalg.norm(x1-x2)
    
    def predict(self,x_test):

        predictions = []
        for i in range(len(x_test)):            
            dist = np.array([self.euclidean_distances(x_test[i], x) for x in self.x_train])
            dist_sorted = dist.argsort()[:self.K]
            neigh_count = {}
            for idx in dist_sorted:
                if self.y_train[idx] in neigh_count:
                    neigh_count[self.y_train[idx]] += 1
                else:
                    neigh_count[self.y_train[idx]] = 1
            sorted_neigh_count = sorted(neigh_count.items(), key=operator.itemgetter(1), reverse=True)
            predictions.append(sorted_neigh_count[0][0])
        return np.array(predictions)