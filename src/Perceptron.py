# -*- coding: utf-8 -*-
"""
@author: Satyajit
Perceptron classifier 
"""
from __future__ import division
from operator import itemgetter
from collections import defaultdict
from random import shuffle

class Perceptron(object):
    """
    Average Perceptron Classifier
    Based on http://www.ciml.info/dl/v0_9/ciml-v0_9-ch03.pdf
    Modified for multiclass classifcation
    """
    def __init__(self, max_iter, labels):        
        self.max_iter = max_iter
        self.labels = frozenset(labels)
        self.weights = defaultdict(lambda: defaultdict(float))
        
        #clever averaging requires keeping track of how long weights survived        
        self.timestamps = defaultdict(lambda: defaultdict(int))
        self.accum_weights = defaultdict(lambda: defaultdict(float))
        self.clock = 1
        
    def predict(self, features):
        """
        compute inner product between weights and input feature vector
        """
        scores = {label:0 for label in self.labels}   
        for feature, value in features.items():
            if feature in self.weights:
                for label, weight in self.weights[feature].items():
                    scores[label] += value * weight
        return max(scores.iteritems(), key=itemgetter(1))
            
    def __update_helper(self, feature, label, value):
        """
        helper to increment or penalize weights
        and update accumulated weights
        """
        assert (value == 1 or value == -1)
        wt = self.weights[feature].get(label, 0.0)
        #bring accumulated weights up to date
        self.accum_weights[feature][label] += wt * (self.clock - self.timestamps[feature][label])
        #increment or penalize        
        self.weights[feature][label] += value
        #set ts to current clock
        self.timestamps[feature][label] = self.clock
        
    def _update(self, known_label, pred_label, features):
        """
        update weights if misclassified.
        keep track of accumulated weights and survival times
        """
        assert known_label != pred_label
        for feature, value in features.items():
            self.__update_helper(feature, known_label, +1)
            self.__update_helper(feature, pred_label, -1)
    
    def _average(self):
        "average weights from all iterations"""
        for feature, class_weights in self.weights.items():
            avg_wt = {}
            for class_label, weight in class_weights.items():
                acc_wt = self.accum_weights[feature][class_label]
                #accumulate any remaining unchanged weights  
                acc_wt += (self.clock - self.timestamps[feature][class_label]) * weight
                avg_wt[class_label] = acc_wt/self.clock
            self.weights[feature] = avg_wt
            
        
    def debug(self):
        for f, w in self.weights.items():
            for l, v in w.items():
                print "%s:%s,%s\t"%(f,l,v),
            print
            
    def train(self, train_data):
        shuffle(train_data)
        for _ in range(self.max_iter):
            for features, known_label in train_data:
                pred_label, _ = self.predict(features)
                self.clock += 1
                if pred_label != known_label:
                    self._update(known_label, pred_label, features)
        self._average()
        return self.weights
            
    
            
#
# Testing code
#            
if __name__ == '__main__':
    x1 = {'b':1, 'f1':1, 'f2':0}
    x2 = {'b':1, 'f1':0, 'f2':1}
    x3 = {'b':1, 'f1':0.9, 'f2':0.1}
    x4 = {'b':1, 'f1':0.1, 'f2':0.9, 'f3':0.7}
    trainData = [(x1, "c1"), (x2, "c2"), (x4, "c2"), (x3, "c1")]
    model = Perceptron(1, ("c1","c2"))
    model.train(trainData)
    
            
            
        
        
        
    