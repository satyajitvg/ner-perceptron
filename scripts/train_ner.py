# -*- coding: utf-8 -*-
"""
@author: Satyajit
"""
import argparse
import os
import sys
lib_path = os.path.abspath('../src/')
sys.path.insert(0, lib_path)
from Perceptron import Perceptron
from utils import parse_file, make_data, get_labels
from ml_utils import evaluate_model, print_confusion_matrix, metrics

if __name__ == '__main__':
    #parse cmd args    
    p = argparse.ArgumentParser()
    p.add_argument("-trainfile", help="training file path", type=str)
    p.add_argument("-testfile", help="testing file path", type=str)
    p.add_argument("-iter", help="number of iterations", type=int)
    p.add_argument("-cf", help="print confusion matrix", choices=("y","n"))
    args = p.parse_args()
    
    #set up training
    labelset = get_labels(args.trainfile)
    train_data = make_data(parse_file(args.trainfile))
    model = Perceptron(args.iter, labelset)    
    weights = model.train(train_data)
    
    #evaluate on test set 
    test_data = make_data(parse_file(args.testfile)) 
    cf_matrix = evaluate_model(test_data, model, labelset)
    if args.cf == "y":
        print_confusion_matrix(cf_matrix)
    prec, recall, fscore = metrics(cf_matrix)
    print "Num Iter: %s\tFscore: %s\tPrec: %s\tRecall: %s"%(args.iter, fscore, prec, recall)



