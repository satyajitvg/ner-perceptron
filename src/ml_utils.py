# -*- coding: utf-8 -*-
"""
@author: Satyajit
ML utils
"""
from __future__ import division
from operator import itemgetter

def evaluate_model(data, model, labelset):
    """
    calculate confusion matrix for data using model
    model must implement predict method
    """
    confusion_matrix = {}
    for l in labelset:
        counts = {x:0 for x in labelset}
        confusion_matrix[l] = counts          
    for features, known_label in data:
        pred_label, score = model.predict(features)
        confusion_matrix[pred_label][known_label]+=1
    return confusion_matrix

def print_confusion_matrix(confusion_matrix):
    """pretty print CF matrix"""
    key_order = sorted(confusion_matrix.keys())
    print "\t"+'\t'.join('known_' + x for x in key_order)
    for pred_k in key_order:
        known_k = confusion_matrix[pred_k]
        sorted_known = sorted(known_k.items(), key = itemgetter(0))
        print '%s\t%s'%('predicted_'+pred_k ,'\t'.join(str(x[1]) for x in sorted_known))
        
        
def is_entity(label): #hacky!
    return label != "O"
        
def metrics(confusion_matrix):
    """
    prec, recall and fscore according to 
    http://www.cnts.ua.ac.be/conll2002/pdf/15558tjo.pdf    
    """ 
    correct_entities = 0
    pred_entities = 0
    known_entities = 0
    for pred_label, data in confusion_matrix.items():
        for known_label, count in data.items():
            if known_label == pred_label and is_entity(known_label):
                correct_entities += count
            if is_entity(known_label):
                known_entities += count
            if is_entity(pred_label):
                pred_entities += count
    prec = correct_entities/pred_entities
    recall = correct_entities/known_entities
    fscore = 2*prec*recall/(prec + recall)
    return prec, recall, fscore
   

            
    
    
    
