# -*- coding: utf-8 -*-
"""
@author: Satyajit
misc. utility functions 
"""
from NerFeatures import NerFeatures


def detect_ftype(path):
    """
    hacky stuff to detect if file is esp or ned
    spanish & dutch files are in different format!
    esp: label is in pos 1 (after line is split by space)
    ned: label is in pos 2 (after line is split by space)
    Note: Will not work if file names are changed from ned.* or esp.*
    """
    import os
    parts = os.path.split(path)
    if "ned" in parts[1]:
        return "ned", 2
    elif "esp" in parts[1]:
        return "esp", 1
    else:
        return None, None
    
def parse_file(path):
    """
    parse data file from CoNLL-2002. 
    Returns a list of sentences
    Every sentence is a list of (word, label) tuples
    """
    ftype, pos = detect_ftype(path)
    sentences = []
    curr_sentence = []
    with open(path) as infile:
        for line in infile:
            line = line.decode("utf-8", "ignore").strip()
            if line == "":
                #new sentence starts
                sentences.append(curr_sentence)
                curr_sentence = []
                continue
            toks = line.split()
            if len(toks) != pos + 1:
                #This condition should Ideally not be required
                #But there are 4 lines in esp.testa that have some weird unicode
                continue
            curr_sentence.append((toks[0], toks[pos]))
        sentences.append(curr_sentence)
    return sentences
    
def get_labels(path):
    """Return list of entity labels"""
    labels = set()
    ftype, pos = detect_ftype(path)
    with open(path) as infile:
        for line in infile:
            line = line.decode("utf-8", "ignore").strip()
            toks = line.split(" ")
            if len(toks) == pos+1:
                labels.add(toks[pos])
    return labels
    
def make_data(sentences):
    """make data that can be used for training or scoring perceptron"""
    data = []
    for sentence in sentences:
        words = [tup[0] for tup in sentence]
        labels = [tup[1] for tup in sentence]
        for idx, word in enumerate(words):
            nerfeat = NerFeatures()
            features = nerfeat.word_features(idx,words)
            data.append((features, labels[idx]))
    return data 
            
            

#
#testing
#

if __name__ == '__main__':
    import sys
    sentences = parse_file(sys.argv[1])
    print len(sentences)
    print get_labels(sys.argv[1])