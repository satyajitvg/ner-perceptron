# -*- coding: utf-8 -*-
"""
@author: Satyajit
"""


class NerFeatures:
    """
    Common features for NER
    Features are sparse and implemented as a dict
    Feature values are always 1 (binary true features only)
    """
    def __init__(self):
        self.features = {}
        
    def _addfeature(self, k1, k2):
        """
        add k1_k2 to to feature dict        
        """
        fname = str(k1) +"_"+ str(k2)
        self.features[fname] = 1
    
    def normalize(self, word):
        if word.isdigit() and len(word) == 4:
            return "<YEAR>"
        elif any(s.isdigit() for s in word):
            return "<WORDDIGIT>"
        else:
            return word
            
        
    def word_features(self, word_pos, context, window=2):
        """
        Get features for a word.
        word is indexed by word_pos
        context can be the sentence where word occured
        """
        context_sz = len(context)
        word = context[word_pos]     
        #add bias
        self._addfeature("<BIAS>","")
        
        #word
        self._addfeature("<WORD>",word)
        
        #word prefix
        self._addfeature("<PREFIX>", word[:3])
        
        #word sufffix
        self._addfeature("<SUFFIX>", word[-3:])
        
        #pev and next word in window  
        for idx in range(1, window+1):   
            #prev word
            if word_pos > idx - 1:
                self._addfeature("<PREVWORD>%s"%idx , context[word_pos-idx])
            else:
                self._addfeature("<PREVWORD>%s"%idx, "<ABSENT>")
            #next word
            if word_pos < context_sz - idx:
                self._addfeature("<NEXTWORD>%s"%idx, context[word_pos+idx])
            else:
                self._addfeature("<NEXTWORD>%s"%idx, "<ABSENT>")
            
        #starts with upper case
        if word[0].isupper():
            self._addfeature("<STARTSCAP>","Y")
        else:
            self._addfeature("<STARTSCAP>","N")
            
        #all caps
        if word.isupper():
            self._addfeature("<ALLCAP>","Y")
        else:
            self._addfeature("<ALLCAP>","N")   
            
        return self.features
    
    