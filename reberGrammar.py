# © Christian Herta, Davide Posillipo
# Adapted from http://christianherta.de/lehre/dataScience/machineLearning/neuralNetworks/reberGrammar.php, 
# for the Sequences class at Data Science Retreat, 2020, Berlin. 
# Originally in Python 2. 

import numpy as np

class reberGrammar:
    
    def __init__(self):

        self.chars='BTSXPVE'

        self.graph = [[(1,5),('T','P')] , [(1,2),('S','X')], \
                   [(3,5),('S','X')], [(6,),('E')], \
                   [(3,2),('V','P')], [(4,5),('V','T')] ]
        
        self.emb_chars = "TP"


    def in_grammar(self, word):
        if word[0] != 'B':
            return False
        node = 0    
        for c in word[1:]:
            transitions = self.graph[node]
            try:
                node = transitions[0][transitions[1].index(c)]
            except ValueError: # using exceptions for flow control in python is common
                return False
        return True        

    
    def sequenceToWord(self, sequence):
        """
        converts a sequence (one-hot) in a reber string
        """
        reberString = ''
        for s in sequence:
            index = np.where(s==1.)[0][0]
            reberString += self.chars[index]
        return reberString

    
    def generateSequences(self, minLength):
        while True:
            inchars = ['B']
            node = 0
            outchars = []    
            while node != 6:
                transitions = self.graph[node]
                i = np.random.randint(0, len(transitions[0]))
                inchars.append(transitions[1][i])
                outchars.append(transitions[1])
                node = transitions[0][i]
            if len(inchars) > minLength:  
                return inchars, outchars


    def get_one_example(self, minLength):
        inchars, outchars = self.generateSequences(minLength)
        inseq = []
        outseq= []
        for i,o in zip(inchars, outchars): 
            inpt = np.zeros(7)
            inpt[self.chars.find(i)] = 1.     
            outpt = np.zeros(7)
            for oo in o:
                outpt[self.chars.find(oo)] = 1.
            inseq.append(inpt)
            outseq.append(outpt)
        return inseq, outseq


    def get_char_one_hot(self, char):
        char_oh = np.zeros(7)
        for c in char:
          char_oh[self.chars.find(c)] = 1.
        return [char_oh] 
    

    def get_n_examples(self, n, minLength=10):
        examples = []
        for i in range(n):
            examples.append(self.get_one_example(minLength))
        return examples

    
    def get_one_embedded_example(self, minLength=10):
        i, o = self.get_one_example(minLength)
        emb_char = self.emb_chars[np.random.randint(0, len(self.emb_chars))]
        new_in = self.get_char_one_hot(('B',))
        new_in += self.get_char_one_hot((emb_char,))
        new_out= self.get_char_one_hot(self.emb_chars)
        new_out+= self.get_char_one_hot('B',)
        new_in += i
        new_out += o
        new_in += self.get_char_one_hot(('E',))
        new_in += self.get_char_one_hot((emb_char,))
        new_out += self.get_char_one_hot((emb_char, ))
        new_out += self.get_char_one_hot(('E',))
        return new_in, new_out

    
    def get_n_embedded_examples(self, n, minLength=10):
        examples = []
        for i in range(n):
            examples.append(self.get_one_embedded_example(minLength))
        return examples