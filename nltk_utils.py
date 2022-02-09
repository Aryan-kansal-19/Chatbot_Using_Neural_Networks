from re import A
from nltk.stem import LancasterStemmer
import numpy as np

stemmer = LancasterStemmer()

from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')

def tokenize(sentence):
    return tokenizer.tokenize(sentence)
    

def lemma(word):
    word = word.lower()
    return stemmer.stem(word)

def bag_of_words(tokized_sent, words):
    bag =[]
    bag = [0 for idx in range(len(words))]
    sent = [lemma(a) for a in tokized_sent]
    

    for idx, w in enumerate(words):
        if w in sent:
            bag[idx] = 1
            
    
    return bag

