import pandas as pd
import json 
from nltk_utils import tokenize, lemma, bag_of_words
import numpy as np
import tflearn
import tensorflow.compat.v1 as tf
import pickle as pk
import random

with open("intents.json",'r') as f:
    intents = json.load(f)

try:
    
    with open("data.pickle","rb") as f:
        all_words, tags, xy, x_train, y_train = pk.load(f)
except:
    
    all_words = [] # to contain all words for tokenization and futher process like stemmed words and also for making bag of words

    tags =[] # holds patterns and tags for the patterns for recoginizing

    xy = [] # both pattern and text

    for intent in intents['intents']:
        tag = intent['tag']
        if tag not in tags:
            tags.append(tag)
        for pattern in intent['patterns']:
            w = tokenize(pattern)
            all_words.extend(w) # we do not used append as w is already an array/list and it will make all_words 3D list which is not useful while extend will add element as individual
            xy.append((w, tag))

    all_words = sorted(list(set([lemma(w) for w in all_words])))
    tags = sorted(tags)


    X_train = [] # putting bag of words and vectorizing them
    Y_train = []

    out_empty = [0 for _ in range(len(tags))]



    for(pattern_sent, tag) in xy:
        '''print("Pattern--------",pattern_sent)
        print("Tag|||||||||",tag)'''
        bag = bag_of_words(pattern_sent, all_words)
        X_train.append(bag)
        
        output_row = out_empty[:]
        output_row[tags.index(tag)] = 1
        
        Y_train.append(output_row)#crossentropyloss

    x_train = np.array(X_train)
    y_train = np.array(Y_train)

    with open("data.pickle","wb") as f:
        pk.dump((all_words, tags, xy, x_train, y_train),f)
# Developing Model



tf.disable_v2_behavior()
tf.reset_default_graph()

net = tflearn.input_data(shape=[None, len(x_train[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)



net = tflearn.fully_connected(net, len(y_train[0]), activation = "softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

try:
    
    model.load("model.tflearn")
except:

    model.fit(x_train, y_train, n_epoch=1000, batch_size = 8 , show_metric=True)
    model.save("model.tflearn")


def bow(s, all_words):
    s_words = tokenize(s)
    s_words = [lemma(_) for _ in s_words]
    
    bag = bag_of_words(s_words, all_words)
    return np.array(bag)

def get_response(inp):
    print("Hello! How Can I Help you? (To Stop Please Type Quit)")
    
    while True:
        #inp = input("You : ")
        if (inp.lower() == "quit"):
            break


        results = model.predict([bow(inp, all_words)])[0]
        result_idx = np.argmax(results)
        lab = tags[result_idx]
        
        if (results[result_idx]*100 <= 50):
            l = ["I was not able to Understand. Please can you say it again!", "I was not able to understand you clearly. Please Say it again!"]
            
            
            return random.choice(l)
            
            
        else:
            
            for tg in intents["intents"]:
                if tg["tag"] == lab:
                    responses = tg["responses"]
                    
            return random.choice(responses)


















