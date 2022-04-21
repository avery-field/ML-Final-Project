#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 15:05:08 2022

@author: averyfield2
"""

import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
from keras.models import Sequential
from keras.layers.recurrent import LSTM,SimpleRNN
import numpy
import tflearn
import tensorflow
import random
import json
import pickle

with open("/Users/averyfield2/Desktop/Spring_22/Machine Learning/intents_copy.json") as file:
    data = json.load(file)

words = []
labels = []
docs_x = []
docs_y = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])

    if intent["tag"] not in labels:
        labels.append(intent["tag"])

words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))

labels = sorted(labels)

training = []
output = []

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []

    wrds = [stemmer.stem(w.lower()) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)


training = numpy.array(training)
output = numpy.array(output)


model=Sequential()

training = training.reshape(training.shape[0], 1, training.shape[1])
output =output.reshape(output.shape[0], 1, output.shape[1])
model.add(LSTM(6,input_shape=(training.shape[1:]), return_sequences=True, kernel_initializer='glorot_normal', recurrent_initializer='glorot_normal', activation='sigmoid'))


model.compile(loss='cosine_similarity', optimizer='adam', metrics=['accuracy'])
model.fit(training, output, epochs=1000)

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return numpy.array(bag)


def chat():
    print("Start talking with the bot (type quit to stop)!")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        p = numpy.array([bag_of_words(inp, words)])
        p = p.reshape(p.shape[0], 1, p.shape[1])
        results = model.predict(p)
        results_index = numpy.argmax(results)
        tag = labels[results_index]

        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']

        print(random.choice(responses))

chat()