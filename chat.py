import nltk
from nltk.stem.lancaster import LancasterStemmer
import tensorflow
import tflearn
stemmer = LancasterStemmer()

import numpy
import random
import json
import pickle

with open('intents.json') as file:
    data = json.load(file)

with open('data.pickle', 'rb') as f:
    words, labels, training, output = pickle.load(f)

tensorflow.compat.v1.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation='softmax')
net = tflearn.regression(net)

model = tflearn.DNN(net)
model.load('model.tflearn')

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(w.lower()) for w in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = (1)

    return numpy.array(bag)

def chat():
    print('Welcome, I am StoicBot!\nI will respond to any concerns you have with a quote from one of the great stoics\n please mention a feeling in your concern to help me\ntype "quit" to exit')
    while True:
        inp = input('You: ')
        if inp.lower() == 'quit':
            break
        
        results = model.predict([bag_of_words(inp, words)])[0]
        results_index = numpy.argmax(results)
        tag = labels[results_index]

        if results[results_index] > 0.6:
            for block in data['intents']:
                if block['tag'] == tag:
                    responses = block['responses']
                    break
            print('Machine:', random.choice(responses))
        else:
            print('Machine: Sorry, I do not understnad, please ask another question!')

chat()