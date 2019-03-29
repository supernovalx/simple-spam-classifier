
import re
import numpy as np
import nltk
from nltk.stem import PorterStemmer

def loadVocalList():
    with open("vocab.txt") as f:
        return {x.split('\t')[1].replace("\n",""):int(x.split('\t')[0]) for x in f.readlines()}

def preprocessEmail(email):
    # Lower case email
    email = email.lower()

    # Strip HTML tags
    email = re.sub("<[^<>]+>"," ",email)

    # Replace numbers with 'number'
    email = re.sub("[0-9]+","number",email)

    # Replace URLs with 'httpaddr'
    email = re.sub("(http|https)://[^\s]*","httpaddr",email)

    # Replace Emails with 'emailaddr'
    email = re.sub("[^\s]+@[^\s]+","emailaddr",email)

    # Replace $ with 'dollar'
    email = re.sub("[$]+","dollar",email)

    # Remove any non alphanumeric characters
    # Keep white spaces for tokenize later
    email = re.sub("[^a-zA-Z0-9]"," ",email)

    # Tokenize
    tokens = nltk.word_tokenize(email)

    # Stem
    ps = PorterStemmer()
    stems = [ps.stem(w) for w in tokens]

    # Load vocab list
    vocab = loadVocalList()

    # Return word indices
    return [vocab[w] for w in stems if w in vocab]

def emailFeatures(word_indices):
    vocab = loadVocalList()

    features = np.zeros(len(vocab))

    for w in word_indices:
        features[w] = 1

    return features.reshape(1,-1)