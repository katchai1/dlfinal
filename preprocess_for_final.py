import csv
import numpy as np

def get_data():
    words = {}
    # however, probably need to convert every word to lowercase and also not sure
    # what to do with the punctuation

    # TODO: look into porter stemmer to process punctuation and lowercase-ness 
    with open("comments.txt") as f:
        for line in f:
            split = line.split()
            for w in split:
                if w not in words:
                    words[w] = 1
                else:
                    words[w] = words[w] + 1

    # going to guess words that appear less than 10 times can get UNK'd
    processed_sentences = []
    with open("comments.txt") as f:
        for line in f:
            split = line.split()
            a = []
            for w in split:
                if words[w] <= 10:
                    a.append("UNK")
                else:
                    a.append(w)
            processed_sentences.append(a)
    # Do we need to pad?

    vocab_dict = {}
    word_set = set([])
    for sentence in processed_sentences:
        for word in sentence:
            word_set.add(word)
    word_list = list(word_set)
    for i in range(len(word_list)):
        vocab_dict[word_list[i]] = i
    numerical_words = [vocab_dict[i] for i in word_list]
    return (numerical_words, vocab_dict)
