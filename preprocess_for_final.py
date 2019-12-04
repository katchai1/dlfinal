import csv
import numpy as np
#import nltk
#nltk.download()
#from nltk.tokenize import word_tokenize
# if you havin problems do this: https://stackoverflow.com/questions/4867197/failed-loading-english-pickle-with-nltk-data-load

def get_data():
    words = {}
    # however, probably need to convert every word to lowercase and also not sure
    # what to do with the punctuation

    # TODO: look into porter stemmer to process punctuation and lowercase-ness
    # i = 0
    with open("comments.txt") as f:
        for line in f:
            # if i > 3000:
            #     break
            # i += 1
            split = line.split()
            for w in split:
                if w not in words:
                    words[w] = 1
                else:
                    words[w] = words[w] + 1

    print(len(words.keys()))
    # going to guess words that appear less than 10 times can get UNK'd
    # i = 0
    processed_sentences = []
    PAD_TOKEN = "*PAD*"
    STOP_TOKEN = "*STOP*"

    max_len = 0
    with open("comments.txt") as f:
        for line in f:
            # if i > 3000:
            #     break
            # i += 1
            split = line.split()
            max_len = max(max_len, len(split))
            a = []
            for w in split:
                if words[w] <= 100:
                    a.append("UNK")
                else:
                    a.append(w)
            a.append(STOP_TOKEN)
            processed_sentences.append(a)


    # Do we need to pad?

    vocab_dict = {}
    word_set = set([])
    for sentence in processed_sentences:
        for word in sentence:
            word_set.add(word)
    word_list = list(word_set) + [STOP_TOKEN, PAD_TOKEN]
    for i in range(len(word_list)):
        vocab_dict[word_list[i]] = i
    print(len(word_list))
    numerical_words = []
    for sentence in processed_sentences:
        processed_sentence = []
        for word in sentence:
            processed_sentence.append(vocab_dict[word])
        processed_sentence = processed_sentence + [vocab_dict[STOP_TOKEN]] + [vocab_dict[PAD_TOKEN]] * (max_len - len(sentence))
        numerical_words.append(processed_sentence)
    return (numerical_words, vocab_dict, vocab_dict[PAD_TOKEN])

if __name__ == "__main__":
    print(get_data()[0])
