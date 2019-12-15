import csv
import numpy as np
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import pickle
from statistics import stdev
# if you havin problems do this: https://stackoverflow.com/questions/4867197/failed-loading-english-pickle-with-nltk-data-load

PAD_TOKEN = "*PAD*"
STOP_TOKEN = "*STOP*"
UNK_TOKEN = "*UNK*"

# AMOUNT_BELOW_TO_UNK = 15

def get_data():
    ps = PorterStemmer()

    words = {}
    # however, probably need to convert every word to lowercase and also not sure
    # what to do with the punctuation


    STOP_AMOUNT = 25 # If a comment length is > 25 words, remove it


    processed_sentences = []
    with open("comments.txt") as f:
        for line in f:
            total += 1
            # if i > 3000:
            #     break
            # i += 1
            
            split = word_tokenize(line)
            if len(split) > STOP_AMOUNT:
                continue
            a = []
            for w in split:
                w_stem = ps.stem(w)
                a.append(w_stem)
                if w_stem not in words:
                    words[w_stem] = 1
                else:
                    words[w_stem] = words[w_stem] + 1
            processed_sentences.append(a)
    with open('words.data', 'wb') as f:
        pickle.dump(words, f)
    with open('processed_sentences.data', 'wb') as f:
        pickle.dump(processed_sentences, f)


if __name__ == "__main__":
    get_data()
