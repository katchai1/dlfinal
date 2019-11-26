import csv

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
    return processed_sentences
