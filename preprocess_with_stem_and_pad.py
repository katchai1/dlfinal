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

def get_data(AMOUNT_BELOW_TO_UNK):
    ps = PorterStemmer()

    words = {}
    # however, probably need to convert every word to lowercase and also not sure
    # what to do with the punctuation

    # TODO: look into porter stemmer to process punctuation and lowercase-ness 
    # i = 0
    STOP_AMOUNT = 25

    try:
        with open('words.data', 'rb') as f:
            words = pickle.load(f)
        # Do something with the file
    except IOError:
        print("here")
        processed_sentences = []
        with open("comments.txt") as f:
            for line in f:
                # if i > 3000:
                #     break
                # i += 1
                if len(line) > STOP_AMOUNT:
                    continue
                split = word_tokenize(line)
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

    assert False
        
    # print("1")
    
    

    # going to guess words that appear less than 10 times can get UNK'd
    # i = 0
    processed_sentences = []

    # hello
    with open("comments.txt") as f:
    # with open("processed_words.txt", mode='w') as written: 
        for line in f:
            # if i > 3000:
            #     break
            # i += 1
            split = word_tokenize(line)
            if len(split) > STOP_AMOUNT:
                continue
            # max_len = max(max_len, len(split))
            a = []
            for w in split:
                w_stem = ps.stem(w)
                a.append(w_stem)
            a.append(STOP_TOKEN)
            # written.write(str(a))
            processed_sentences.append(a)
    with open('processed_sentences.data', 'wb') as f:
            pickle.dump(processed_sentences, f)
    

    with open("comments.txt") as f:
    # with open("processed_words.txt", mode='w') as written: 
        for line in f:
            # if i > 3000:
            #     break
            # i += 1
            split = word_tokenize(line)
            if len(split) > STOP_AMOUNT:
                continue
            # max_len = max(max_len, len(split))
            a = []
            for w in split:
                w_stem = ps.stem(w)
                if words[w_stem] <= AMOUNT_BELOW_TO_UNK:
                    a.append(UNK_TOKEN)
                else:
                    a.append(w_stem)
            a.append(STOP_TOKEN)
            # written.write(str(a))
            processed_sentences.append(a)
                                  
    # print("2")
    # print("average is 12")
    # print("Standard deviation of sample is:  9.11731883544809")

    
    # Do we need to pad?
    word_list = []
    for k, v in words:
        if v > AMOUNT_BELOW_TO_UNK:
            word_list.append(k)
    word_list = word_list + [STOP_TOKEN, PAD_TOKEN, UNK_TOKEN]

    vocab_dict = {k: v for v, k in enumerate(word_list)}
    # word_set = set([])
    # for sentence in processed_sentences:
    #     for word in sentence:
    #         word_set.add(word)
    # word_list = list(word_set) + [STOP_TOKEN, PAD_TOKEN]
    
    
    # for i in range(len(word_list)):
    #     vocab_dict[word_list[i]] = i
    
    # with open('vocab_dict_abbreviated.data', 'wb') as f:
    #     pickle.dump(vocab_dict, f)
    
    # print("2.5")
    
    numerical_words = []
    # with open("processed_indices.txt", mode='w') as f_2:
    for sentence in processed_sentences:
        processed_sentence = []
        for word in sentence:
            processed_sentence.append(vocab_dict[word])
        processed_sentence = processed_sentence + [vocab_dict[PAD_TOKEN]] * (STOP_AMOUNT + 1 - len(sentence))
        # f_2.write(str(processed_sentence))
        numerical_words.append(processed_sentence)
    # print("3")

    # with open('numerical_sentences_abbreviated.data', 'wb') as f:
    #     pickle.dump(numerical_words, f)
    
    return (numerical_words, vocab_dict, vocab_dict[PAD_TOKEN])

def get_data_from_pickle():
    vocab_dict = None
    with open('vocab_dict_abbreviated.data', 'rb') as vocab:
        vocab_dict = pickle.load(vocab)
    
    sentences = None
    with open('numerical_sentences_abbreviated.data', 'rb') as sentence_file:
        sentences = pickle.load(sentence_file, encoding='bytes') 
    # print("SUM %s" %(np.sum(np.array([len(x1) != 26 for x1 in sentences]))))
    return (np.array(sentences), vocab_dict, vocab_dict[PAD_TOKEN])

if __name__ == "__main__":
    get_data()
