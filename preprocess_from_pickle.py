import pickle
import numpy as np

PAD_TOKEN = "*PAD*"
STOP_TOKEN = "*STOP*"
UNK_TOKEN = "*UNK*"
STOP_AMOUNT = 25

def get_data(AMOUNT_BELOW_TO_UNK):
    words = {}
    # however, probably need to convert every word to lowercase and also not sure
    # what to do with the punctuation

    words = None # words will be the word to frequency dictionary
    pre_processed_sentences = None # this is a list of lists of a list of the stemmed words in the sentence
    with open('words.data', 'rb') as f:
        words = pickle.load(f)
    with open('processed_sentences.data', 'rb') as f:
        pre_processed_sentences = pickle.load(f)

   

    processed_sentences = []
    for line in pre_processed_sentences:
        if len(line) > STOP_AMOUNT: # second check to make sure no sentence is greater than 25
            continue
        a = []
        for w in line:
            if words[w] <= AMOUNT_BELOW_TO_UNK: # Unk words below a certain amount
                a.append(UNK_TOKEN)
            else:
                a.append(w)
        a.append(STOP_TOKEN)
        a = a + [PAD_TOKEN] * (STOP_AMOUNT + 1 - len(a)) #pad all sentences to 26 (to include the stop token and max length 25)
        # written.write(str(a))
        processed_sentences.append(a)
                                  
    word_list = [] # this will be a list of all words contained in all sentences
    for k, v in words.items():
        if v > AMOUNT_BELOW_TO_UNK:
            word_list.append(k)
    word_list = word_list + [STOP_TOKEN, PAD_TOKEN, UNK_TOKEN]

    vocab_dict = {k: v for v, k in enumerate(word_list)} # now a word to id dictionary
    
    numerical_words = []
    for sentence in processed_sentences:
        processed_sentence = []
        for word in sentence:
            processed_sentence.append(vocab_dict[word])
        numerical_words.append(processed_sentence)
    

    return (np.array(numerical_words), vocab_dict, vocab_dict[PAD_TOKEN], words)
