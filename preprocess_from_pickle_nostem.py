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

    # TODO: look into porter stemmer to process punctuation and lowercase-ness
    # i = 0
    words = None
    pre_processed_sentences = None
    with open('words_nostem.data', 'rb') as f:
        words = pickle.load(f)
    with open('processed_sentences_nostem.data', 'rb') as f:
        pre_processed_sentences = pickle.load(f)

    # going to guess words that appear less than 10 times can get UNK'd
    # i = 0
    print(len(pre_processed_sentences))

    processed_sentences = []
    for line in pre_processed_sentences:
        if len(line) > STOP_AMOUNT:
            continue
        a = []
        for w in line:
            if words[w] <= AMOUNT_BELOW_TO_UNK:
                a.append(UNK_TOKEN)
            else:
                a.append(w)
        a.append(STOP_TOKEN)
        a = a + [PAD_TOKEN] * (STOP_AMOUNT + 1 - len(a))
        # written.write(str(a))
        processed_sentences.append(a)

    # Do we need to pad?
    word_list = []
    for k, v in words.items():
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
        # f_2.write(str(processed_sentence))
        numerical_words.append(processed_sentence)
    # print("3")

    # with open('numerical_sentences_abbreviated.data', 'wb') as f:
    #     pickle.dump(numerical_words, f)

    return (np.array(numerical_words), vocab_dict, vocab_dict[PAD_TOKEN], words)