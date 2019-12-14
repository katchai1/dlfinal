import os
import numpy as np
import tensorflow as tf
import numpy as np
import transformer_funcs as transformer
from preprocess_from_pickle import *
# from preprocess_from_pickle_nostem import *
import sys
import random

class Model(tf.keras.Model):
    def __init__(self, vocab_size, sentence_size):
        super(Model, self).__init__()

        # Model Vars
        self.batch_size = 64
        self.embedding_size = 100
        self.vocab_size = vocab_size
        self.window_size = sentence_size - 1
        # self.learning_rate = 0.01
        self.learning_rate = 0.01
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = self.learning_rate)

        # Layers
        self.embedding_model = tf.keras.layers.Embedding(self.vocab_size, self.embedding_size, input_length=self.window_size)
        self.positional_model = transformer.Position_Encoding_Layer(self.window_size, self.embedding_size)
        self.transformer1 = transformer.Transformer_Block(self.embedding_size, False, multi_headed=False)
        # self.transformer2 = transformer.Transformer_Block(self.embedding_size, False)
        self.dense_model = tf.keras.layers.Dense(self.vocab_size, activation="softmax")

    def call(self, sentences):
        tfsentence = tf.convert_to_tensor(sentences) #np.asarray(sentences)
        embeddings = self.embedding_model(tfsentence)
        positional = self.positional_model(embeddings)
        transformer = self.transformer1(positional)
        # transformer = self.transformer2(transformer)
        prbs = self.dense_model(transformer)
        return prbs

    def accuracy_function(self, prbs, labels, mask):
        """
        DO NOT CHANGE
        Computes the batch accuracy
        :param prbs:  float tensor, word prediction probabilities [batch_size x window_size x english_vocab_size]
        :param labels:  integer tensor, word prediction labels [batch_size x window_size]
        :param mask:  tensor that acts as a padding mask [batch_size x window_size]
        :return: scalar tensor of accuracy of the batch between 0 and 1
        """

        decoded_symbols = tf.argmax(input=prbs, axis=2)
        accuracy = tf.reduce_mean(tf.boolean_mask(tf.cast(tf.equal(decoded_symbols, labels), dtype=tf.float32),mask))
        return accuracy


    def loss_function(self, prbs, labels, mask):
        """
        Calculates the model cross-entropy loss after one forward pass
        :param prbs:  float tensor, word prediction probabilities [batch_size x window_size x english_vocab_size]
        :param labels:  integer tensor, word prediction labels [batch_size x window_size]
        :param mask:  tensor that acts as a padding mask [batch_size x window_size]
        :return: the loss of the model as a tensor
        """

        # Note: you can reuse this from rnn_model.
        loss = tf.keras.losses.sparse_categorical_crossentropy(tf.boolean_mask(labels, mask), tf.boolean_mask(prbs, mask), from_logits=False)
        loss = tf.math.reduce_mean(loss)
        return loss

def train(model, sentences, padding_index, vocab_dict, f, words_dict):
    total_loss = 0
    # accuracies = []
    # total_accuracy = 0
    # total_mask = 0
    # print(sentences[0])
#    print("total num sentences: " + str(len(sentences)))
    print(len(sentences) / model.batch_size)
    n = len(sentences)
    indices = np.arange(n)
    indices = tf.random.shuffle(indices)
    sentences = tf.gather(sentences, indices)
    for i in range(0, len(sentences) - 1 - model.batch_size, model.batch_size):
    #for i in range(0, 64000, model.batch_size): # TODO change back
        if i % 3200 == 0:
           print(i)
        if i % 12800 == 0:
            for i in range(5):
                words = random.sample(vocab_dict.keys(), 2)
                # generate_sentence2(words[0], 26, vocab_dict, model, padding_index)
                generate_sentence2(words[0], 25, vocab_dict, model, padding_index, f, words_dict)
        batch_input = sentences[i:i+model.batch_size:1]
        decoder_input = batch_input[:, :-1]
        batch_labels = batch_input[:, 1:]
        # print(decoder_input.shape)
#        with np.printoptions(threshold=np.inf):
#            print(batch_input)
#            print(str(batch_labels) + "\n\n\n\n\n")
        batch_mask = (batch_labels != padding_index)
        with tf.GradientTape() as tape:
            # print(np.array(batch_input).shape)
            prbs = model.call(decoder_input)
            loss = model.loss_function(prbs, batch_labels, batch_mask) # should we divide by loss here?
        total_loss += loss
        accuracy = model.accuracy_function(prbs, batch_labels, batch_mask)
        # if i % 3200 == 0:
        #     accuracies.append(accuracy.numpy())
        #     print(accuracies)
        # print(accuracy)
        # total_accuracy += accuracy * batch_mask_sum
        grad = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(grad,model.trainable_variables))
        """if (i / model.batch_size) % 1000 == 0:
            print("Training on input: " + str(i) + " with perplexity: " + str(tf.math.exp(loss / batch_mask_sum)) + " and accuracy: " + str(model.accuracy_function(prbs, batch_english_labels, batch_mask)))
    print("Training completed with perplexity: " + str(tf.math.exp(total_loss / total_mask)) + " with accuracy: " + str(total_accuracy / total_mask))"""

def test(model, sentences, padding_index):
    total_loss = 0
    total_accuracy = 0
    total_mask = 0
    for i in range(0, sentences.shape[0] - 1 - model.batch_size, model.batch_size):
        batch_input = sentences[i:i+model.batch_size:1]
        decoder_input = batch_input[:, :-1]
        batch_labels = batch_input[:, 1:]
        batch_mask = (batch_labels != padding_index)
        #batch_mask = batch_english_labels != eng_padding_index
        batch_mask_sum = np.sum(batch_mask)
        total_mask += batch_mask_sum
        prbs = model.call(decoder_input)
        loss = model.loss_function(prbs, batch_labels, batch_mask) # should we divide by loss here?
        total_loss += loss
        accuracy = model.accuracy_function(prbs, batch_labels, batch_mask)
        total_accuracy += accuracy * batch_mask_sum
        """if (i / model.batch_size) % 1000 == 0:
            print("Testing on input: " + str(i) + " with perplexity: " + str(tf.math.exp(loss / batch_mask_sum)) + " and accuracy: " + str(model.accuracy_function(prbs, batch_english_labels, batch_mask)))
    final_perplexity = tf.math.exp(total_loss / total_mask)
    final_accuracy = total_accuracy / total_mask
    print("Testing completed with perplexity: " + str(final_perplexity) + " with accuracy: " + str(final_accuracy))"""

    return (total_loss, total_accuracy / total_mask)

def generate_sentence2(word1, length, vocab, model, padding_index, f, words_dict):
    # starting_words = ["what", "you", "oh", "yes", "ok", "sure", "whi", "do", "but", "for", "if", "I", "will"]
    # i = np.random(len(starting_words))
    # word_start = starting_words[i]
    # word_start = np.random.choice(starting_words)
    # word_start = word1
    # print("generate sentence")
    word_probs = np.fromiter(words_dict.values(), dtype='float32')
    words_keys = list(words_dict.keys())
    top_word_inds = np.argpartition(word_probs, -100)[-100:]
    top_word_probs = word_probs[top_word_inds]
    top_word_probs = top_word_probs / np.sum(top_word_probs)
    # print(probs.shape)
    # print(probs)
    # probs = probs / np.sum(probs)
    word_start_ind = np.random.choice(top_word_inds, p=top_word_probs)
    word_start = words_keys[word_start_ind]


    reverse_vocab = {idx:word for word, idx in vocab.items()}
    previous_state = None
    unk_index = vocab["*UNK*"]
    stop_index = vocab["*STOP*"]

    # first_string = word1
    # first_word_index = vocab[word1]
    # next_input = [[first_word_index]]
    first_string = word_start
    first_word_index = vocab[word_start]
    next_input = [[first_word_index]]

    # append all indices here
    input = [[first_word_index] + [padding_index for i in range(length-1)]]
    # print(input)
    text = [first_string]

    for i in range(length):
        # probs = model.call(next_input)
        probs = model.call(input)
        # print(probs.shape)
        # probs_single = np.array(probs[0][0])
        probs_single = np.array(probs[0][i])
        # ind = np.argpartition(probs[0][0], -4)[-4:]
        ind = np.argpartition(probs_single, -10)[-10:]
        top_probs = probs_single[ind]
        top_probs = top_probs / np.sum(top_probs)

#        for x in ind:
#            print(reverse_vocab[x])
#        print(probs.shape)
#        print(probs[0][0].shape)
        # out_index = np.argmax(np.array(probs[0][0]))
        # print(reverse_vocab[np.argmax(np.array(probs[0][0]))])
        # out_index = np.random.choice(np.arange(len(vocab)+1), p=np.array(probs[0][0]))

        out_index = unk_index
        while (out_index == unk_index or out_index == stop_index):
            out_index = np.random.choice(ind, p=np.array(top_probs))
        # print(reverse_vocab[out_index])
        text.append(reverse_vocab[out_index])
        next_input = [[out_index]]
        # input[0].append(out_index)
        if (i != length-1):
            input[0][i+1] = out_index
        # print(input)

    print(" ".join(text))
    f.write(" ".join(text) + "\n")

def generate_sentence(word1, word2, length, vocab, model):
    """
    Given initial 2 words, print out predicted sentence of target length.
    :param word1: string, first word
    :param word2: string, second word
    :param length: int, desired sentence length
    :param vocab: dictionary, word to id mapping
    :param model: trained trigram model
    """
    reverse_vocab = {idx:word for word, idx in vocab.items()}
    output_string = np.zeros((1,length), dtype=np.int)
    output_string[:, :2] = vocab[word1], vocab[word2]

    for end in range(2, length):
        print(output_string)
        start = end - 2
        print(start)
        print(end)
        output_string[:, end] = np.argmax(model(output_string[:,start:end]), axis=1)
    text = [reverse_vocab[i] for i in list(output_string[0])]

    print(" ".join(text))

def main():
    best = 15
    best_accuracy = 0.0
    accuracy_dict = {}
    print("Running preprocessing...")
    (sentences, vocab_dict, padding_index, words_dict) = get_data(10)
    # print(words)
    print("Preprocessing complete.")

    model = Model(len(vocab_dict)+1, sentences.shape[1])
    # Train and Test Model for 1 epoch.
    f = open("output6.txt", "w")
    for i in range(5):
        print("epoch: " + str(i))
        train(model, sentences, padding_index, vocab_dict, f, words_dict)
    for i in range(50):
       words = random.sample(vocab_dict.keys(), 2)
       generate_sentence2(words[0], 25, vocab_dict, model, padding_index, f, words_dict)
    # for i in range(10, 110, 100): # TODO change back
    #     print("Running preprocessing...")
    #     (sentences, vocab_dict, padding_index) = get_data(i)
    #     print("Preprocessing complete.")
    #
    #     model = Model(len(vocab_dict)+1, sentences.shape[1])
    #     print("UNK Level: %s" %(i))
    #     # Train and Test Model for 1 epoch.
    #     f = open("output3.txt", "w")
    #     for i in range(3):
    #         train(model, sentences, padding_index, vocab_dict, f)
    #     for i in range(50):
    #        words = random.sample(vocab_dict.keys(), 2)
    #        generate_sentence2(words[0], 25, vocab_dict, model)
    #     (loss, accuracy) = test(model, sentences, padding_index)
    #     accuracy_dict[i] = accuracy
    #     print("UNK BELOW LEVEL: %s, accuracy: %s, loss: %s" %(i, accuracy, loss))
    #
    #     if accuracy > best_accuracy:
    #         best = i
    #         best_accuracy = accuracy
    #
    #     print("The best accuracy is: %s at UNK level %s" %(best_accuracy, best))
    #     print(accuracy_dict)

if __name__ == '__main__':
   main()
