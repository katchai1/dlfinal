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
        loss = tf.keras.losses.sparse_categorical_crossentropy(tf.boolean_mask(labels, mask), tf.boolean_mask(prbs, mask), from_logits=False)
        loss = tf.math.reduce_mean(loss)
        return loss

def train(model, sentences, padding_index, vocab_dict, f, words_dict):
    """
    Performs one epoch of the training step. Batches the input, and calls the model on the inputs and labels to train.
    Prints out accuracies, perplexities, and sample sentences intermittently
    """
    total_loss = 0
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
                generate_sentence2(words[0], 25, vocab_dict, model, padding_index, f, words_dict)
        batch_input = sentences[i:i+model.batch_size:1]
        decoder_input = batch_input[:, :-1]
        batch_labels = batch_input[:, 1:]
        batch_mask = (batch_labels != padding_index)
        with tf.GradientTape() as tape:
            prbs = model.call(decoder_input)
            loss = model.loss_function(prbs, batch_labels, batch_mask) # should we divide by loss here?
        total_loss += loss
        accuracy = model.accuracy_function(prbs, batch_labels, batch_mask)
        if i % 3200 == 0:
            print(accuracy)
            print(tf.math.exp(loss))
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
    """
    Chooses one word from the top 100 words in the dictionary, then generates a sentence starting with that word,
    and feeding in the generated output so far as the input into the next step.
    """
    word_probs = np.fromiter(words_dict.values(), dtype='float32')
    words_keys = list(words_dict.keys())
    top_word_inds = np.argpartition(word_probs, -100)[-100:]
    top_word_probs = word_probs[top_word_inds]
    top_word_probs = top_word_probs / np.sum(top_word_probs)
    word_start_ind = np.random.choice(top_word_inds, p=top_word_probs)
    word_start = words_keys[word_start_ind]


    reverse_vocab = {idx:word for word, idx in vocab.items()}
    previous_state = None
    unk_index = vocab["*UNK*"]
    stop_index = vocab["*STOP*"]

    first_string = word_start
    first_word_index = vocab[word_start]

    # append all indices here
    input = [[first_word_index] + [padding_index for i in range(length-1)]]
    text = [first_string]

    for i in range(length):
        probs = model.call(input)
        probs_single = np.array(probs[0][i])
        ind = np.argpartition(probs_single, -10)[-10:]
        top_probs = probs_single[ind]
        top_probs = top_probs / np.sum(top_probs)

        out_index = unk_index
        while (out_index == unk_index or out_index == stop_index):
            out_index = np.random.choice(ind, p=np.array(top_probs))
        text.append(reverse_vocab[out_index])
        if (i != length-1):
            input[0][i+1] = out_index

    print(" ".join(text))
    f.write(" ".join(text) + "\n")

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

if __name__ == '__main__':
   main()
