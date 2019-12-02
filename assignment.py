import os
import numpy as np
import tensorflow as tf
import numpy as np
import transformer_funcs as transformer
from preprocess_for_final import *
import sys

class Model(tf.keras.Model):
    def __init__(self, sentence_size):
        super(Model, self).__init__()

        # Model Vars
        self.batch_size = 64
        self.embedding_size = 34
        self.vocab_size = sentence_size
        self.window_size = 277

        # Layers
        self.embedding_model = tf.keras.layers.Embedding(self.vocab_size, self.embedding_size, input_length=self.window_size)
        self.positional_model = transformer.Position_Encoding_Layer(self.window_size, self.embedding_size)
        self.transformer = transformer.Transformer_Block(self.embedding_size, True)
        self.dense_model = tf.keras.layers.Dense(self.vocab_size, activation="softmax")

    def call(self, sentences):
        npsentence = np.asarray(sentences)
        embeddings = self.embedding_model(tf.convert_to_tensor(npsentence))
        print(embeddings.shape)
        positional = self.positional_model(embeddings)
        transforer = self.transformer(positional)
        prbs = self.dense_model(transformer)
        return prbs

def train(model, sentences):
    total_loss = 0
    total_accuracy = 0
    total_mask = 0
    print(sentences[0])
    for i in range(0, len(sentences), model.batch_size):
        batch_input = sentences[i:i+model.batch_size:1]
        batch_labels = sentences[i+1:i+model.window_size+1:1] 
        #batch_mask = batch_english_labels != eng_padding_index
        #batch_mask_sum = np.sum(batch_mask)
        #total_mask += batch_mask_sum
        with tf.GradientTape() as tape:
            print(np.array(batch_input).shape)
            prbs = model.call(batch_input)
            loss = model.loss_function(prbs, batch_labels) #, batch_mask) # should we divide by loss here?
        total_loss += loss
        accuracy = model.accuracy_function(prbs, batch_english_labels) #, batch_mask)
        total_accuracy += accuracy * batch_mask_sum
        grad = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(grad,model.trainable_variables))
        """if (i / model.batch_size) % 1000 == 0:
            print("Training on input: " + str(i) + " with perplexity: " + str(tf.math.exp(loss / batch_mask_sum)) + " and accuracy: " + str(model.accuracy_function(prbs, batch_english_labels, batch_mask)))
    print("Training completed with perplexity: " + str(tf.math.exp(total_loss / total_mask)) + " with accuracy: " + str(total_accuracy / total_mask))"""

def test(model, sentences):
    total_loss = 0
    total_accuracy = 0
    total_mask = 0
    for i in range(0, sentences.shape[0], model.batch_size):
        batch_input = sentences[i:i+model.batch_size:1]
        batch_labels = sentences[i+1:i+model.batch_size+1:1]
        #batch_mask = batch_english_labels != eng_padding_index
        #batch_mask_sum = np.sum(batch_mask)
        #total_mask += batch_mask_sum
        prbs = model.call(batch_input)
        loss = model.loss_function(prbs, batch_labels) #, batch_mask) # should we divide by loss here?
        total_loss += loss
        #accuracy = model.accuracy_function(prbs, batch_labels), batch_mask)
        #total_accuracy += accuracy * batch_mask_sum
        """if (i / model.batch_size) % 1000 == 0:
            print("Testing on input: " + str(i) + " with perplexity: " + str(tf.math.exp(loss / batch_mask_sum)) + " and accuracy: " + str(model.accuracy_function(prbs, batch_english_labels, batch_mask)))
    final_perplexity = tf.math.exp(total_loss / total_mask)
    final_accuracy = total_accuracy / total_mask
    print("Testing completed with perplexity: " + str(final_perplexity) + " with accuracy: " + str(final_accuracy))"""
    
    return (total_loss)

def main():    
    print("Running preprocessing...")
    (sentences, vocab_dict) = get_data()
    print("Preprocessing complete.")

    model = Model(len(sentences))

    # Train and Test Model for 1 epoch.
    train(model, sentences)
    print(test())

if __name__ == '__main__':
   main()
