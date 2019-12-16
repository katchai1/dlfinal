## File descriptions

assignment.py 	The main file that contains the model. You run this file to start training the model and generating sentences

comments.txt 	Only sarcastic comments from the dataset: https://www.kaggle.com/sherinclaudia/sarcastic-comments-on-reddit

csvToText.py 	This is what turns the dataset as a csv into the two text files, comments.txt and parents.txt	

parents.txt 	The parent comments of the sarcastic comments from the dataset, line-by-line order. We don't use this, but it'd be the next step

preprocess_from_pickle.py 	Has the get_data() function that takes the pickle'd files (processed_sentences.data and words.data) of the vocab to frequency dict and the list of stemmed sentences to UNK words below a threshold, add a STOP token to the end of sentences and PAD the length to 26, and return to the caller

preprocess_from_pickle_nostem.py 	Similar to the preprocess_from_pickle, but it uses the processed_sentences_nostem.data that has nonstemmed words


preprocess_to_pickle.py 	Converts the comments.txt into a stemmed word to frequency dict (and saves it as words.data) and a list of lists (stemmed words in sentences)

processed_sentences.data 	The pickle'd file of the list of stemmed words in sentences

processed_sentences_nostem.data 	The pickle'd file of the list of nonstemmed words in sentences

transformer_funcs.py 	The transformer architecture given from class	

words.data 	The pickle'd stemmed word to frequency dict from preprocess_to_pickle.py 

words_nostem.data The pickle'd non_stemmed word to frequency dict from a modified preprocess_to_pickle.py

## How to run our model

Make sure you are using the course env, and then you will call "python assignment.py" to train the model for 5 epochs, generating sample sentences every 100 batches, and then generate a final sample of 50 sentences after the 5 epochs, saving them to output6.txt

If you want to run the preprocess_to_pickle.py file, you will need to run "pip install nltk" and then follow the steps from the first response: https://stackoverflow.com/questions/31295957/nltk-word-tokenize-giving-attributeerror-module-object-has-no-attribute-de