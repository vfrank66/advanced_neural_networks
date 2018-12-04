#!/usr/bin/env python
# coding: utf-8

# In[37]:


from keras.models import Model
from keras.layers.recurrent import LSTM
from keras.layers import Dense, Input, Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint, TensorBoard
from collections import Counter
import nltk
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
from pathlib import Path


# In[38]:
#

# you should only need to run this once.
#nltk.set_proxy('http://web-spfld-vwsa2.int.trsil.org:9001/proxy.pac', ('frankvw','Basement#5'))
#nltk.download('punkt')

# In[38]:
#
import os
currentDir = Path("c:/dev/advanced_neural_networks/module_2/week_2")

# In[39]:


class TextData:
    """data class for text data"""
    
    def __init__(self, input_texts, target_texts, input_counter, target_counter):
        self.input_texts = input_texts
        self.target_texts = target_texts
        self.input_counter = input_counter
        self.target_counter = target_counter

        
    def save(self):
        self.filename = Path(os.path.join(currentDir, "text_data.pkl"))
        pickle.dump(self, open(self.filename,'wb'))
    
    @classmethod
    def load(self):
        self.filename = Path(os.path.join(currentDir, "text_data.pkl"))
        return pickle.load(open(self.filename,'rb'))


# In[40]:


def load_data(path=Path(os.path.join(currentDir, "./data/movie_lines.txt")), max_seq_length=20, max_vocab_size=100):
    """Based on: https://github.com/subpath/ChatBot/blob/master/Chatbot_training.py"""
    
    input_counter = Counter()
    target_counter = Counter()
    input_texts = []
    target_texts = []

    with open(path, 'r', encoding="latin-1") as f:
        df = f.read()
    rows = df.split('\n')
    lines = [row.split(' +++$+++ ')[-1] for row in rows]


    prev_words = []
    for line in lines:

        next_words = [w.lower() for w in nltk.word_tokenize(line)]
        if len(next_words) > max_seq_length:
            next_words = next_words[0:max_seq_length]

        if len(prev_words) > 0:
            input_texts.append(prev_words)
            for w in prev_words:
                input_counter[w] += 1
            target_words = next_words[:]
            target_words.insert(0, 'START')
            target_words.append('END')
            for w in target_words:
                target_counter[w] += 1
            target_texts.append(target_words)

        prev_words = next_words
    td = TextData(input_texts, target_texts, input_counter, target_counter)
    return td


# In[41]:


def encode_raw_text_data(text_data, max_vocab_size=100):
    input_word2idx = dict()
    target_word2idx = dict()
    for idx, word in enumerate(text_data.input_counter.most_common(max_vocab_size)):
        input_word2idx[word[0]] = idx + 2
    for idx, word in enumerate(text_data.target_counter.most_common(max_vocab_size)):
        target_word2idx[word[0]] = idx + 1

    input_word2idx['PAD'] = 0
    input_word2idx['UNK'] = 1
    target_word2idx['UNK'] = 0

    input_idx2word = dict([(idx, word) for word, idx in input_word2idx.items()])
    target_idx2word = dict([(idx, word) for word, idx in target_word2idx.items()])

    num_encoder_tokens = len(input_idx2word)
    num_decoder_tokens = len(target_idx2word)

    text_data.input_word2idx = input_word2idx
    text_data.input_idx2word = input_idx2word
    text_data.target_word2idx = target_word2idx
    text_data.target_idx2word = target_idx2word
    

    encoder_input_data = []

    encoder_max_seq_length = 0
    decoder_max_seq_length = 0

    for input_words, target_words in zip(text_data.input_texts, text_data.target_texts):
        encoder_input_wids = []
        for w in input_words:
            w2idx = 1
            if w in input_word2idx:
                w2idx = input_word2idx[w]
            encoder_input_wids.append(w2idx)

        encoder_input_data.append(encoder_input_wids)
        encoder_max_seq_length = max(len(encoder_input_wids), encoder_max_seq_length)
        decoder_max_seq_length = max(len(target_words), decoder_max_seq_length)

    
    text_data.encoder_input_data = encoder_input_data
    text_data.num_encoder_tokens = num_encoder_tokens
    text_data.num_decoder_tokens = num_decoder_tokens
    text_data.encoder_max_seq_length = encoder_max_seq_length
    text_data.decoder_max_seq_length = decoder_max_seq_length
    
    return text_data
    


# In[42]:


def generate_batch(input_data, output_text_data, text_data, batch_size=128):
    num_batches = len(input_data) // batch_size
    while True:
        for batchIdx in range(0, num_batches):
            start = batchIdx * batch_size
            end = (batchIdx + 1) * batch_size
            encoder_input_data_batch = pad_sequences(input_data[start:end], text_data.encoder_max_seq_length)
            decoder_target_data_batch = np.zeros(shape=(batch_size, text_data.decoder_max_seq_length, text_data.num_decoder_tokens))
            decoder_input_data_batch = np.zeros(shape=(batch_size, text_data.decoder_max_seq_length, text_data.num_decoder_tokens))
            for lineIdx, target_words in enumerate(output_text_data[start:end]):
                for idx, w in enumerate(target_words):
                    w2idx = 0
                    if w in text_data.target_word2idx:
                        w2idx = text_data.target_word2idx[w]
                    decoder_input_data_batch[lineIdx, idx, w2idx] = 1
                    if idx > 0:
                        decoder_target_data_batch[lineIdx, idx - 1, w2idx] = 1
            yield [encoder_input_data_batch, decoder_input_data_batch], decoder_target_data_batch


# In[43]:


def build_model(text_data, hidden_units=256):
    #encoder_inputs = Input(shape=(None,), name='encoder_inputs')
    encoder_inputs = Input(shape=(None,text_data.num_encoder_tokens), name='encoder_inputs')
    # encoder_embedding = Embedding(input_dim=text_data.num_encoder_tokens, output_dim=hidden_units,
    #                               input_length=text_data.encoder_max_seq_length, name='encoder_embedding')
    encoder_lstm = LSTM(units=hidden_units, return_state=True, name='encoder_lstm')
    #encoder_outputs, encoder_state_h, encoder_state_c = encoder_lstm(encoder_embedding(encoder_inputs))
    encoder_outputs, encoder_state_h, encoder_state_c = encoder_lstm(encoder_inputs)
    encoder_states = [encoder_state_h, encoder_state_c]

    decoder_inputs = Input(shape=(None, text_data.num_decoder_tokens), name='decoder_inputs')
    decoder_lstm = LSTM(units=hidden_units, return_state=True, return_sequences=True, name='decoder_lstm')
    decoder_outputs, decoder_state_h, decoder_state_c = decoder_lstm(decoder_inputs,
                                                                     initial_state=encoder_states)
    decoder_dense = Dense(units=text_data.num_decoder_tokens, activation='softmax', name='decoder_dense')
    decoder_outputs = decoder_dense(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    return model


# In[45]:


text_data = load_data()
text_data = encode_raw_text_data(text_data)


# In[47]:


print("HW Q1: Build a loop that prints the first 10 text_data.input_texts and text_data.target_texts")
print('')
print("text_data.input_texts")
print('')
for line in text_data.input_texts[:10]:
    print(line)
print('')
print("text_data.target_texts")
print('')
for line in text_data.target_texts[:10]:
    print(line)


# In[48]:


print("HW Q2: print the content of encoder_input_data[0]")
print(text_data.encoder_input_data[0])


# In[55]:


print("HW Q3: use text_data.input_idx2word to translate encoder_input_data[0] back to english words")
hw3_decoder = []
for item in text_data.encoder_input_data[0]:
    hw3_decoder.append(text_data.input_idx2word.get(item))
print(hw3_decoder)


# In[57]:


text_data.save()


# In[58]:


model = build_model(text_data)


# In[ ]:


print('HW4: Explain the model architecture in your own words')
print('Part 1 of the model is the encoding, this multi-layer part produced a fixed length encoding of the text_data.')
print('The input here is the ')
print('Part 2 of the model is the decoding, this multi-layer part produces the prediction for the output sequence.')
print('The  input is the encoder state and fixed length encoding')
print('Part 3 is tying the model together')


# In[ ]:


print('HW5: How does the encoder condition the decoder? What are the inputs and outputs to the decoder?')
print('The encoder creates a fixed length encoding, which is then used an the initial state of the decoder. Giving the decoder context')
print('In addition the decoder takes in the number of decode tokens, different here from encoder tokens by 2 additional tokens and the same number of hidden units.')


# In[59]:


X_train, X_test, y_train, y_test = train_test_split(text_data.encoder_input_data, text_data.target_texts, test_size=0.2, random_state=42)

train_gen = generate_batch(X_train, y_train, text_data)
test_gen = generate_batch(X_test, y_test, text_data)

BATCH_SIZE=128
train_num_batches = len(X_train) // BATCH_SIZE
test_num_batches = len(X_test) // BATCH_SIZE


# In[60]:


TENSORBOARD = 'TensorBoard/'
WEIGHT_FILE_PATH = 'word-weights.h5'
checkpoint = ModelCheckpoint(filepath=WEIGHT_FILE_PATH, save_best_only=True)
tbCallBack = TensorBoard(log_dir=TENSORBOARD, histogram_freq=0, write_graph=True, write_images=True)


# In[61]:


model.fit_generator(generator=train_gen,
                    steps_per_epoch=train_num_batches,
                    epochs=10,
                    verbose=1,
                    validation_data=test_gen,
                    validation_steps=test_num_batches
                    # ,callbacks=[checkpoint, tbCallBack ]
                    )


# In[62]:


WEIGHT_FILE_PATH = 'weights.h5'
model.save_weights(WEIGHT_FILE_PATH)


# In[ ]:




