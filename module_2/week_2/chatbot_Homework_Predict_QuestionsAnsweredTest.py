#!/usr/bin/env python
# coding: utf-8

# In[17]:


from keras.models import Model, model_from_json
from keras.layers import Input, LSTM, Dense, Embedding
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import nltk
import pickle
from pathlib import Path

import os 
os.chdir(Path("c:/dev/advanced_neural_networks/module_2/week_2"))


# In[18]:


class TextData:
    """data class for text data"""
    
    def __init__(self, input_texts, target_texts, input_counter, target_counter):
        self.input_texts = input_texts
        self.target_texts = target_texts
        self.input_counter = input_counter
        self.target_counter = target_counter

        
    def save(self):
        self.filename = "./model/text_data.pkl"
        pickle.dump(self, open(self.filename,'wb'))
    
    @classmethod
    def load(self):
        self.filename = "./model/text_data.pkl"
        return pickle.load(open(self.filename,'rb'))


# In[19]:


class ChatBot():
    """
    This is ChatBot class it takes weights for the Neural Network, compliling model
    and returns prediction in responce to input text
    """
    def __init__(self):
        """
        define all required parameters, rebuild model and load weights
        """
        
        self.text_data = TextData.load()
        self.hidden_units=256

        encoder_inputs = Input(shape=(None, ), name='encoder_inputs')
        encoder_embedding = Embedding(input_dim=self.text_data.num_encoder_tokens, output_dim=self.hidden_units,
                                      input_length=self.text_data.encoder_max_seq_length, name='encoder_embedding')
        encoder_lstm = LSTM(units=self.hidden_units, return_state=True, name="encoder_lstm")
        encoder_outputs, encoder_state_h, encoder_state_c = encoder_lstm(encoder_embedding(encoder_inputs))
        encoder_states = [encoder_state_h, encoder_state_c]

        decoder_inputs = Input(shape=(None, self.text_data.num_decoder_tokens), name='decoder_inputs')
        decoder_lstm = LSTM(units=self.hidden_units, return_sequences=True, return_state=True, name='decoder_lstm')
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
        decoder_dense = Dense(self.text_data.num_decoder_tokens, activation='softmax', name='decoder_dense')
        decoder_outputs = decoder_dense(decoder_outputs)

        self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        self.model.load_weights('model/word-weights.h5')
        self.model.compile(optimizer='adam', loss='categorical_crossentropy')

        self.encoder_model = Model(encoder_inputs, encoder_states)

        decoder_state_inputs = [Input(shape=(self.hidden_units,)), Input(shape=(self.hidden_units,))]
        decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_state_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        self.decoder_model = Model([decoder_inputs] + decoder_state_inputs, [decoder_outputs] + decoder_states)

    def reply(self, input_text):
        """
        Takes input_text and return predicted responce
        :param input_text: string
        :return: predicted_text: string
        """

        input_seq = []
        input_wids = []
        for word in nltk.word_tokenize(input_text.lower()):
            idx = 1
            if word in self.text_data.input_word2idx:
                idx = self.text_data.input_word2idx[word]
            input_wids.append(idx)
        input_seq.append(input_wids)
        input_seq = pad_sequences(input_seq, self.text_data.encoder_max_seq_length)
        states_value = self.encoder_model.predict(input_seq)
        target_seq = np.zeros((1, 1, self.text_data.num_decoder_tokens))
        target_seq[0, 0, self.text_data.target_word2idx['START']] = 1
        target_text = ''
        target_text_len = 0
        terminated = False
        while not terminated:
            output_tokens, h, c = self.decoder_model.predict([target_seq] + states_value)
            sample_token_idx = np.argmax(output_tokens[0, -1, :])
            sample_word = self.text_data.target_idx2word[sample_token_idx]
            target_text_len += 1

            if sample_word != 'START' and sample_word != 'END':
                target_text += ' ' + sample_word

            if sample_word == 'END' or target_text_len >= self.text_data.decoder_max_seq_length:
                terminated = True

            target_seq = np.zeros((1, 1, self.text_data.num_decoder_tokens))
            target_seq[0, 0, sample_token_idx] = 1

            states_value = [h, c]
        return target_text.strip().replace('UNK', '')


# In[20]:


bot = ChatBot()


# In[21]:


bot.reply("hello")
bot.reply("whats up bro")

# In[22]:


print('HW 6: Ask your chatbot 5 questions. How does it work?')
bot.reply("How does it work?")


# In[23]:


bot.reply("whats up")
bot.reply("Can you say anything besides a long space")


# In[ ]:




