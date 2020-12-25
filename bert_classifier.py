# !pip install transformers
from transformers import TFBertModel,  BertConfig, BertTokenizerFast
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.utils import to_categorical
import os
from nltk import metrics
import pandas as pd
import nltk
import string
from nltk import PorterStemmer, WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.models import Sequential,Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
import keras
from keras.preprocessing import sequence
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, RMSprop
from keras.callbacks import EarlyStopping
import pickle
import logging
from absl import flags
from absl import app
from tensorflow.python.keras.layers.core import Flatten


df = pd.read_csv("spam_data.csv")
df.dropna(inplace= True)
df['Category'] = pd.Categorical(df['Category'])

model_name = 'bert-base-uncased'
max_length = 100

config = BertConfig.from_pretrained(model_name)
config.output_hidden_states = False

tokenizer =  BertTokenizerFast.from_pretrained(pretrained_model_name_or_path=model_name, config = config)

transformer_model = TFBertModel.from_pretrained(model_name, config= config)

bert = transformer_model.layers[0]
input_ids = Input(shape=(max_length,), name ='input_ids', dtype='int32')
inputs= {'input_ids': input_ids}

bert_model = bert(inputs)[1]
dropout = Dropout(config.hidden_dropout_prob, name='pooled_output')
pooled_output = dropout(bert_model, training=False)
# Then build your model output
issue = Dense(units=1, kernel_initializer=TruncatedNormal(stddev=config.initializer_range), name='issue')(pooled_output)
# product = Dense(units=len(df['Category'].value_counts()), kernel_initializer=TruncatedNormal(stddev=config.initializer_range), name='product')(pooled_output)
outputs = {'issue': issue}
# And combine it all in a model object
model = Model(inputs=inputs, outputs=outputs, name='BERT_MultiLabel_MultiClass')
# Take a look at the model
model.summary()

label_encoder = LabelEncoder()


# Set an optimizer
optimizer = Adam(
learning_rate=5e-05,
epsilon=1e-08,
decay=0.01,
clipnorm=1.0)
# Set loss and metrics
loss = {'issue': BinaryCrossentropy(from_logits = True)}
metric = {'issue': CategoricalAccuracy('accuracy')}
# Compile the model
model.compile(
optimizer = optimizer,
loss = loss, 
metrics = metric)
# Ready output data for the model
y_issue = label_encoder.fit_transform(df['Category'])
y_issue.reshape(-1,1)
# y_product = to_categorical(df['Category'])
# Tokenize the input (takes some time)
x = tokenizer(
text=df['Message'].to_list(),
add_special_tokens=True,
max_length=max_length,
truncation=True,
padding=True, 
return_tensors='tf',
return_token_type_ids = False,
return_attention_mask = False,
verbose = True)
# Fit the model
history = model.fit(
x={'input_ids': x['input_ids']},
y={'issue': y_issue},
validation_split=0.2,
batch_size=64,
epochs=10)