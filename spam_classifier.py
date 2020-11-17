import os
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
import keras
from keras.callbacks import EarlyStopping
import pickle
import logging
from absl import flags
from absl import app

FLAGS = flags.FLAGS
logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=os.environ.get("LOGLEVEL", "INFO"))

flags.DEFINE_integer(
    "train_batch_size", 32, "Total batch size for training"
)

flags.DEFINE_integer(
    "train_epochs", 20, "Total batch size for training"
)

flags.DEFINE_float(
    "train_dropout", 0.1, "Dropout layer "
)

flags.DEFINE_string(
    "train_optimizer", None,
    "Optimizer to be used for model training.")

flags.DEFINE_string(
    "train_loss_fn", None,
    "Loss function to be used for model training.")

flags.DEFINE_string(
    "dense_layer_activation", None,
    "Loss function to be used for model training.")


flags.DEFINE_string(
    "output_layer_activation", None,
    "Loss function to be used for model training.")
    

class DataFormatting():

    def __init__(self, text):
        logging.info('Init run')
        self.text = text

    def remove_punc(self, text):
        return ("".join([ch for ch in text if ch not in string.punctuation]))

        
    def lower_txt(self,text):
        text = text.split(" ")
        return [x.lower() for x in  text]

    def remove_stopwords(self,text):
        return [word for word in text if word not in nltk.corpus.stopwords.words("english")]

    def stemming(self,text):
        ps = PorterStemmer()
        return [ps.stem(word) for word in text]


    def lemmatizing(self, text):
        wl = WordNetLemmatizer()
        return [wl.lemmatize(word) for word in text]
    
    # def __repr__(self):
    #     return repr(self.text)
    def preprocessing(self):
        logging.info('preprocessing started')
        logging.info('removing punc')
        self.text = self.text.apply(lambda ch: self.remove_punc(ch))   
        logging.info('lowering text')
        self.text = self.text.apply(lambda x: self.lower_txt(x))    
        logging.info('removing stopword')
        self.text = self.text.apply(lambda word: self.remove_stopwords(word))   
        logging.info('stemming')
        self.text = self.text.apply(lambda word: self.stemming(word)) 
        return self.text

class Keras_lstm():

    def label_encoder(self, labels):
        encoder = LabelEncoder()
        Y = encoder.fit_transform(labels)
        return Y.reshape(-1,1)

    def tokenization(self, x_train, max_words):
        tok = Tokenizer(num_words= max_words)
        tok.fit_on_texts(x_train)
        with open('tokenizer.pickle', 'wb') as handle:
            pickle.dump(tok, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return tok.texts_to_sequences(x_train)

    def padding(self, sentences, max_len):
        return pad_sequences(sentences, maxlen= max_len)

    def keras_lstm(self, max_len, max_words):
        model = Sequential()
        model.add(Embedding(max_words, 50, input_length=max_len))
        model.add(LSTM(64))
        model.add(Dense(256, activation= FLAGS.dense_layer_activation))
        model.add(Dropout(FLAGS.train_dropout))
        model.add(Dense(1, activation= FLAGS.output_layer_activation))
        model.summary()

        input= Input(shape= [max_len])
        output = model(input)
        return Model(input, output)

    def model_training(self, padded_sequence, Y_train, model):
        history = model.fit(padded_sequence, Y_train,
                            batch_size=FLAGS.train_batch_size,
                            epochs=FLAGS.train_epochs,
                            validation_split=0.1,
                            callbacks=[EarlyStopping(monitor='val_loss',
                                                                    min_delta=0,
                                                                    patience=10,
                                                                    verbose=0,
                                                                    mode='min')])
                                                                    
    def model_save(self, model):
        model.save("model.h5")
        
def main(argv):
    df = pd.read_csv("spam_data.csv")
    
    processor = DataFormatting(df['Message'])

    X = processor.preprocessing()
    logging.info('preprocessing done')

    model_fn = Keras_lstm()

    Y = model_fn.label_encoder(df['Category'])

    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.15)
    f'split traintest done'

    tokenized_text = model_fn.tokenization(X_train, 1000)
    padded_sequence = model_fn.padding(tokenized_text, 200)

    keras_model = model_fn.keras_lstm(200,1000)
    keras_model.compile(loss=FLAGS.train_loss_fn, optimizer=FLAGS.train_optimizer, metrics=['accuracy'])
    model_fn.model_training(padded_sequence, Y_train,keras_model)
    model_fn.model_save(keras_model)



if __name__ == '__main__':
    app.run(main)


    
