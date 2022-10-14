import pandas as pd
from app import config as c
import re
from nltk.tokenize.casual import TweetTokenizer
import string
import numpy as np
import pickle
from tensorflow.keras.models import load_model


def load_data():
    file_name = 'data/test_set.csv'
    df = pd.read_csv(file_name)
    return df


def get_user_preds(messages, request_form):
    """Get user predictions from POST and do some basic cleansing"""
    user_preds = []

    for i in range(len(messages)):
        i += 1  # match the loop.index in jinja
        pred = request_form.get(f'answer_{i}')

        try:
            pred = int(pred) - 1  # rematch original numbering
            if pred not in c.SENDER_MAPPER_REV.keys():  # if not in the list of allowed senders, return empty string
                pred = ''
        except:
            pred = ''

        user_preds.append(str(pred))

    return user_preds


def compute_user_score(user_preds, senders):
    score_user = 0

    for i, p in enumerate(user_preds):
        sender_mapper = c.SENDER_MAPPER_REV
        try:
            p = sender_mapper[int(p)]
            if p == senders.iloc[i]:
                score_user += 1
        except:
            pass

    return score_user


def split_user_preds(user_preds, num_samples):

    # The user predictions will be passed as a string with each input separated by double quotes
    # Get the position of the double quotes and then extract the substring in the middle
    positions_quotes = [pos for pos, char in enumerate(user_preds) if char == '"']
    preds_list = [user_preds[i[0]+1:i[1]] for i in chunks(positions_quotes, 2)]
    # assert len(preds_list) == num_samples

    # Convert components of preds_list to integers (they have already been checked to be in range of permitted values)
    preds_cleansed = []
    for i in range(len(preds_list)):
        try:
            intg = int(preds_list[i])
            preds_cleansed.append(intg)
        except:
            preds_cleansed.append('')

    return preds_cleansed


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def process_message(text):
    """Simple processor to prepare sentence for the NN model"""

    # replace URLs (if there are any left)
    text = re.sub(r'https?:\/\/.*[\r\n]*', '<<LINK>>', text)

    # tokenize, lower case and remove punctuation
    text_l = [item.lower() for item in TweetTokenizer().tokenize(text) if item not in string.punctuation]

    return text_l


def load_embedding_vocab():
    with open('data/embedding_vocab.pkl', 'rb') as handle:
        embedding_vocab = pickle.load(handle)
    return embedding_vocab


def get_nn_predictions(sentence, embedding_vocab, model, sender_mapper_reverse):
    """Get input sentence, cleanse it and prepare it for the model predictions and return output prob distribution"""
    max_length = 30

    # sanitize
    sentence = process_message(sentence)

    # if there are no sanitized words, make a list of 0s and
    # add it to X
    if len(sentence) == 0:
        x = np.array([0] * max_length)
    else:

        word_ids = []

        for word in sentence:

            # if it is a rare word, set it to the placeholder value
            if word not in embedding_vocab.keys():
                word = '<<RARE>>'

            # append the id of the word to the list
            word_ids.append(embedding_vocab[word])

        # padding
        if len(word_ids) < max_length:
            word_ids = word_ids + [0] * (max_length - len(word_ids))
        # shorten the sentence to the first max_length words
        elif len(word_ids) > max_length:
            word_ids = word_ids[:max_length]

        # add to x
        x = np.array(word_ids)

    x.shape = (max_length, 1)
    preds = model.predict(x.T).squeeze()

    # get the chosen sender (i.e. max estimated prob)
    chosen_sender = sender_mapper_reverse[preds.argmax()]

    # get whole distribution and return it
    full_distr = {sender_mapper_reverse[i]:p for i, p in enumerate(preds)}

    return chosen_sender, full_distr


def load_nn_model():
    return load_model('data/nn_classifier/')



