# general imports
import math
import numpy as np
import csv
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from sklearn.model_selection import train_test_split
import pandas as pd
# keras imports
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from keras.layers.embeddings import Embedding

def read_in_data():
    """
    reads in csv file containing labeled tweets and processes into list
    of dictionary with tweet id, text and label
    """

    tweets_df = []
    vocab = {}
    tweets = []
    labels = []

    stop_words = set(stopwords.words('english'))

    with open('data/labeled_data.csv') as f:
        csv_reader = csv.reader(f, delimiter=',')
        for row in csv_reader:
            tweet = re.sub('[^a-z A-Z@]+','',row[6].lower())
            tweet_tokens = tweet.split()  # word_tokenize(tweet)
            filtered = [w for w in tweet_tokens if w not in stop_words and w != 'rt'
                        and 'http' not in w and '@' not in w]
            for w in filtered:
                if w in vocab:
                    vocab[w] += 1
                else:
                    vocab[w] = 1
            tweets_df.append({
                'id': row[0],
                'tweet': ' '.join(filtered),
                'label': row[5]
            })
            tweets.append(filtered)
            if row[5] == '0':
                labels.append([1,0,0])
            elif row[5] == '1':
                labels.append([0,1,0])
            elif row[5] == '2':
                labels.append([0,0,1])
    print(labels)
    labels = np.array(labels)
    print(labels)
    tweets = np.array(tweets)
    return tweets_df, vocab, tweets, labels


def get_embeddings():
    tweets, vocab, X, y = read_in_data()
    # sum = 0
    # count = 0
    # for t in tweets:
    #     sum += len(t['tweet'].split())
    #     count += 1
    # print(sum)
    # print(count)
    # print(sum/count)
    vocab_size = len(vocab)
    embeddings_index = {}
    f = open('glove.twitter.27B.100d.txt', 'r', encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        embedding = np.asarray([float(val) for val in values[1:]])
        embeddings_index[word] = embedding

    for tweet in X:
        for word in tweet:
            if word not in embeddings_index:
                embeddings_index[word] = np.random.normal(scale=0.6, size=(100,))
    all_embeddings = []
    for tweet in X:
        vec = np.zeros((len(tweet),100))
        for i in range(len(tweet)):
             vec[i] = embeddings_index[tweet[i]]
        front = max(0,math.floor((24 - len(tweet))/2))
        back = max(0,math.ceil((24 - len(tweet))/2))
        vec = np.pad(vec, [(front, back),(0,0)], 'constant')
        all_embeddings.append(vec.flatten())

    all_embeddings = np.array(all_embeddings)
    print('++++++++++++++++++++++++++++++')
    print(all_embeddings.shape)
    print(y.shape)
    print('++++++++++++++++++++++++++++++')
    X_train, X_test, y_train, y_test = train_test_split(all_embeddings, y, test_size=0.3, random_state=42)
    X_train = np.expand_dims(X_train, axis=2)
    X_test = np.expand_dims(X_test, axis=2)
    # np.reshape(X_train, (1, X_train.shape[0], X_train.shape[1]))
    # np.reshape(X_test, (1, X_test.shape[0], X_test.shape[1]))

    model = Sequential()
    model.add(Conv1D(64, 5, activation='relu', input_shape=(2400,1)))
    # model.add(Conv1D(64, 5, activation='relu'))
    model.add(MaxPooling1D(pool_size=4))
    model.add(Flatten())
    model.add(Dense(3, activation='sigmoid'))
    # model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(x=X_train, y=y_train, epochs=5)
    score = model.evaluate(X_test, y_test)
    hypothesis = model.predict(X_test)
    print(score)

def main():
    label_value = {
        0: 'hate speech',
        1: 'offensive language',
        2: 'clean'
    }
    get_embeddings()


if __name__ == '__main__':
    main()