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
# from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences
# from keras.models import Sequential
# from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
# from keras.layers.embeddings import Embedding

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
            tweets.append(' '.join(filtered))
            labels.append(row[5])
    labels = np.array(labels)
    tweets = np.array(tweets)
    return tweets_df, vocab, tweets, labels


def get_embeddings():
    tweets, vocab, X, y = read_in_data()
    sum = 0
    count = 0
    for t in tweets:
        sum += len(t['tweet'].split())
        count += 1

    print(sum)
    print(count)
    print(sum/count)
    vocab_size = len(vocab)
    embeddings_index = {}
    f = open('glove.twitter.27B.100d.txt', 'r', encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        embedding = np.asarray([float(val) for val in values[1:]])
        embeddings_index[word] = embedding
    # embedding_matrix = np.zeros((vocab_size, 100))
    # for word, index in tokenizer.word_index.items():
    #     if index > vocab_size - 1:
    #         break
    #     else:
    #         embedding_vector = embeddings_index.get(word)
    #         if embedding_vector is not None:
    #             embedding_matrix[index] = embedding_vector
    # for i, word in enumerate(X):
    #     try:
    #         embedding_matrix[i] = embeddings_index[word]
    #     except KeyError:
    #         embedding_matrix[i] = np.random.normal(scale=0.6, size=(100,))
    for tweet in X:
        for word in tweet:
            if word not in embeddings_index:
                embeddings_index[word] = np.random.normal(scale=0.6, size=(100,))
    all_embeddings = []
    for tweet in X:
        for i in range(len(tweet)):
            print(tweet[i])
        # empty = np.zeros((max_len, 100))
        # empty = np.zeros((len(tweet),100))
        # for i in range(len(tweet)):
        #     empty[i] = embeddings_index[tweet[i]]
        # print(empty)
        # front = max(0,math.floor((24 - len(tweet))/2))
        # back = max(0,math.ceil((24 - len(tweet))/2))
        # print(front)
        # print(back)
        # empty = np.pad(empty, [(front, back),(0,0)], 'constant')
        # print(empty)
        all_embeddings.append(empty)

    print("______________________")
    for em in all_embeddings:
        print(em.shape)
    print("__________________________")
    all_embeddings = np.array(all_embeddings)
    # for em in all_embeddings:
        # print(em.shape)


    #print(embedding_matrix.shape)
    #print(y.shape)

    # X_train, X_test, y_train, y_test = train_test_split(embedding_matrix, y, test_size=0.3, random_state=42)
    #
    # model = Sequential()
    # model.add(Embedding(vocab_size, 100, input_length=50,
    #                     weights=[embedding_matrix], trainable=False))
    # model.add(Dropout(0.2))
    # model.add(Conv1D(64, 5, activation='relu'))
    # model.add(MaxPooling1D(pool_size=4))
    # model.add(LSTM(100))
    # model.add(Dense(1, activation='sigmoid'))
    # model.compile(loss='crossentropy', optimizer='adam', metrics=['accuracy'])
    # model.fit(x=X_train, y=y_train, epochs=5)
    # score = model.evaluate(X_test, y_test)
    # hypothesis = model.predict(X_test)
    # print(score)

def main():
    label_value = {
        0: 'hate speech',
        1: 'offensive language',
        2: 'clean'
    }
    get_embeddings()


if __name__ == '__main__':
    main()