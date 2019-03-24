# general imports
import math
import numpy as np
import csv
from nltk.corpus import stopwords
import re
from sklearn.model_selection import train_test_split
import pickle
import json
# keras imports
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv1D, MaxPooling1D

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
    labels = np.array(labels)
    tweets = np.array(tweets)
    return tweets_df, vocab, tweets, labels


def get_embeddings():
    tweets, vocab, X, y = read_in_data()
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
    X_train, X_test, y_train, y_test = train_test_split(all_embeddings, y, test_size=0.3, random_state=42)
    X_train = np.expand_dims(X_train, axis=2)
    X_test = np.expand_dims(X_test, axis=2)

    model = Sequential()
    model.add(Conv1D(64, 5, activation='relu', input_shape=(2400,1)))
    model.add(MaxPooling1D(pool_size=4))
    model.add(Flatten())
    model.add(Dense(3, activation='sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(x=X_train, y=y_train, epochs=5)
    score = model.evaluate(X_test, y_test)
    # serialize model to JSON
    # model_json = model.to_json()
    # with open("model.json", "w") as json_file:
    #     json_file.write(model_json)

    # save to h5df
    model.save("model.hdf5")
    # serialize weights to HDF5
    model.save_weights("model_weights.h5")
    print("saved model to disk")
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