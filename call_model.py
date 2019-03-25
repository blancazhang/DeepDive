from nltk.corpus import stopwords
import twitter
import re
import numpy as np
from keras.models import load_model
import math
import sys


def pull_tweets(username):
    CONSUMER_KEY = 'OhBsZtfygfVk7TcOJexnCPeeE'
    CONSUMER_SECRET ='6omHgLFHvFCWqqnB5LP5PKEhocE2b4VkySrUBL57OcrXJzVrKe'
    OAUTH_TOKEN = '103314349-FfrgieFfppIyFtOsIl8skpPr9gas5eWL5TZOjIfF'
    OAUTH_TOKEN_SECRET = '2P5dgpFcht5rh31Nq8CXiYZyBC2JcHpZLcG4MUBFdjGlb'
    auth = twitter.oauth.OAuth(OAUTH_TOKEN, OAUTH_TOKEN_SECRET,
                           CONSUMER_KEY, CONSUMER_SECRET)
    twitter_api = twitter.Twitter(auth=auth)
    tl = twitter_api.statuses.user_timeline(screen_name=username, count=200)
    tweetlist = []
    for tweet in tl:
        tweetlist.append(tweet['text'])
    return tweetlist


def process_tweets(tweetlist):
    tweets = []
    stop_words = set(stopwords.words('english'))
    embeddings_index = {}
    f = open('glove.twitter.27B.100d.txt', 'r', encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        embedding = np.asarray([float(val) for val in values[1:]])
        embeddings_index[word] = embedding

    for tweet in tweetlist:
        tweet = re.sub('[^a-z A-Z@]+','',tweet.lower())
        tweet_tokens = tweet.split()
        filtered = [w for w in tweet_tokens if w not in stop_words and w != 'rt'
                    and 'http' not in w and '@' not in w]
        tweets.append(filtered)
    tweets = np.array(tweets)

    embeddings = []
    for tweet in tweets:
        vec = np.zeros((len(tweet),100))
        for i in range(len(tweet)):
            if tweet[i] in embeddings_index:
                vec[i] = embeddings_index[tweet[i]]
            else:
                vec[i] = np.random.normal(scale=0.6,size=(100,))
        front = max(0, math.floor((24 - len(tweet)) / 2))
        back = max(0, math.ceil((24 - len(tweet)) / 2))
        vec = np.pad(vec, [(front, back), (0, 0)], 'constant')
        embeddings.append(vec.flatten())

    embeddings = np.array(embeddings)

    final_test = np.expand_dims(embeddings, axis=2)

    return final_test


def test_tweets(tweet_embeddings):
    model = load_model('model.hdf5')
    model.load_weights('model_weights.h5')
    print('loaded model from disk')
    # evaluate loaded model on new data
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    hypothesis = model.predict(tweet_embeddings)
    return hypothesis


def main():
    username = sys.argv[1]
    pulled = pull_tweets(username)
    print("loaded %d tweets" % (len(pulled)))
    # trumpdata = np.load('trump.npy')
    # user1data = np.load('user1.npy')
    # user2data = np.load('user2.npy')
    embeddings = process_tweets(pulled)
    hypothesis = test_tweets(embeddings)
    hypothesis = hypothesis.tolist()
    labels = []
    hate_count = 0
    offensive_count = 0
    for h in hypothesis:
        labels.append(h.index(max(h)))
    for i in range(len(labels)):
        if labels[i] == 0:
            hate_count += 1
            print('hate speech detected')
            print(pulled[i])
            print("\n")
        elif labels[i] == 1:
            offensive_count += 1
            print('offensive language detected')
            print(pulled[i])
            print("\n")


if __name__ == '__main__':
    main()