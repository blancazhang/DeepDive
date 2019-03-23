import numpy as np
import csv
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

def read_in_data():
    """
    reads in csv file containing labeled tweets and processes into list of tuples
    """

    stop_words = set(stopwords.words('english'))

    with open('data/labeled_data.csv') as f:
        csv_reader = csv.reader(f, delimiter=',')
        for row in csv_reader:
            label = row[5]
            tweet = re.sub(r'[^\w\s]','',row[6].lower())
            print(tweet)
            tweet_tokens = word_tokenize(tweet)
            print(tweet_tokens)
            filtered = [w for w in tweet_tokens if w not in stop_words]
            print(filtered)


def main():
    print('test')
    read_in_data()


if __name__ == '__main__':
    main()