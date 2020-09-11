import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import re
from nltk.stem import WordNetLemmatizer
import numpy as np

path_to_dataset = os.path.abspath(os.path.join('code', 'dataset', 'CDs_and_Vinyl_5.json'))
data = pd.read_json(path_to_dataset, lines=True)

def prepare_data(size, rating_range=(0, 5), train_size = 0.8, simplify_ratings=False):
    sample = None
    if rating_range != (0, 5):
        data_in_rating_range = data[(data['overall'] >= rating_range[0]) & (data['overall'] <= rating_range[1])]
        sample = data_in_rating_range.sample(n=min(size, data_in_rating_range.shape[0]), random_state=57)
    else:
         sample = data.sample(n=min(size, data.shape[0]))
    sample = sample[['reviewText', 'overall']]
    if simplify_ratings:
        sample['overall'] = (sample['overall'] > (rating_range[0] + rating_range[1]) / 2).astype(int)
    sample['reviewText'] = [re.sub('[.;:!?,()\[\]\'\"]', ' ', review.lower()) for review in sample['reviewText']]
    lemmatizer = WordNetLemmatizer()
    sample['reviewText'] = [' '.join([lemmatizer.lemmatize(word) for word in review.split()]) for review in sample['reviewText']]
    vectorizer = CountVectorizer(binary=True, stop_words=['in','of','at','a','the'], dtype=np.int8)
    x = vectorizer.fit_transform(sample['reviewText']).toarray()
    return train_test_split(x, sample['overall'], train_size=train_size)

def train_simple_model(size=data.shape[0]):
    x_train, x_test, y_train, y_test = prepare_data(size)
    grid = {'C': [0.01, 0.1, 0.5, 1.0]}
    regression = LogisticRegression()
    grid_search = GridSearchCV(regression, grid)
    grid_search.fit(x_train, y_train)
    print("tuned hpyerparameters :(best parameters) ",grid_search.best_params_)
    print("accuracy :",grid_search.best_score_)

train_simple_model(size=5000)