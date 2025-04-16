from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import os

datapath = os.path.join('..', 'data', 'arxiv_normalized_corpus.csv')
df = pd.read_csv(datapath, sep='\t', encoding='utf-8-sig')

textos = df['Normalized_text']
y = df['Section']

# Vectorizador binario (solo 0 o 1)
binary_vectorizer = CountVectorizer(binary=True)

# Vectorizador por frecuencia (conteo simple)
frequency_vectorizer = CountVectorizer()

# Vectorizador TF-IDF
tfidf_vectorizer = TfidfVectorizer()

pipelines = [
    ('binario', binary_vectorizer),
    ('frecuencia', frequency_vectorizer),
    ('tfidf', tfidf_vectorizer)
]

for nombre, vectorizador in pipelines:
    print(f"Probando con vectorización: {nombre}")

    pipeline = Pipeline([
        ('vect', vectorizador),
        ('clf', LogisticRegression(solver='lbfgs', max_iter=600))
    ])

    param_grid  = {
        'clf__C' : [0.01,0.1,1.0,2.0,5.0]
    }

    grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1_macro')
    grid.fit(textos, y)

    print("Mejor C: ", grid.best_params_['clf__C'])
    print("Mejor score: ", grid.best_score_)