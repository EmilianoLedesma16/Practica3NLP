from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd
import os

# Cargar datos
datapath = os.path.join('..', 'data', 'arxiv_normalized_corpus.csv')
df = pd.read_csv(datapath, sep='\t', encoding='utf-8-sig')

textos = df['Normalized_text']
y = df['Section']

# Vectorizadores
binary_vectorizer = CountVectorizer(binary=True)
frequency_vectorizer = CountVectorizer()
tfidf_vectorizer = TfidfVectorizer()

pipelines = [
    ('binario', binary_vectorizer),
    ('frecuencia', frequency_vectorizer),
    ('tfidf', tfidf_vectorizer)
]

for nombre, vectorizador in pipelines:
    print(f"\nProbando SVC con vectorización: {nombre}")

    pipeline = Pipeline([
        ('vect', vectorizador),
        ('clf', SVC(
            kernel='linear',
            max_iter=2000
        ))
    ])

    param_grid = {
        'clf__C': [0.1, 1.0, 2.0, 5.0]
    }

    grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1_macro', n_jobs=-1)
    grid.fit(textos, y)

    print("Mejor C:", grid.best_params_['clf__C'])
    print("Mejor F1-macro score:", grid.best_score_)
