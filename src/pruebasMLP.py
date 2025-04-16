from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
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
    print(f"\nProbando con vectorización: {nombre}")

    pipeline = Pipeline([
        ('vect', vectorizador),
        ('clf', MLPClassifier(
            hidden_layer_sizes=(300, 150),
            activation='relu',
            solver='adam',
            learning_rate_init=0.0005,
            max_iter=600,
            early_stopping=False,
            random_state=42
        ))
    ])

    param_grid = {
        'clf__hidden_layer_sizes': [(100,), (300, 150)],
        'clf__activation': ['relu', 'tanh'],
        'clf__learning_rate_init': [0.001, 0.0005],
    }

    grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1_macro', n_jobs=-1)
    grid.fit(textos, y)

    print("Mejores parámetros:", grid.best_params_)
    print("Mejor F1-macro score:", grid.best_score_)
