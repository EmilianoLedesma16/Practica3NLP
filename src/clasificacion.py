import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
import os

# === 1. Cargar el corpus normalizado ===
data_path = os.path.join('..', 'data', 'arxiv_normalized_corpus.csv')
df = pd.read_csv(data_path, sep='\t', encoding='utf-8-sig')

X = df['Normalized_text']
y = df['Section']

# === 2. Dividir los datos ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True, random_state=0
)

# === 3. Representaciones de texto ===
vectorizadores = {
    'Binaria': CountVectorizer(binary=True),
    'Frecuencia': CountVectorizer(),
    'TF-IDF': TfidfVectorizer()
}

# === 4. Modelos de clasificación ===
clasificadores = {
    # Bayes con menor suavizado, se recomiendan valores de entre 0.1-1.0
    'Naive Bayes': MultinomialNB(
        alpha = 0.1
    ),
    
    #Regresión logística con una mayor iteración y menor regularización
    'Regresión Logística': LogisticRegression(
        max_iter = 600,
        C = 0.1,
        solver = 'lbfgs'
    ),
    
    #Perceptrón Multicapa con capas más profundad y early stopping
    'MLP': MLPClassifier(
        hidden_layer_sizes = (300, 150), #se definen en este caso dos capas para la red neuronal
        activation = 'relu',
        solver = 'adam',
        learning_rate_init = 0.001,        
        max_iter = 600,
        early_stopping = False,
        random_state = 42
    ),
    
    #SVM con kernel lineal y menor regularización
    'SVM': SVC(
        kernel = 'linear',
        C = 1.0,
        max_iter = 2000
    )
}

# === 5. Entrenar, predecir y evaluar === (Rodas se la come)
resultados = []

for nombre_vec, vectorizador in vectorizadores.items():
    for nombre_clf, clf in clasificadores.items():
        print(f"\n=== {nombre_clf} con {nombre_vec} ===")
        pipe = Pipeline([
            ('vectorizador', vectorizador),
            ('clasificador', clf)
        ])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        reporte = classification_report(y_test, y_pred, output_dict=True)
        f1_macro = reporte['macro avg']['f1-score']

        print(classification_report(y_test, y_pred))

        resultados.append({
            'Metodo': nombre_clf,
            'Parametros': str(clf.get_params()),
            'Representacion': nombre_vec,
            'F1_macro': round(f1_macro, 4)
        })

# === 6. Guardar tabla de resultados ===
resultados_df = pd.DataFrame(resultados)
resultados_path = os.path.join('..', 'data', 'resultados_experimentos.csv')
resultados_df.to_csv(resultados_path, index=False)

print("\n=== Resultados resumidos guardados en resultados_experimentos.csv ===")
print(resultados_df)
