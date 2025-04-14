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

# === 4. Modelos de clasificación === (Omi si ves esto eres gay)
clasificadores = {
    'Naive Bayes': MultinomialNB(),
    'Regresión Logística': LogisticRegression(max_iter=300),
    'MLP': MLPClassifier(hidden_layer_sizes=(200, 100), max_iter=300),
    'SVM': SVC()
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
