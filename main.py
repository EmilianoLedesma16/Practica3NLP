'''
    Script para reordenar las columnas (temporal)
'''

import pandas as pd

# Cargar el CSV
df = pd.read_csv('data/arxiv_corpus_features.csv')

# Mostrar las columnas actuales (opcional para confirmar orden)
print("Columnas originales:", df.columns.tolist())

# Mover 'Section' al final
columnas = [col for col in df.columns if col != 'Section'] + ['Section']
df = df[columnas]

# Guardar el nuevo CSV
df.to_csv('data/arxiv_raw_corpus_reordered.csv', index=False)

print("Archivo guardado con 'Section' al final.")
