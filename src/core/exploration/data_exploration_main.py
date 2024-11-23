import pandas as pd
import numpy as np
from collections import Counter

# Carregar seus dados
from behavior.behavior_data_loader import BehaviorDataLoader

data_path = 'data/new_logs_labels.csv'

df = BehaviorDataLoader.load_data(data_path, delimiter=';')
#df.head(5)

# 1. Informações básicas do dataset
print("=== Informações Básicas ===")
print(df.info())
print("\n=== Primeiras linhas ===")
print(df.head())

# 2. Distribuição das classes de comportamento
print("\n=== Distribuição das Classes de Comportamento ===")
print(df['comportamento'].value_counts(normalize=True))

# 3. Quantidade de instâncias por aluno
print("\n=== Distribuição de instâncias por aluno ===")
alunos_count = df['aluno'].value_counts()
print(f"Média de instâncias por aluno: {alunos_count.mean():.2f}")
print(f"Mínimo: {alunos_count.min()}")
print(f"Máximo: {alunos_count.max()}")

# 4. Verificar features disponíveis
print("\n=== Grupos de Features ===")
comportamento_cols = [col for col in df.columns if 'comportamento' in col]
traco_cols = [col for col in df.columns if 'traco_' in col]
print(f"Features de comportamento: {comportamento_cols}")
print(f"Features de traços: {traco_cols}")
print(f"Outras features: {[col for col in df.columns if col not in comportamento_cols + traco_cols]}")

# 5. Verificar valores ausentes
print("\n=== Valores Ausentes ===")
missing_values = df.isnull().sum()
if missing_values.any():
    print(missing_values[missing_values > 0])
else:
    print("Não há valores ausentes no dataset")