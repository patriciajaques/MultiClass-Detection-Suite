# behavior_detection.py

import os
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold

def setup_project_path(path_name) -> Dict[str, Path]:
    """Configura os caminhos do projeto para Colab ou local."""
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        base_path = Path(path_name)
        print("Executando no Google Colab")
    except:
        current_path = Path.cwd()
        while current_path.name != 'behavior-detection' and current_path.parent != current_path:
            current_path = current_path.parent
        
        if current_path.name != 'behavior-detection':
            raise FileNotFoundError("Diretório 'behavior-detection' não encontrado")
            
        base_path = current_path
        print("Executando localmente")
    
    paths = {
        'data': base_path / 'data',
        'output': base_path / 'output',
        'models': base_path / 'models',
        'src': base_path / 'src'
    }
    
    for path in paths.values():
        path.mkdir(exist_ok=True)
    
    import sys
    src_path = str(paths['src'])
    if src_path not in sys.path:
        sys.path.append(src_path)
    
    return paths

def load_and_preprocess_data(paths: Dict[str, Path], test_size: float = 0.8) -> tuple:
    """Carrega e pré-processa os dados."""
    from behavior.data.behavior_data_loader import BehaviorDataLoader
    from core.preprocessors.data_cleaner import DataCleaner
    
    # Carregar dados
    data = BehaviorDataLoader.load_data(paths['data'] / 'new_logs_labels.csv', delimiter=';')
    
    # Remover instâncias com comportamento indefinido
    data = DataCleaner.remove_instances_with_value(data, 'comportamento', '?')
    
    # Split inicial para teste
    data, _ = train_test_split(data, test_size=test_size, 
                              stratify=data['comportamento'], random_state=42)
    data.reset_index(drop=True, inplace=True)
    
    return data

def prepare_data(data: pd.DataFrame) -> tuple:
    """Prepara os dados para treinamento."""
    from core.preprocessors.data_splitter import DataSplitter
    from behavior.data.behavior_data_encoder import BehaviorDataEncoder
    
    # Split treino/teste
    train_data, test_data = DataSplitter.split_by_student_level(
        data, test_size=0.2, column_name='aluno'
    )
    
    # Remover coluna aluno
    train_data = train_data.drop('aluno', axis=1)
    test_data = test_data.drop('aluno', axis=1)
    
    # Split X/y
    X_train, y_train = DataSplitter.split_into_x_y(train_data, 'comportamento')
    X_test, y_test = DataSplitter.split_into_x_y(test_data, 'comportamento')
    
    # Encoding
    X_encoder = BehaviorDataEncoder(num_classes=5)
    X_encoder.fit(X_train)
    
    X_train = X_encoder.transform(X_train)
    X_test = X_encoder.transform(X_test)
    
    y_train = BehaviorDataEncoder.encode_y(y_train)
    y_test = BehaviorDataEncoder.encode_y(y_test)
    
    return X_train, X_test, y_train, y_test

def train_models(X_train, X_test, y_train, y_test, paths: Dict[str, Path]) -> None:
    """Treina os modelos em etapas."""
    from core.management.stage_training_manager import StageTrainingManager
    from core.models.multiclass.behavior_model_params import BehaviorModelParams
    
    # Configurações
    model_params = BehaviorModelParams()
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    # Definir etapas
    stages = [
        ('etapa1', ['Logistic Regression', 'Decision Tree'], 
         ['none', 'pca', 'rfe', 'rf', 'mi']),
        ('etapa2', ['Random Forest', 'Gradient Boosting'], 
         ['none', 'pca', 'rfe', 'rf', 'mi']),
        ('etapa3', ['SVM', 'KNN'], 
         ['none', 'pca', 'rfe', 'rf', 'mi']),
        ('etapa4', ['XGBoost', 'Naive Bayes', 'MLP'], 
         ['none', 'pca', 'rfe', 'rf', 'mi'])
    ]
    
    # Inicializar gerenciador
    training_manager = StageTrainingManager(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        model_params=model_params,
        n_iter=50,
        cv=cv,
        scoring='balanced_accuracy',
        n_jobs=6
    )
    
    # Executar treinamento
    try:
        training_manager.execute_all_stages(training_manager, stages)
        
        # Gerar relatórios finais
        from core.reporting import metrics_reporter
        final_results = training_manager.combine_results()
        training_results, class_metrics, avg_metrics = final_results
        metrics_reporter.generate_reports(
            class_metrics, 
            avg_metrics, 
            filename_prefix="_Final_Combined_"
        )
    except Exception as e:
        print(f"\nExecução interrompida: {str(e)}")
        print("Você pode executar novamente para retomar do último stage não completado.")