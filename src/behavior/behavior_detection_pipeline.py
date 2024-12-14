from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import os

from behavior.data.behavior_data_loader import BehaviorDataLoader
from core.preprocessors.data_cleaner import DataCleaner
from core.preprocessors.data_splitter import DataSplitter
from behavior.data.behavior_data_encoder import BehaviorDataEncoder
from core.preprocessors.data_balancer import DataBalancer
from core.models.multiclass.behavior_model_params import BehaviorModelParams
from core.management.stage_training_manager import StageTrainingManager
from core.reporting import metrics_reporter

class BehaviorDetectionPipeline:
    def __init__(self, n_iter=50, n_jobs=6, test_size=0.2, base_path=None):
        self.n_iter = n_iter
        self.n_jobs = n_jobs
        self.test_size = test_size
        self.setup_paths(base_path)
        
    def setup_paths(self, base_path=None):
        """
        Configure project paths flexibly for any environment.
        
        Args:
            base_path: Optional path to override automatic detection
        """
        # Se um caminho base foi fornecido, use-o
        if base_path:
            base_path = Path(base_path)
        else:
            # Tenta encontrar o diretório base do projeto
            current_file = Path(__file__).resolve()
            
            # Primeiro, tenta encontrar /app (Docker)
            if Path('/app').exists():
                base_path = Path('/app')
                print("Ambiente Docker detectado")
            
            # Depois, tenta encontrar o diretório 'behavior-detection'
            else:
                # Sobe nos diretórios até encontrar 'behavior-detection' ou chegar à raiz
                current_path = current_file.parent
                while current_path.name != 'behavior-detection' and current_path != current_path.parent:
                    current_path = current_path.parent
                
                if current_path.name == 'behavior-detection':
                    base_path = current_path
                    print("Ambiente local detectado")
                else:
                    # Se não encontrar, usa o diretório atual
                    base_path = Path.cwd()
                    print("Usando diretório atual como base")
        
        # Configura os caminhos relativos ao diretório base
        self.paths = {
            'data': base_path / 'data',
            'output': base_path / 'output',
            'models': base_path / 'models',
            'src': base_path / 'src'
        }
        
        # Cria os diretórios se não existirem
        for path in self.paths.values():
            path.mkdir(exist_ok=True)
            
        print(f"\nCaminhos configurados:")
        for key, path in self.paths.items():
            print(f"{key}: {path}")
    
    def load_and_clean_data(self):
        """Load and clean the dataset."""
        # Load data
        data = BehaviorDataLoader.load_data(self.paths['data'] / 'new_logs_labels.csv', delimiter=';')
        print(f"Dataset inicial shape: {data.shape}")
        
        # Remove undefined behaviors
        data = DataCleaner.remove_instances_with_value(data, 'comportamento', '?')
        
        # Remove unnecessary columns
        columns_to_remove = self._get_columns_to_remove()
        cleaned_data = DataCleaner.remove_columns(data, columns_to_remove)
        
        # Handle missing values
        numeric_columns = cleaned_data.select_dtypes(include=['float64', 'int64']).columns
        categorical_columns = cleaned_data.select_dtypes(exclude=['float64', 'int64']).columns
        
        cleaned_data[numeric_columns] = cleaned_data[numeric_columns].fillna(cleaned_data[numeric_columns].median())
        cleaned_data[categorical_columns] = cleaned_data[categorical_columns].fillna('missing')
        
        return cleaned_data
    
    # ... resto da classe permanece igual ...
    
    def _get_columns_to_remove(self):
        """Define columns to be removed from the dataset."""
        columns_to_remove_ids = ['id_log', 'grupo', 'num_dia', 'num_log']
        columns_to_remove_emotions = [
            'estado_afetivo', 'estado_engajamento_concentrado', 
            'estado_confusao', 'estado_frustracao', 'estado_tedio', 'estado_indefinido', 
            'ultimo_estado_afetivo', 'ultimo_engajamento_concentrado', 'ultimo_confusao', 
            'ultimo_frustracao', 'ultimo_tedio', 'ultimo_estado_indefinido'
        ]
        columns_to_remove_personality = [
            'traco_amabilidade_fator', 'traco_extrovercao_fator', 'traco_conscienciosidade_fator', 
            'traco_abertura_fator', 'traco_neuroticismo_fator', 'traco_amabilidade_cat', 
            'traco_extrovercao_cat', 'traco_conscienciosidade_cat', 'traco_abertura_cat', 
            'traco_neuroticismo_cat'
        ]
        columns_to_remove_behaviors = [
            'comportamento_on_task', 'comportamento_on_task_conversation', 'comportamento_on_task_out',
            'comportamento_off_task', 'comportamento_on_system', 'comportamento_indefinido',
            'ultimo_comportamento', 'ultimo_comportamento_on_task', 'ultimo_comportamento_on_task_conversation',
            'ultimo_comportamento_on_task_out', 'ultimo_comportamento_off_task', 'ultimo_comportamento_on_system',
            'ultimo_comportamento_indefinido'
        ]
        return columns_to_remove_ids + columns_to_remove_emotions + columns_to_remove_personality + columns_to_remove_behaviors
    
    def prepare_data(self, data):
        """Prepare data for training including splitting and encoding."""
        # Split by student level
        train_data, test_data = DataSplitter.split_by_student_level(
            data, test_size=self.test_size, column_name='aluno'
        )
        
        # Remove student ID column
        train_data = DataCleaner.remove_columns(train_data, ['aluno'])
        test_data = DataCleaner.remove_columns(test_data, ['aluno'])
        
        # Split features and target
        X_train, y_train = DataSplitter.split_into_x_y(train_data, 'comportamento')
        X_test, y_test = DataSplitter.split_into_x_y(test_data, 'comportamento')
        
        # Encode target variables
        y_train = BehaviorDataEncoder.encode_y(y_train)
        y_test = BehaviorDataEncoder.encode_y(y_test)
        
        # Encode features
        X_encoder = BehaviorDataEncoder(num_classes=5)
        X_encoder.fit(X_train)
        X_train = X_encoder.transform(X_train)
        X_test = X_encoder.transform(X_test)
        
        return X_train, X_test, y_train, y_test
    
    def balance_data(self, X_train, y_train):
        """Apply SMOTE to balance the dataset."""
        data_balancer = DataBalancer()
        return data_balancer.apply_smote(X_train, y_train)
    
    def train_models(self, X_train, X_test, y_train, y_test):
        """Train all models using stage-based training."""
        # Setup cross-validation
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        
        # Initialize model parameters
        model_params = BehaviorModelParams()
        
        # Define training stages
        stages = self._get_training_stages()
        
        # Initialize training manager
        training_manager = StageTrainingManager(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            model_params=model_params,
            n_iter=self.n_iter,
            cv=cv,
            scoring='balanced_accuracy',
            n_jobs=self.n_jobs
        )
        
        # Execute training
        try:
            training_manager.execute_all_stages(training_manager, stages)
            return training_manager
        except Exception as e:
            print(f"\nExecução interrompida: {str(e)}")
            print("Execute novamente para retomar do último stage não completado.")
            return None
    
    def _get_training_stages(self):
        """Define all training stages."""
        models = ['Logistic Regression', 'Decision Tree', 'Random Forest', 
                 'Gradient Boosting', 'SVM', 'KNN', 'XGBoost', 'Naive Bayes', 'MLP']
        selectors = ['none', 'pca', 'rfe', 'rf', 'mi']
        
        stages = []
        stage_num = 1
        
        for model in models:
            for selector in selectors:
                stage_name = f'etapa_{stage_num}_{model.lower().replace(" ", "_")}_{selector}'
                stages.append((stage_name, [model], [selector]))
                stage_num += 1
                
        return stages
    
    def generate_reports(self, training_manager):
        """Generate final reports for all models."""
        if training_manager:
            final_results = training_manager.combine_results()
            training_results, class_metrics, avg_metrics = final_results
            metrics_reporter.generate_reports(
                class_metrics, 
                avg_metrics, 
                filename_prefix="_Final_Combined_"
            )
    
    def run(self):
        """Execute the complete behavior detection pipeline."""
        print("Iniciando pipeline de detecção de comportamentos...")
        
        # Load and clean data
        print("\n1. Carregando e limpando dados...")
        data = self.load_and_clean_data()
        
        # Prepare data
        print("\n2. Preparando dados para treinamento...")
        X_train, X_test, y_train, y_test = self.prepare_data(data)
        
        # Balance data
        print("\n3. Balanceando dados de treino...")
        X_train, y_train = self.balance_data(X_train, y_train)
        
        # Train models
        print("\n4. Iniciando treinamento dos modelos...")
        training_manager = self.train_models(X_train, X_test, y_train, y_test)
        
        # Generate reports
        print("\n5. Gerando relatórios finais...")
        self.generate_reports(training_manager)
        
        print("\nPipeline concluído!")