from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

from behavior.data.behavior_data_loader import BehaviorDataLoader
from core.preprocessors.data_cleaner import DataCleaner
from core.preprocessors.data_splitter import DataSplitter
from behavior.data.behavior_data_encoder import BehaviorDataEncoder
from core.preprocessors.data_balancer import DataBalancer
from core.models.multiclass.behavior_model_params import BehaviorModelParams
from core.management.stage_training_manager import StageTrainingManager
from core.reporting import metrics_reporter

class BehaviorDetectionPipeline:
    def __init__(self, n_iter=50, n_jobs=6, test_size=0.2, base_path=None, stage_range=None):
        """
        Inicializa o pipeline de detecção de comportamentos.
        
        Args:
            n_iter (int): Número de iterações para otimização
            n_jobs (int): Número de jobs paralelos
            test_size (float): Tamanho do conjunto de teste
            base_path (str): Caminho base para os arquivos
            stage_range (tuple): Intervalo de stages a executar (inicio, fim)
        """
        self.n_iter = n_iter
        self.n_jobs = n_jobs
        self.test_size = test_size
        self.stage_range = stage_range
        self.base_path = base_path
        self.paths = self._setup_paths(base_path)
        self.model_params = BehaviorModelParams()
        
    def _setup_paths(self, base_path=None):
        """
        Configura os caminhos do projeto flexivelmente para qualquer ambiente.
        
        Args:
            base_path: Caminho base opcional para sobrescrever a detecção automática
                
        Returns:
            dict: Dicionário com os caminhos configurados
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
                # Primeiro encontra o diretório src
                src_dir = current_file.parent.parent
                
                # Se o parent do src não é behavior-detection, procura pelo nome correto
                if src_dir.parent.name != 'behavior-detection':
                    print("Procurando diretório behavior-detection...")
                    
                    # Lista todos os diretórios pais até encontrar 'behavior-detection'
                    current_path = src_dir
                    found = False
                    while current_path != current_path.parent and not found:
                        for sibling in current_path.parent.iterdir():
                            if sibling.name == 'behavior-detection' and sibling.is_dir():
                                base_path = sibling
                                found = True
                                break
                        if not found:
                            current_path = current_path.parent
                    
                    if not found:
                        # Se ainda não encontrou, usa um caminho absoluto
                        base_path = Path('/Users/patricia/Documents/code/python-code/behavior-detection')
                        print(f"Usando caminho absoluto: {base_path}")
                else:
                    base_path = src_dir.parent
                    print("Diretório behavior-detection encontrado diretamente")
        
        print(f"Diretório base selecionado: {base_path}")
        
        # Configura os caminhos relativos ao diretório base
        paths = {
            'data': base_path / 'data',
            'output': base_path / 'output',
            'models': base_path / 'models',
            'src': base_path / 'src'
        }
        
        # Cria os diretórios se não existirem
        for path in paths.values():
            try:
                path.mkdir(exist_ok=True, parents=True)
                print(f"Diretório criado/verificado: {path}")
            except Exception as e:
                print(f"Erro ao criar diretório {path}: {e}")
            
        print(f"\nCaminhos configurados:")
        for key, path in paths.items():
            print(f"{key}: {path}")
            
        return paths
    
    def load_and_clean_data(self):
        """Carrega e limpa o dataset."""
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
    
    def _get_columns_to_remove(self):
        """Define as colunas a serem removidas do dataset."""
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

    # src/mnist/mnist_detection_pipeline.py


    def prepare_data(self, data):
        print("Preparando dados...")

        # Split data
        print("Dividindo dados em treino e teste...")
        train_data, test_data = DataSplitter.split_data_stratified(
            data, test_size=self.test_size, target_column='target'
        )

        # Split features and target
        print("Separando features e target...")
        X_train, y_train = DataSplitter.split_into_x_y(train_data, 'target')
        X_test, y_test = DataSplitter.split_into_x_y(test_data, 'target')

        print(f"X_train shape antes da normalização: {X_train.shape}")
        print(f"Colunas em X_train: {list(X_train.columns)[:5]}...")

        # Scale features usando StandardScaler diretamente
        print("Normalizando features...")
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Converter arrays numpy de volta para DataFrames com os nomes das colunas originais
        X_train_scaled = pd.DataFrame(
            X_train_scaled, columns=X_train.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(
            X_test_scaled, columns=X_test.columns, index=X_test.index)

        print(f"X_train shape após normalização: {X_train_scaled.shape}")
        print(
            f"Colunas em X_train após normalização: {list(X_train_scaled.columns)[:5]}...")

        return X_train_scaled, X_test_scaled, y_train, y_test
    

    def balance_data(self, X_train, y_train):
        print("\nIniciando balanceamento de dados...")
        print(f"Shape de X_train antes do balanceamento: {X_train.shape}")
        print(f"Número de features: {len(X_train.columns)}")
        print(f"Primeiras 5 colunas: {list(X_train.columns)[:5]}")
        print(
            f"Distribuição de classes antes do balanceamento:\n{y_train.value_counts()}")

        data_balancer = DataBalancer()
        X_resampled, y_resampled = data_balancer.apply_smote(X_train, y_train)

        # Garantir que o resultado está em DataFrame/Series
        if not isinstance(X_resampled, pd.DataFrame):
            X_resampled = pd.DataFrame(X_resampled, columns=X_train.columns)
        if not isinstance(y_resampled, pd.Series):
            y_resampled = pd.Series(y_resampled, name='target')

        print(f"Shape de X_train após balanceamento: {X_resampled.shape}")
        print(
            f"Distribuição de classes após balanceamento:\n{y_resampled.value_counts()}")

        return X_resampled, y_resampled

    def _get_training_stages(self):
        """Define todos os stages de treinamento."""
        models = ['Logistic Regression', 'Decision Tree', 'Random Forest', 
                 'Gradient Boosting', 'KNN', 'XGBoost', 'Naive Bayes', 'MLP']
        selectors = ['none', 'pca', 'rfe', 'rf', 'mi']
        
        stages = []
        stage_num = 1
        
        for model in models:
            for selector in selectors:
                stage_name = f'etapa_{stage_num}_{model.lower().replace(" ", "_")}_{selector}'
                stages.append((stage_name, [model], [selector]))
                stage_num += 1
                
        return stages
    
    def run(self):
        """Executa o pipeline completo de detecção de comportamentos."""
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
        training_manager = StageTrainingManager(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            model_params=self.model_params,
            n_iter=self.n_iter,
            cv=10,
            scoring='balanced_accuracy',
            n_jobs=self.n_jobs,
            stage_range=self.stage_range
        )
        
        # Define training stages
        stages = self._get_training_stages()
        
        # Execute training stages
        print(f"\n5. Executando stages {self.stage_range if self.stage_range else 'todos'}...")
        training_manager.execute_all_stages(training_manager, stages)
        
        print("\nPipeline concluído!")