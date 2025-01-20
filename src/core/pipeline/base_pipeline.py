# src/core/pipeline/base_pipeline.py
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd
from core.config.config_manager import ConfigManager
from core.management.stage_training_manager import StageTrainingManager
from core.preprocessors.data_balancer import DataBalancer


class BasePipeline(ABC):
    """Classe base abstrata para pipelines de detecção."""

    def __init__(self, n_iter=50, n_jobs=6, test_size=0.2, base_path=None,
                 stage_range=None, config_dir=None):
        """
        Inicializa o pipeline base.
        
        Args:
            n_iter (int): Número de iterações para otimização
            n_jobs (int): Número de jobs paralelos
            test_size (float): Tamanho do conjunto de teste
            base_path (str): Caminho base para os arquivos
            stage_range (tuple): Intervalo de stages a executar (inicio, fim)
            config_dir (str): Diretório de configurações
        """
        self.n_iter = n_iter
        self.n_jobs = n_jobs
        self.test_size = test_size
        self.stage_range = stage_range
        self.base_path = base_path
        self.paths = self._setup_paths(base_path)
        self.config = ConfigManager(config_dir) if config_dir else None
        self.model_params = self._get_model_params()


    def _setup_paths(self, base_path=None):
        """Configura os caminhos do projeto."""
        if base_path:
            base_path = Path(base_path)
        else:
            current_file = Path(__file__).resolve()
            # Navega até encontrar o diretório raiz do projeto
            while current_file.parent.name != 'behavior-detection' and current_file.parent != current_file.parent.parent:
                current_file = current_file.parent
            base_path = current_file.parent

        paths = {
            'data': base_path / 'data',
            'output': base_path / 'output',
            'models': base_path / 'models',
            'src': base_path / 'src'
        }

        for path in paths.values():
            path.mkdir(exist_ok=True, parents=True)

        return paths

    def _get_training_stages(self):
        """Define os stages de treinamento usando configuração."""
        try:
            training_config = self.config.get_config('training_settings')

            models = training_config.get('models', ['Naive Bayes'])
            selectors = training_config.get('selectors', ['none'])

            stages = []
            stage_num = 1

            for model in models:
                for selector in selectors:
                    stage_name = f'etapa_{stage_num}_{model.lower().replace(" ", "_")}_{selector}'
                    stages.append((stage_name, [model], [selector]))
                    stage_num += 1

            return stages

        except Exception as e:
            print(f"Erro ao carregar configurações de treinamento: {str(e)}")
            return [('etapa_1_naive_bayes_none', ['Naive Bayes'], ['none'])]

    @abstractmethod
    def _get_model_params(self):
        """Retorna os parâmetros do modelo específico do pipeline."""
        pass

    @abstractmethod
    def load_and_clean_data(self):
        """Carrega e prepara os dados específicos do pipeline."""
        pass

    @abstractmethod
    def prepare_data(self, data):
        """Prepara os dados para treinamento."""
        pass

    def _verify_split_quality(self, train_data, test_data):
        """
        Verifica se o split manteve as proporções desejadas
        """
        # Verifica se todos os alunos estão em apenas um conjunto
        train_students = set(train_data['aluno'])
        test_students = set(test_data['aluno'])
        overlap = train_students & test_students
        assert len(overlap) == 0, f"Alunos presentes em ambos conjuntos: {overlap}"

        # Verifica proporções das classes
        train_dist = train_data['comportamento'].value_counts(normalize=True)
        test_dist = test_data['comportamento'].value_counts(normalize=True)

        for behavior in train_dist.index:
            diff = abs(train_dist[behavior] - test_dist[behavior])
            assert diff < 0.1, f"Diferença grande na distribuição do comportamento {behavior}"

    def balance_data(self, X_train, y_train, strategy='auto'):
        print("\nIniciando balanceamento de dados...")
        print(f"Tipo de X_train: {type(X_train)}")
        print(f"Tipo de y_train: {type(y_train)}")
        print(f"Shape de X_train antes do balanceamento: {X_train.shape}")
        print(f"Distribuição de classes antes do balanceamento:")
        print(y_train.value_counts())

        # Garantir que os dados estão no formato correto
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
        if isinstance(y_train, pd.Series):
            y_train = y_train.values

        data_balancer = DataBalancer()
        X_resampled, y_resampled = data_balancer.apply_smote(X_train, y_train)

        print(f"Shape de X_train após balanceamento: {X_resampled.shape}")
        print(f"Distribuição de classes após balanceamento:")
        print(pd.Series(y_resampled).value_counts())

        return X_resampled, y_resampled

    def run(self):
        """Executa o pipeline completo."""
        print(f"Iniciando pipeline de {self.__class__.__name__}...")

        # Load and prepare data
        print("\n1. Carregando e preparando dados...")
        data = self.load_and_clean_data()

        # Prepare data
        print("\n2. Preparando dados para treinamento...")
        X_train, X_test, y_train, y_test = self.prepare_data(data)

        # Balance data
        print("\n3. Balanceando dados de treino...")
        X_train, y_train = self.balance_data(X_train, y_train, strategy='auto')

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
        print(
            f"\n5. Executando stages {self.stage_range if self.stage_range else 'todos'}...")
        training_manager.execute_all_stages(training_manager, stages)

        print("\nPipeline concluído!")
