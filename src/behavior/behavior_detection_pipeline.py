from pathlib import Path

from behavior.data.behavior_data_loader import BehaviorDataLoader
from core.preprocessors.data_cleaner import DataCleaner
from core.preprocessors.data_imputer import DataImputer
from core.preprocessors.data_splitter import DataSplitter
from behavior.data.behavior_data_encoder import BehaviorDataEncoder
from core.preprocessors.data_balancer import DataBalancer
from core.models.multiclass.behavior_model_params import BehaviorModelParams
from core.management.stage_training_manager import StageTrainingManager

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
            else:
                # Navega para cima na hierarquia até encontrar 'behavior-detection'
                current_path = current_file.parent
                while current_path.name != 'behavior-detection' and current_path != current_path.parent:
                    current_path = current_path.parent

                if current_path.name == 'behavior-detection':
                    base_path = current_path
                    print(
                        f"Diretório behavior-detection encontrado: {base_path}")
                else:
                    # Se não encontrar, usa o diretório pai do arquivo atual
                    base_path = current_file.parent.parent.parent
                    print(f"Usando diretório pai como base: {base_path}")

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

        return paths

    def load_and_clean_data(self):
        """Carrega e limpa o dataset."""
        # Load data
        data = BehaviorDataLoader.load_data(
            self.paths['data'] / 'new_logs_labels.csv', delimiter=';')
        print(f"Dataset inicial shape: {data.shape}")

        # Remove undefined behaviors
        data = DataCleaner.remove_instances_with_value(
            data, 'comportamento', '?')

        # Remove unnecessary columns
        columns_to_remove = self._get_columns_to_remove()
        cleaned_data = DataCleaner.remove_columns(data, columns_to_remove)

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

    def prepare_data(self, data):
        """Prepara os dados para treinamento, incluindo divisão e codificação."""
        print("\nIniciando preparação dos dados...")

        # 1. Encode target (generally, not needed for most models)
        encoder = BehaviorDataEncoder(num_classes=5)
        y = data['comportamento']
        y_encoded = encoder.fit_transform_y(y)
        data['comportamento'] = y_encoded

        # 2. Divide the data into train and test sets stratified by student ID and target
        train_data, test_data = DataSplitter.split_stratified_by_groups(
            data=data,
            test_size=self.test_size,
            group_column='aluno',
            target_column='comportamento'
        )

        # 3. Remove student ID column
        train_data = DataCleaner.remove_columns(train_data, ['aluno'])
        test_data = DataCleaner.remove_columns(test_data, ['aluno'])

        # 4. Split features and target
        X_train, y_train = DataSplitter.split_into_x_y(
            train_data, 'comportamento')
        X_test, y_test = DataSplitter.split_into_x_y(
            test_data, 'comportamento')
        
        # 5. Impute missing values
        print("\nRealizando imputação de valores faltantes...")
        imputer = DataImputer(
            numerical_strategy='knn',
            categorical_strategy='most_frequent',
            knn_neighbors=5
        )

        # Importante: fit apenas no treino, transform em ambos
        print("Ajustando imputador nos dados de treino...")
        X_train_imputed = imputer.fit_transform(X_train)
        print("Aplicando imputação nos dados de teste...")
        X_test_imputed = imputer.transform(X_test)

        # 6. Encode features
        print("\nRealizando encoding das features...")
        X_encoder = BehaviorDataEncoder(num_classes=5)
        X_encoder.fit(X_train)
        X_train_encoded = X_encoder.transform(X_train_imputed)
        X_test_encoded = X_encoder.transform(X_test_imputed)

        return X_train_encoded, X_test_encoded, y_train, y_test

    def balance_data(self, X_train, y_train):
        """Aplica SMOTE para balancear o dataset."""
        data_balancer = DataBalancer()
        return data_balancer.apply_smote(X_train, y_train)

    def _get_training_stages(self):
        """Define todos os stages de treinamento."""
        models = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'XGBoost'
                  'Gradient Boosting', 'KNN', 'Naive Bayes', 'MLP']
        # sempre inserir 'rfe' por ultimo pois é muito lento
        selectors = ['none', 'pca',  'rf'] #rfe, mi

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
        print(
            f"\n5. Executando stages {self.stage_range if self.stage_range else 'todos'}...")
        training_manager.execute_all_stages(training_manager, stages)

        print("\nPipeline concluído!")
