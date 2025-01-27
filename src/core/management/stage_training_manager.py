# stage_training_manager.py

from datetime import datetime
import json
import os
from core.evaluation.evaluation import Evaluation
from core.management.checkpoint_manager import CheckpointManager
from core.management.results_manager import ResultsManager
from core.training.optuna_bayesian_optimization_training import OptunaBayesianOptimizationTraining
from core.reporting import metrics_reporter
from core.utils.path_manager import PathManager


class StageTrainingManager:
    def __init__(self, X_train, X_test, y_train, y_test, model_params,
                n_iter=50,
                cv=5,
                scoring='balanced_accuracy',
                n_jobs=-1
                ):
        """
        Inicializa o gerenciador de treinamento.
        
        Args:
            X_train: Features de treino
            X_test: Features de teste
            y_train: Labels de treino
            y_test: Labels de teste
            model_params: Parâmetros dos modelos
            checkpoint_path: Caminho para salvar checkpoints
            results_path: Caminho para salvar resultados
            progress_file: Arquivo para controle de progresso
            n_iter: Número de iterações para otimização
            cv: Número de folds para validação cruzada
            scoring: Métrica de avaliação
            n_jobs: Número de jobs paralelos
        """
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model_params = model_params
        self.n_iter = n_iter
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.progress_file = PathManager.get_path('output') / 'progress.json'
        # Inicializar handlers
        self.checkpoint_handler = CheckpointManager()
        self.results_handler = ResultsManager()

    def train_models(self, selected_models, selected_selectors):
        """Executa o treinamento dos modelos selecionados."""

        print(f"Training models: {selected_models}")
        print(f"Model params type: {type(self.model_params)}")
        print(f"Available models: {self.model_params.get_models().keys()}")

        training = OptunaBayesianOptimizationTraining()
        trained_models = training.train_model(
            X_train=self.X_train,
            y_train=self.y_train,
            model_params=self.model_params,
            selected_models=selected_models,
            selected_selectors=selected_selectors,
            n_iter=self.n_iter,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=self.n_jobs
        )
        
        class_metrics, avg_metrics = Evaluation.evaluate_all_models(
            trained_models, self.X_train, self.y_train, 
            self.X_test, self.y_test
        )
        
        return trained_models, class_metrics, avg_metrics
    
    def save_stage_results(self, trained_models, class_metrics, avg_metrics, stage_name):
        """Salva os resultados de uma etapa."""
        checkpoint = {
            'trained_models': trained_models,
            'stage': stage_name,
            'timestamp': datetime.now()
        }
        self.checkpoint_handler.save_checkpoint(checkpoint, f"stage_{stage_name}")
        
        self.results_handler.save_training_results(trained_models, stage_name)
        self.results_handler.save_evaluation_results(class_metrics, avg_metrics, stage_name)
    
    def execute_stage(self, stage_name, models, selectors):
        """Executa uma etapa completa do treinamento."""
        print(f"Executando etapa: {stage_name}")
        print(f"Modelos: {models}")
        print(f"Seletores: {selectors}")
        
        checkpoint = self.checkpoint_handler.load_latest_checkpoint(f"stage_{stage_name}")
        
        if checkpoint is None:
            print(f"Iniciando treinamento da etapa {stage_name}...")
            trained_models, class_metrics, avg_metrics = self.train_models(models, selectors)
            self.save_stage_results(trained_models, class_metrics, avg_metrics, stage_name)
            print(f"Etapa {stage_name} concluída e resultados salvos.")
            return trained_models, class_metrics, avg_metrics
        else:
            print(f"Etapa {stage_name} já foi executada anteriormente.")
            return None
    
    def combine_results(self):
        """Combina os resultados de todas as etapas."""
        class_metrics, avg_metrics = self.results_handler.load_all_results()
        
        if not class_metrics or not avg_metrics:
            print("Aviso: Nenhum resultado para combinar")
            return {}, {}
            
        print(f"Resultados carregados para {len(class_metrics)} modelos")
        return class_metrics, avg_metrics
    
    def _load_progress(self):
        """Carrega o progresso atual do treinamento."""
        try:
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {'completed_stages': [], 'last_stage': None}
            
    def _save_progress(self, completed_stages, current_stage=None):
        """Salva o progresso atual do treinamento."""
        progress = {
            'completed_stages': completed_stages,
            'last_stage': current_stage
        }
        os.makedirs(os.path.dirname(self.progress_file), exist_ok=True)
        with open(self.progress_file, 'w') as f:
            json.dump(progress, f)
    
    def execute_all_stages(self, training_manager, stages):
        """
        Executa todas as combinações de modelos e seletores, mantendo controle do progresso.
        """
        progress = training_manager._load_progress()
        completed_pairs = set(progress['completed_stages'])

        # Exibe o status inicial
        print("\nStatus de execução:")
        for stage_name, models, selectors in stages:
            for model in models:
                for selector in selectors:
                    pair = f"{model}_{selector}"
                    print(
                        f"{pair}: {'já executado' if pair in completed_pairs else 'pendente'}")


        # Executa cada combinação modelo/seletor
        for stage_name, models, selectors in stages:
            for model in models:
                for selector in selectors:
                    pair = f"{model}_{selector}"
                    if pair in completed_pairs:
                        print(f"\n{pair} já foi completado. Pulando...")
                        continue

                    print(f"\n{'='*50}")
                    print(f"Iniciando {pair}")
                    print(f"{'='*50}")

                    try:
                        training_manager._save_progress(
                            list(completed_pairs), pair)
                        results = training_manager.execute_stage(
                            stage_name, [model], [selector])

                        if results:
                            trained_models, class_metrics, avg_metrics = results
                            print(f"\nGerando relatórios para {pair}...")
                            metrics_reporter.generate_reports(
                                class_metrics,
                                avg_metrics,
                                filename_prefix=f"_{pair}_"
                            )
                            completed_pairs.add(pair)
                            training_manager._save_progress(list(completed_pairs))
                            print(f"\n{pair} concluído com sucesso!")

                    except Exception as e:
                        print(f"\nErro em {pair}: {str(e)}")
                        raise

        # Gera relatórios finais após todas as execuções
        self._generate_final_reports(training_manager)


    def _generate_final_reports(self, training_manager):
        """
        Gera os relatórios finais consolidados após a execução de todos os stages.
        """
        print("\nGerando relatório final consolidado...")
        class_metrics, avg_metrics = training_manager.combine_results()

        if not class_metrics or not avg_metrics:
            print("Erro: Não foi possível carregar os resultados para gerar o relatório final")
            return

        try:
            # Define o prefixo do arquivo baseado no range de execução
            filename_prefix = "_Final_"

            # Usa caminho absoluto para garantir a localização correta dos arquivos
            output_dir = os.path.abspath("../output")
            print(f"\nSalvando relatórios finais:")
            print(f"- Diretório: {output_dir}")
            print(f"- Prefixo: {filename_prefix}")

            metrics_reporter.generate_reports(
                class_metrics,
                avg_metrics,
                directory=output_dir,
                filename_prefix=filename_prefix,
                force_overwrite=True
            )
            print("Relatórios finais gerados com sucesso!")

        except Exception as e:
            print(f"Erro ao gerar relatórios finais: {str(e)}")
            raise
