# stage_training_manager.py

from datetime import datetime
import json
import os
from core.evaluation.evaluation import Evaluation
from core.management.checkpoint_manager import CheckpointManager
from core.management.results_manager import ResultsManager
from core.training.optuna_bayesian_optimization_training import OptunaBayesianOptimizationTraining
from core.reporting import metrics_reporter

class StageTrainingManager:
    def __init__(self, X_train, X_test, y_train, y_test, model_params,
                    checkpoint_path='../output/checkpoints/',
                    results_path='../output/results/',
                    progress_file='../output/progress.json',
                    n_iter=50,
                    cv=5,
                    scoring='balanced_accuracy',
                    n_jobs=-1,
                    stage_range=None):  # Novo parâmetro
            """
            Inicializa o gerenciador de treinamento em etapas.
            
            Args:
                ...
                stage_range (tuple): Tupla (início, fim) indicando o intervalo de stages a executar
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
            self.stage_range = stage_range
            
            # Modificar os caminhos para incluir o intervalo de stages no nome
            if stage_range:
                range_str = f"_{stage_range[0]}_{stage_range[1]}"
                checkpoint_path = checkpoint_path.rstrip('/') + range_str + '/'
                results_path = results_path.rstrip('/') + range_str + '/'
                progress_file = progress_file.replace('.json', f'{range_str}.json')
            
            self.checkpoint_handler = CheckpointManager(checkpoint_path)
            self.results_handler = ResultsManager(results_path)
            self.progress_file = progress_file

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
        Executa um intervalo específico de etapas sequencialmente.
        """

        print("\nOrdem de execução dos stages:")
        for i, (stage_name, models, selectors) in enumerate(stages, 1):
            print(f"{i}. {stage_name}: {models}")

        if self.stage_range:
            start_idx, end_idx = self.stage_range
            stages = stages[start_idx-1:end_idx]
            print(f"\nExecutando stages do {start_idx} ao {end_idx}")
        
        progress = training_manager._load_progress()
        completed_stages = set(progress['completed_stages'])

        print("\nVerificando progresso anterior...")
        if completed_stages:
            print(f"Stages já completados: {', '.join(completed_stages)}")
        else:
            print("Nenhum stage completado anteriormente. Iniciando do começo.")
        
        all_stages_completed = len(completed_stages) == len(stages)
        
        if not all_stages_completed:
            # Executa os stages que faltam
            for stage_num, (stage_name, models, selectors) in enumerate(stages, 1):
                if stage_name in completed_stages:
                    print(f"\nStage {stage_num} ({stage_name}) já foi completado. Pulando...")
                    continue
                    
                print(f"\n{'='*50}")
                print(f"Iniciando Stage {stage_num}: {stage_name}")
                print(f"{'='*50}")
                
                try:
                    training_manager._save_progress(list(completed_stages), stage_name)
                    results = training_manager.execute_stage(stage_name, models, selectors)
                    
                    if results:
                        trained_models, class_metrics, avg_metrics = results
                        print(f"\nGerando relatórios para {stage_name}...")
                        metrics_reporter.generate_reports(
                            class_metrics, 
                            avg_metrics, 
                            filename_prefix=f"_{stage_name}_"
                        )
                        completed_stages.add(stage_name)
                        training_manager._save_progress(list(completed_stages))
                        print(f"\nStage {stage_num} ({stage_name}) concluído com sucesso!")
                    
                except Exception as e:
                    print(f"\nErro no Stage {stage_num} ({stage_name}): {str(e)}")
                    print("O treinamento pode ser retomado deste ponto posteriormente.")
                    raise
        

        # Sempre gera os relatórios finais, mesmo quando todos os stages já foram completados
        print("\nGerando relatório final consolidado...")
        class_metrics, avg_metrics = training_manager.combine_results()
        if class_metrics and avg_metrics:
            if self.stage_range:
                filename_prefix = f"_Final_Combined_{self.stage_range[0]}_{self.stage_range[1]}_"
            else:
                filename_prefix = "_Final_Combined_All_"

            output_dir = os.path.abspath("../output")  # Tornar caminho absoluto
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
        else:
            print("Erro: Não foi possível carregar os resultados para gerar o relatório final")
        
        print("\nProcesso completo! Todos os stages do intervalo foram executados.")