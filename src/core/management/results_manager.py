from datetime import datetime
import pandas as pd
from pathlib import Path
from typing import Dict, Any
from core.utils.path_manager import PathManager


class ResultsManager:
    """
    Gerencia o salvamento e carregamento de resultados de treinamento e avaliação.
    Resultados são salvos em output/results/{module_name}/
    """
    def __init__(self):

        self.results_dir = PathManager.get_path('output') / 'results'
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.training_dir = self.results_dir / 'training'
        self.metrics_dir = self.results_dir / 'metrics'
        self.training_dir.mkdir(exist_ok=True)
        self.metrics_dir.mkdir(exist_ok=True)

    def save_training_results(self, trained_models: Dict[str, Any], stage_name: str) -> str:
        """
        Salva resultados do treinamento em CSV.
        
        Args:
            trained_models: Dicionário com modelos treinados e seus resultados
            stage_name: Nome do estágio de treinamento
            
        Returns:
            str: Caminho do arquivo salvo
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")

        results = []
        for model_name, info in trained_models.items():
            results.append({
                'Model': model_name,
                'CV_Score': info['cv_result'],
                'Training_Type': info['training_type'],
                'Hyperparameters': str(info['hyperparameters'])
            })

        df_results = pd.DataFrame(results)
        output_file = self.training_dir / \
            f"training_results_{stage_name}_{timestamp}.csv"
        df_results.to_csv(output_file, index=False, sep=';')

        return str(output_file)

    def save_evaluation_results(self, class_metrics: Dict, avg_metrics: Dict, stage_name: str) -> tuple:
        """
        Salva métricas de avaliação em CSV.
        
        Args:
            class_metrics: Métricas por classe
            avg_metrics: Métricas médias
            stage_name: Nome do estágio de avaliação
            
        Returns:
            tuple: Caminhos dos arquivos salvos (class_metrics_path, avg_metrics_path)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")

        # Salvar métricas por classe
        class_metrics_rows = []
        for model_name, metrics in class_metrics.items():
            class_metrics_rows.append({
                'Model': model_name,
                'Train_Metrics': metrics['train_class_report'].to_dict('index'),
                'Test_Metrics': metrics['test_class_report'].to_dict('index')
            })

        class_metrics_df = pd.DataFrame(class_metrics_rows)
        class_metrics_file = self.metrics_dir / \
            f"class_metrics_{stage_name}_{timestamp}.csv"
        class_metrics_df.to_csv(class_metrics_file, index=False, sep=';')

        # Salvar métricas médias
        avg_metrics_rows = []
        for model_name, metrics in avg_metrics.items():
            avg_metrics_rows.append({
                'Model': model_name,
                'CV_Score': metrics.get('cv_result', metrics.get('cv_report', 0.0)),
                'Training_Type': metrics.get('training_type', 'unknown'),
                'Train_Avg_Metrics': metrics.get('train_avg_metrics', pd.DataFrame()).to_dict('index'),
                'Test_Avg_Metrics': metrics.get('test_avg_metrics', pd.DataFrame()).to_dict('index')
            })

        avg_metrics_df = pd.DataFrame(avg_metrics_rows)
        avg_metrics_file = self.metrics_dir / \
            f"avg_metrics_{stage_name}_{timestamp}.csv"
        avg_metrics_df.to_csv(avg_metrics_file, index=False, sep=';')

        return str(class_metrics_file), str(avg_metrics_file)

    def load_all_results(self) -> tuple:
        """
        Carrega todos os resultados salvos.
        
        Returns:
            tuple: (class_metrics_dict, avg_metrics_dict)
        """
        try:
            # Identificar arquivos
            class_metrics_files = list(
                self.metrics_dir.glob("class_metrics_*.csv"))
            avg_metrics_files = list(
                self.metrics_dir.glob("avg_metrics_*.csv"))

            if not class_metrics_files or not avg_metrics_files:
                print("Aviso: Nenhum arquivo de métricas encontrado.")
                return {}, {}

            # Processar métricas por classe
            class_metrics_dict = self._process_class_metrics_files(
                class_metrics_files)

            # Processar métricas médias
            avg_metrics_dict = self._process_avg_metrics_files(
                avg_metrics_files)

            return class_metrics_dict, avg_metrics_dict

        except Exception as e:
            print(f"Erro ao carregar resultados: {str(e)}")
            return {}, {}

    def _process_class_metrics_files(self, files) -> Dict:
        class_metrics_dict = {}
        for file in files:
            try:
                df = pd.read_csv(file, sep=';')
                # Normalize column names
                df.columns = df.columns.str.lower()

                for _, row in df.iterrows():
                    model_name = row['model']
                    train_metrics = self._safe_eval(row['train_metrics'])
                    test_metrics = self._safe_eval(row['test_metrics'])

                    class_metrics_dict[model_name] = {
                        'train_class_report': pd.DataFrame(train_metrics),
                        'test_class_report': pd.DataFrame(test_metrics)
                    }
            except Exception as e:
                print(f"Erro ao processar arquivo {file}: {str(e)}")

        return class_metrics_dict

    def _process_avg_metrics_files(self, files) -> Dict:
        avg_metrics_dict = {}
        for file in files:
            try:
                df = pd.read_csv(file, sep=';')
                df.columns = df.columns.str.lower()

                for _, row in df.iterrows():
                    model_name = row['model']
                    avg_metrics_dict[model_name] = {
                        'cv_result': float(row.get('cv_score', 0.0)),
                        'train_avg_metrics': pd.DataFrame(self._safe_eval(row['train_avg_metrics'])),
                        'test_avg_metrics': pd.DataFrame(self._safe_eval(row['test_avg_metrics'])),
                        'training_type': row.get('training_type', 'unknown')
                    }
            except Exception as e:
                print(f"Erro ao processar arquivo {file}: {str(e)}")

        return avg_metrics_dict

    def _safe_eval(self, value: str) -> Dict:
        """Avalia strings de forma segura."""
        if not isinstance(value, str):
            return {}
        try:
            value = value.strip()
            if value.startswith('{') and value.endswith('}'):
                return eval(value)
            return {}
        except:
            return {}
