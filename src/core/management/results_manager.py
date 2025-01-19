import pandas as pd
from datetime import datetime
import os


class ResultsManager:
    def __init__(self, base_path='drive/MyDrive/behavior_detection/results/'):
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)

    def save_training_results(self, trained_models, stage_name):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")

        # Salvar hiperparâmetros e scores CV
        results = []
        for model_name, info in trained_models.items():
            results.append({
                'Model': model_name,  # Mudança aqui - nome da coluna para maiúsculo
                'CV_Score': info['cv_result'],
                'Training_Type': info['training_type'],
                'Hyperparameters': str(info['hyperparameters'])
            })

        df_results = pd.DataFrame(results)
        df_results.to_csv(f"{self.base_path}training_results_{stage_name}_{timestamp}.csv",
                          index=False, sep=';')

    def save_evaluation_results(self, class_metrics, avg_metrics, stage_name):
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
        class_metrics_df.to_csv(f"{self.base_path}class_metrics_{stage_name}_{timestamp}.csv",
                                index=False, sep=';')

        # Salvar métricas médias
        avg_metrics_rows = []
        for model_name, metrics in avg_metrics.items():
            avg_metrics_rows.append({
                'Model': model_name,
                'CV_Score': metrics.get('cv_report', 0.0),
                'Training_Type': metrics.get('training_type', 'unknown'),
                'Train_Avg_Metrics': metrics.get('train_avg_metrics').to_dict('index') if isinstance(metrics.get('train_avg_metrics'), pd.DataFrame) else {},
                'Test_Avg_Metrics': metrics.get('test_avg_metrics').to_dict('index') if isinstance(metrics.get('test_avg_metrics'), pd.DataFrame) else {}
            })

        avg_metrics_df = pd.DataFrame(avg_metrics_rows)
        avg_metrics_df.to_csv(f"{self.base_path}avg_metrics_{stage_name}_{timestamp}.csv",
                              index=False, sep=';')

    def load_all_results(self):
        """
        Carrega e combina todos os resultados salvos.
        """
        try:
            # Identificar arquivos
            class_metrics_files = [f for f in os.listdir(
                self.base_path) if f.startswith('class_metrics_')]
            avg_metrics_files = [f for f in os.listdir(
                self.base_path) if f.startswith('avg_metrics_')]

            if not class_metrics_files or not avg_metrics_files:
                print("Aviso: Nenhum arquivo de métricas encontrado.")
                return {}, {}

            # Carregar e processar métricas por classe
            class_metrics_dict = {}
            for f in class_metrics_files:
                try:
                    df = pd.read_csv(os.path.join(self.base_path, f), sep=';')
                    column_mapping = {
                        'model': 'Model',
                        'train_metrics': 'Train_Metrics',
                        'test_metrics': 'Test_Metrics'
                    }
                    # Converter todas para minúsculo primeiro
                    df = df.rename(columns=str.lower)
                    df = df.rename(columns=column_mapping)

                    for idx, row in df.iterrows():
                        model_name = row['Model']
                        try:
                            # Criar DataFrames com índices explícitos
                            train_metrics = eval(row['Train_Metrics']) if isinstance(
                                row['Train_Metrics'], str) else {}
                            test_metrics = eval(row['Test_Metrics']) if isinstance(
                                row['Test_Metrics'], str) else {}

                            if isinstance(train_metrics, dict):
                                train_df = pd.DataFrame.from_dict(
                                    train_metrics, orient='index')
                            else:
                                train_df = pd.DataFrame()

                            if isinstance(test_metrics, dict):
                                test_df = pd.DataFrame.from_dict(
                                    test_metrics, orient='index')
                            else:
                                test_df = pd.DataFrame()

                            class_metrics_dict[model_name] = {
                                'train_class_report': train_df,
                                'test_class_report': test_df
                            }
                        except Exception as e:
                            print(
                                f"Erro ao processar métricas para modelo {model_name}: {str(e)}")
                            continue

                except Exception as e:
                    print(f"Erro ao processar arquivo {f}: {str(e)}")
                    continue

            # Carregar e processar métricas médias
            avg_metrics_dict = {}
            for f in avg_metrics_files:
                try:
                    df = pd.read_csv(os.path.join(self.base_path, f), sep=';')
                    column_mapping = {
                        'model': 'Model',
                        'cv_report': 'CV_Score',
                        'train_avg_metrics': 'Train_Avg_Metrics',
                        'test_avg_metrics': 'Test_Avg_Metrics',
                        'training_type': 'Training_Type',
                        'hyperparameters': 'Hyperparameters'
                    }
                    # Converter todas para minúsculo primeiro
                    df = df.rename(columns=str.lower)
                    df = df.rename(columns=column_mapping)

                    for idx, row in df.iterrows():
                        model_name = row['Model']
                        try:
                            # Processar métricas médias com tratamento de erro melhorado
                            train_metrics = self._safe_eval(
                                row['Train_Avg_Metrics'])
                            test_metrics = self._safe_eval(row['Test_Avg_Metrics'])

                            if isinstance(train_metrics, dict):
                                train_df = pd.DataFrame.from_dict(
                                    train_metrics, orient='index')
                            else:
                                train_df = pd.DataFrame()

                            if isinstance(test_metrics, dict):
                                test_df = pd.DataFrame.from_dict(
                                    test_metrics, orient='index')
                            else:
                                test_df = pd.DataFrame()

                            avg_metrics_dict[model_name] = {
                                'cv_report': float(row.get('CV_Score', 0.0)),
                                'train_avg_metrics': train_df,
                                'test_avg_metrics': test_df,
                                'training_type': row.get('Training_Type', 'unknown')
                            }
                        except Exception as e:
                            print(
                                f"Erro ao processar métricas médias para modelo {model_name}: {str(e)}")
                            continue

                except Exception as e:
                    print(f"Erro ao processar arquivo {f}: {str(e)}")
                    continue

            num_class_results = len(class_metrics_dict)
            num_avg_results = len(avg_metrics_dict)

            if num_class_results == 0 or num_avg_results == 0:
                print("Aviso: Nenhum resultado foi carregado com sucesso")
                return {}, {}

            print(
                f"Carregados com sucesso: {num_class_results} resultados de classe e {num_avg_results} resultados médios")
            return class_metrics_dict, avg_metrics_dict

        except Exception as e:
            print(f"Erro ao carregar resultados: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return {}, {}


    def _safe_eval(self, value):
        """Avalia strings de forma segura, com melhor tratamento de erros"""
        if not isinstance(value, str):
            return {}
        try:
            # Limpar a string antes de avaliar
            value = value.strip()
            if value.startswith('{') and value.endswith('}'):
                return eval(value)
            return {}
        except:
            return {}

    def _process_metrics_to_dataframe(self, metrics_series):
        """Converte série de métricas em DataFrame estruturado"""
        try:
            if isinstance(metrics_series, str):
                metrics_dict = eval(metrics_series)
            else:
                metrics_dict = metrics_series.to_dict()

            return pd.DataFrame(metrics_dict)
        except:
            return pd.DataFrame()

    def _create_avg_metrics_df(self, row, prefix):
        """Cria DataFrame de métricas médias a partir da linha do CSV"""
        metrics = {
            'precision': float(row.get(f'precision-{prefix}', 0.0)),
            'recall': float(row.get(f'recall-{prefix}', 0.0)),
            'f1-score': float(row.get(f'f1-score-{prefix}', 0.0)),
            'support': float(row.get(f'support-{prefix}', 0.0))
        }
        return pd.DataFrame([metrics], index=['weighted avg'])
