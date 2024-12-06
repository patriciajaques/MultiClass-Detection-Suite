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
                'model_name': model_name,
                'hyperparameters': str(info['hyperparameters']),
                'cv_score': info['cv_result'],
                'training_type': info['training_type']
            })
        
        df_results = pd.DataFrame(results)
        df_results.to_csv(f"{self.base_path}training_results_{stage_name}_{timestamp}.csv", 
                         index=False, sep=';')
        
    def save_evaluation_results(self, class_metrics, avg_metrics, stage_name):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        
        # Salvar métricas por classe
        class_metrics_df = pd.DataFrame()
        for model_name, metrics in class_metrics.items():
            model_metrics = pd.DataFrame({
                'model': model_name,
                'train_metrics': metrics['train_class_report'].to_dict('index'),
                'test_metrics': metrics['test_class_report'].to_dict('index')
            })
            class_metrics_df = pd.concat([class_metrics_df, model_metrics])
            
        class_metrics_df.to_csv(f"{self.base_path}class_metrics_{stage_name}_{timestamp}.csv", 
                               index=False, sep=';')
        
        # Salvar métricas médias
        avg_metrics_df = pd.DataFrame([{
            'model': model_name,
            **metrics
        } for model_name, metrics in avg_metrics.items()])
        
        avg_metrics_df.to_csv(f"{self.base_path}avg_metrics_{stage_name}_{timestamp}.csv", 
                             index=False, sep=';')
    
    def load_all_results(self):
        training_files = [f for f in os.listdir(self.base_path) if f.startswith('training_results_')]
        class_metrics_files = [f for f in os.listdir(self.base_path) if f.startswith('class_metrics_')]
        avg_metrics_files = [f for f in os.listdir(self.base_path) if f.startswith('avg_metrics_')]
        
        all_training_results = pd.concat([
            pd.read_csv(os.path.join(self.base_path, f), sep=';') 
            for f in training_files
        ])
        
        all_class_metrics = pd.concat([
            pd.read_csv(os.path.join(self.base_path, f), sep=';')
            for f in class_metrics_files
        ])
        
        all_avg_metrics = pd.concat([
            pd.read_csv(os.path.join(self.base_path, f), sep=';')
            for f in avg_metrics_files
        ])
        
        return all_training_results, all_class_metrics, all_avg_metrics