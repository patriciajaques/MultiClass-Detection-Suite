from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score, cohen_kappa_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np

class Evaluation:
    @staticmethod
    def evaluate_all_models(trained_models, X_train, y_train, X_test, y_test):
        """
        Avalia todos os modelos treinados usando conjuntos de treino e teste.
        """
        class_metrics_results = {}
        avg_metrics_results = {}

        if not trained_models:
            print("Aviso: Nenhum modelo foi treinado com sucesso.")
            return {}, {}

        for model_name, model_info in trained_models.items():
            print(f"\nAvaliando modelo: {model_name}")
            try:
                pipeline = model_info['model']
                cv_score = model_info['cv_result']
                
                # Debug information
                print(f"Pipeline steps: {list(pipeline.named_steps.keys())}")
                print(f"X_train shape inicial: {X_train.shape}")
                
                # Se tiver feature_selection, mostrar a transformação
                if 'feature_selection' in pipeline.named_steps:
                    selector = pipeline.named_steps['feature_selection']
                    X_train_transformed = selector.transform(X_train)
                    print(f"Shape após feature selection: {X_train_transformed.shape}")
                
                # Get predictions using the full pipeline
                y_train_pred = pipeline.predict(X_train)
                y_train_prob = pipeline.predict_proba(X_train)
                y_test_pred = pipeline.predict(X_test)
                y_test_prob = pipeline.predict_proba(X_test)

                # Generate evaluation metrics
                train_metrics = Evaluation._generate_metrics(y_train, y_train_pred, y_train_prob)
                test_metrics = Evaluation._generate_metrics(y_test, y_test_pred, y_test_prob)

                # Get feature info
                feature_info = Evaluation._get_feature_info(pipeline, X_train)
                
                # Store results
                class_metrics_results[model_name] = {
                    'train_class_report': train_metrics['class_report'],
                    'train_conf_matrix': train_metrics['conf_matrix'],
                    'test_class_report': test_metrics['class_report'],
                    'test_conf_matrix': test_metrics['conf_matrix'],
                    'feature_info': feature_info
                }

                avg_metrics_results[model_name] = {
                    'cv_report': cv_score,
                    'train_avg_metrics': train_metrics['avg_metrics'],
                    'test_avg_metrics': test_metrics['avg_metrics'],
                    'training_type': model_info['training_type'],
                    'hyperparameters': model_info['hyperparameters']
                }

            except Exception as e:
                print(f"Erro ao avaliar modelo {model_name}: {str(e)}")
                continue

        if not class_metrics_results:
            print("Aviso: Nenhum modelo pôde ser avaliado com sucesso.")
            return {}, {}

        return class_metrics_results, avg_metrics_results

    @staticmethod
    def _get_feature_info(pipeline, X_train):
        """
        Obtém informações sobre as features após a seleção/transformação.
        """
        try:
            if not hasattr(pipeline, 'named_steps') or 'feature_selection' not in pipeline.named_steps:
                return {
                    'type': 'original',
                    'n_features': X_train.shape[1],
                    'description': f'Usando todas as {X_train.shape[1]} features originais'
                }

            selector = pipeline.named_steps['feature_selection']
            X_transformed = selector.transform(X_train)
            n_features_transformed = X_transformed.shape[1]
            
            # Para PCA
            if hasattr(selector, 'components_'):
                feature_info = {
                    'type': 'pca',
                    'n_features': n_features_transformed,
                    'n_components': n_features_transformed,
                    'new_features': [f'PC{i+1}' for i in range(n_features_transformed)],
                    'description': f'Usando {n_features_transformed} componentes principais'
                }
                
                # Adicionar informação de variância explicada se disponível
                if hasattr(selector, 'explained_variance_ratio_'):
                    feature_info['explained_variance_ratio'] = selector.explained_variance_ratio_
                    feature_info['cumulative_variance'] = np.cumsum(selector.explained_variance_ratio_)
                
                return feature_info
            
            # Para seletores baseados em máscara (RF, RFE, MI)
            elif hasattr(selector, 'get_support'):
                mask = selector.get_support()
                # Tratar tanto DataFrames quanto arrays numpy
                if hasattr(X_train, 'columns'):
                    selected = X_train.columns[mask].tolist()
                else:
                    selected = [f'feature_{i}' for i in range(len(mask)) if mask[i]]
                
                return {
                    'type': 'selector',
                    'n_features': len(selected),
                    'selected_features': selected,
                    'description': f'Selecionadas {len(selected)} features originais'
                }
            
            # Para outros transformadores
            else:
                return {
                    'type': 'transform',
                    'n_features': n_features_transformed,
                    'description': f'Transformado para {n_features_transformed} features'
                }
            
        except Exception as e:
            print(f"Erro ao obter informações das features: {str(e)}")
            return {
                'type': 'error',
                'n_features': X_train.shape[1] if hasattr(X_train, 'shape') else 0,
                'description': f'Erro ao obter informações: {str(e)}'
            }

    @staticmethod
    def _generate_metrics(y_true, y_pred, y_prob):
        """
        Gera todas as métricas de avaliação para um conjunto de predições.
        """

        # Generate classification report
        class_report = pd.DataFrame(classification_report(y_true, y_pred, output_dict=True)).transpose()
        
        # Generate confusion matrix
        conf_matrix = pd.DataFrame(
            confusion_matrix(y_true, y_pred),
            index=[f'Actual {i}' for i in range(len(np.unique(y_true)))],
            columns=[f'Predicted {i}' for i in range(len(np.unique(y_true)))]
        )

        # Calculate all average metrics
        metrics_dict = {
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
            'accuracy': class_report.loc['accuracy', 'f1-score'],
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1-score': f1_score(y_true, y_pred, average='weighted'),
            'kappa': cohen_kappa_score(y_true, y_pred)
        }
        
        # Create average metrics DataFrame with consistent structure
        metrics_df = pd.DataFrame(
            {
                'precision': metrics_dict['precision'],
                'recall': metrics_dict['recall'],
                'f1-score': metrics_dict['f1-score'],
                'support': len(y_true)  # Total support for average metrics
            },
            index=['weighted avg']
        )
        
        # Add other metrics maintaining the same column structure
        for metric in ['balanced_accuracy', 'accuracy', 'kappa']:
            metrics_df.loc[metric] = [
                metrics_dict[metric],  # precision column
                metrics_dict[metric],  # recall column
                metrics_dict[metric],  # f1-score column
                len(y_true)           # support column
            ]

        return {
            'class_report': class_report.drop(['accuracy', 'macro avg', 'weighted avg']),
            'conf_matrix': conf_matrix,
            'avg_metrics': metrics_df
        }