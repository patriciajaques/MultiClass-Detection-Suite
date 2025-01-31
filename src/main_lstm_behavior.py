import argparse
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from core.preprocessors.data_cleaner import DataCleaner
from core.config.config_manager import ConfigManager
from core.preprocessors.data_encoder import DataEncoder
from core.preprocessors.data_loader import DataLoader
from lstm.lstm_pipeline import LSTMPipeline


def load_and_clean_data(data_path):
    print("Loading and cleaning data...")
    data = DataLoader.load_data(data_path, delimiter=';')

    config_manager = ConfigManager('src/behavior/config')
    data_cleaner = DataCleaner(config_manager)

    # Remove undefined behaviors
    data = data_cleaner.remove_instances_with_value(data, 'comportamento', '?')

    # Remove unnecessary columns
    cleaned_data = data_cleaner.remove_columns(data, use_config=True)

    # Merge similar behaviors
    cleaned_data['comportamento'] = cleaned_data['comportamento'].replace(
        ['ON TASK OUT', 'ON TASK CONVERSATION'], 'ON TASK OUT'
    )

    print(f"Final data shape: {cleaned_data.shape}")
    print(f"Behavior classes: {cleaned_data['comportamento'].unique()}")
    return cleaned_data


def prepare_data(data, test_size=0.2):
    unique_students = data['aluno'].unique()
    train_students, test_students = train_test_split(
        unique_students, test_size=test_size, random_state=42
    )

    train_data = data[data['aluno'].isin(train_students)]
    test_data = data[data['aluno'].isin(test_students)]

    # Preserve ordering columns before encoding
    ordering_cols = ['aluno', 'num_dia', 'num_log', 'comportamento']
    ordering_data_train = train_data[ordering_cols].copy()
    ordering_data_test = test_data[ordering_cols].copy()

    # Encode features
    encoder = DataEncoder(num_classes=4)
    X_train = encoder.fit_transform(train_data.drop(columns=ordering_cols))
    X_test = encoder.transform(test_data.drop(columns=ordering_cols))

    # Reconstruct dataframes preserving ordering columns
    train_data = pd.concat([ordering_data_train, pd.DataFrame(
        X_train, index=train_data.index)], axis=1)
    test_data = pd.concat([ordering_data_test, pd.DataFrame(
        X_test, index=test_data.index)], axis=1)

    return train_data, test_data

def main():
    parser = argparse.ArgumentParser(
        description='Train LSTM model for behavior detection')
    parser.add_argument('--data-path', type=str, default='data/new_logs_labels.csv',
                        help='Path to data file')
    parser.add_argument('--sequence-length', type=int, default=10,
                        help='Length of input sequences')
    parser.add_argument('--hidden-size', type=int, default=64,
                        help='LSTM hidden layer size')
    parser.add_argument('--num-layers', type=int, default=2,
                        help='Number of LSTM layers')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Learning rate')

    args = parser.parse_args()

    # Load and prepare data
    data = load_and_clean_data(args.data_path)
    train_data, test_data = prepare_data(data)

    # Configure and train LSTM
    lstm_pipeline = LSTMPipeline(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate
    )

    print("\nTraining LSTM model...")
    lstm_pipeline.fit(train_data, train_data['comportamento'])

    # Evaluate
    # Avaliação
    print("\nEvaluating model...")
    predictions = lstm_pipeline.predict(test_data)
    true_labels = test_data['comportamento'].values

    print("\nClassification Report:")
    print(classification_report(true_labels, predictions))
    print(
        f"\nBalanced Accuracy: {balanced_accuracy_score(true_labels, predictions):.4f}")


if __name__ == "__main__":
    main()
