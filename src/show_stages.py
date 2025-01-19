# src/show_stages.py

def list_stages():
    """Lista todos os estágios disponíveis no pipeline."""
    models = ['Logistic Regression', 'Decision Tree', 'Random Forest',
              'Gradient Boosting', 'KNN', 'XGBoost', 'Naive Bayes', 'MLP']
    selectors = ['none', 'pca', 'rfe', 'rf', 'mi']

    print("\nEstágios disponíveis:")
    print("-" * 50)
    stage_num = 1

    for model in models:
        for selector in selectors:
            stage_name = f'etapa_{stage_num}_{model.lower().replace(" ", "_")}_{selector}'
            print(
                f"Estágio {stage_num:2d}: {model:20s} com seletor {selector:4s}")
            stage_num += 1

    print("\nExemplos de uso:")
    print("python src/main_mnist.py --start-stage 1 --end-stage 5  # Primeiros 5 estágios")
    print("python src/main_mnist.py --start-stage 6 --end-stage 10 # Estágios 6 a 10")


if __name__ == "__main__":
    list_stages()
