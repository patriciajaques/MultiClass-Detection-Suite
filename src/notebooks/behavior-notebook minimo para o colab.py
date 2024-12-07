# %% [markdown]
# # Detecção de Comportamentos de Aprendizagem
# 
# Este notebook executa a detecção de comportamentos de aprendizagem usando logs de um Sistema Tutor Inteligente.

# %%
# Célula 1 - Configuração do ambiente
import os
from pathlib import Path
import sys

def setup_environment():
    """Configura o ambiente de execução e adiciona src ao PYTHONPATH"""
    try:
        # Verifica se está no Colab
        from google.colab import drive
        drive.mount('/content/drive')
        base_path = Path('/content/drive/MyDrive/Colab Notebooks/behavior-detection')
        print("Executando no Google Colab")
    except ImportError:
        # Execução local
        # Encontra o diretório raiz do projeto
        current_dir = Path.cwd()
        print(f"Current dir: {current_dir}")
        # Se estiver no diretório notebooks, sobe um nível
        if current_dir.name == 'notebooks':
            base_path = current_dir.parent.parent
        else:
            base_path = current_dir
            
        # Verifica se encontrou o diretório correto
        if base_path.name != 'behavior-detection':
            raise RuntimeError("Não foi possível encontrar o diretório raiz do projeto 'behavior-detection'")
        print("Executando localmente")
    
    # Adiciona src ao Python path
    src_path = str(base_path / 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    
    return base_path

# Configura o ambiente
base_path = setup_environment()
print(f"Diretório base: {base_path}")
print(f"Python path: {sys.path}")

# %%
from behavior.behavior_detection import (
    setup_project_path,
    load_and_preprocess_data,
    prepare_data,
    train_models
)

# Configurar caminhos
base_path = Path('/content/drive/MyDrive/Colab Notebooks/behavior-detection')
paths = setup_project_path(base_path)

# Carregar e pré-processar dados
data = load_and_preprocess_data(paths)
print(f"Shape dos dados: {data.shape}")

# Preparar dados para treinamento
X_train, X_test, y_train, y_test = prepare_data(data)
print(f"\nShapes após preparação:")
print(f"X_train: {X_train.shape}")
print(f"X_test: {X_test.shape}")

# Treinar modelos
train_models(X_train, X_test, y_train, y_test, paths)

# %% [markdown]
# 


