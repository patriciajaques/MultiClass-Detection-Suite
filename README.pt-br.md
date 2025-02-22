# MultiClass Detection Suite

*Leia isto em outros idiomas: [English](README.md)*

Um framework abrangente em Python para detecção e classificação multiclasse, com foco em três aplicações principais:

1. **Detecção de Comportamentos de Aprendizagem** (`main_behavior.py`): Classifica diferentes tipos de comportamentos de estudantes durante interações com um Sistema Tutor Inteligente.
2. **Classificação de Emoções** (`main_emotion.py`): Identifica estados emocionais dos estudantes com base em suas ações registradas.
3. **Classificação de Imagens MNIST** (`main_mnist.py`): Reconhecimento de dígitos manuscritos usando o dataset MNIST.

⚠️ **Nota:** Os datasets de comportamento e emoção não estão incluídos neste repositório devido a restrições de privacidade e confidencialidade. O dataset MNIST está disponível para testes.

## Características

### Algoritmos de Classificação Suportados

1. **Regressão Logística**
   - Modelo linear que estima probabilidades de classes usando a função sigmóide
   - Eficiente para problemas lineares e datasets grandes
   - Oferece boa interpretabilidade dos coeficientes

2. **Árvores de Decisão**
   - Modelo baseado em regras de decisão hierárquicas
   - Fácil interpretação visual da lógica de classificação
   - Capaz de capturar relações não-lineares

3. **Random Forest**
   - Ensemble de árvores de decisão
   - Reduz overfitting através de amostragem aleatória
   - Robusto a outliers e ruídos nos dados

4. **Gradient Boosting**
   - Ensemble que combina modelos fracos sequencialmente
   - Otimiza iterativamente os erros do modelo anterior
   - Excelente performance em diversos tipos de dados

5. **SVM (Support Vector Machines)**
   - Encontra hiperplanos ótimos para separação de classes
   - Eficiente em espaços de alta dimensionalidade
   - Robusto em datasets pequenos a médios

6. **KNN (K-Nearest Neighbors)**
   - Classificação baseada em similaridade com exemplos próximos
   - Não paramétrico e intuitivo
   - Bom para dados com fronteiras de decisão complexas

7. **XGBoost**
   - Implementação otimizada de Gradient Boosting
   - Excelente performance e velocidade
   - Suporte nativo para dados faltantes

8. **MLP (Multilayer Perceptron)**
   - Rede neural feed-forward básica
   - Capaz de aprender padrões complexos não-lineares
   - Versátil para diversos tipos de dados

9. **Naive Bayes**
   - Classificador probabilístico baseado no teorema de Bayes
   - Rápido e eficiente em datasets grandes
   - Bom para classificação de texto

10. **LSTM (Long Short-Term Memory)**
    - Rede neural recorrente para sequências temporais
    - Capaz de aprender dependências de longo prazo
    - Ideal para dados sequenciais ou temporais

### Técnicas de Seleção de Features

1. **PCA (Principal Component Analysis)**
   - Reduz dimensionalidade mantendo a variância máxima
   - Transforma features em componentes principais ortogonais
   - Útil para dados com alta dimensionalidade e correlacionados

2. **RFE (Recursive Feature Elimination)**
   - Elimina features recursivamente baseado em importância
   - Usa um modelo base para rankear features
   - Permite controle fino do número de features desejadas

3. **Seleção baseada em Random Forest**
   - Usa importância de features do Random Forest
   - Robusto a outliers e dados não-lineares
   - Considera interações entre features

4. **MI (Mutual Information)**
   - Mede dependência estatística entre features e target
   - Funciona bem com relações não-lineares
   - Não assume distribuição específica dos dados

### Otimização de Hiperparâmetros

- **GridSearchCV**: Busca exaustiva em grade
  - Testa todas as combinações possíveis
  - Garante encontrar o melhor conjunto de parâmetros
  - Computacionalmente intensivo

- **Random Search**: Busca aleatória
  - Amostra aleatória do espaço de parâmetros
  - Mais eficiente que GridSearch para grandes espaços
  - Bom equilíbrio entre exploração e tempo

- **Optuna**: Otimização Bayesiana
  - Usa aprendizado de máquina para guiar a busca
  - Mais eficiente que buscas aleatórias
  - Adaptativo e paralelizável

### Engenharia de Features

- Processamento de features temporais
- Agregação de sequências
- Normalização e codificação automática
- Tratamento de dados faltantes
- Balanceamento de classes

### Avaliação e Relatórios

- Métricas balanceadas
- Matrizes de confusão
- Relatórios de classificação
- Visualização de resultados
- Exportação de métricas

### Features Específicas para Educação

- Detecção de comportamentos de aprendizagem
- Classificação de estados emocionais
- Análise de sequências temporais
- Processamento de logs educacionais

## Dataset de Comportamentos/Emoções

O dataset principal contém:
- **5,525 instâncias** 
- **372 features**
- **10 sessões** de 50 minutos
- **30 estudantes**

Features incluem:
- Interações com o sistema
- Métricas de desempenho
- Operações matemáticas
- Estados comportamentais/afetivos
- Traços de personalidade

## Estrutura do Projeto

```
src/
├── behavior/              # Módulo de detecção de comportamentos
├── emotion/              # Módulo de classificação de emoções
├── mnist/               # Módulo de classificação MNIST
└── core/                # Componentes centrais
    ├── config/         # Gerenciamento de configurações
    ├── evaluation/     # Avaliação de modelos
    ├── feature_selection/  # Seleção de features
    ├── lstm/           # Implementação LSTM
    ├── models/         # Definições de modelos
    ├── preprocessors/  # Processamento de dados
    ├── reporting/      # Geração de relatórios
    └── utils/          # Utilitários
```

## Configuração

### Requisitos

- Python 3.x
- Principais dependências:
  - scikit-learn
  - pandas
  - numpy
  - matplotlib
  - tensorflow
  - optuna
  - PyYAML
  - torch (para LSTM)

### Configuração do Projeto

Cada novo projeto requer dois arquivos YAML na pasta de configuração:

1. **data_cleaning.yaml**: Define as colunas a serem removidas
   ```yaml
   columns_to_remove:
     - "student_id"
     - "timestamp"
     - "session_id"
   ```

2. **training_settings.yaml**: Define modelos e seletores a serem usados
   ```yaml
   training_settings:
     models:
       - "Logistic Regression"
       - "Random Forest"
       - "XGBoost"
     selectors:
       - "none"
       - "pca"
       - "rfe"
   ```

## Como Usar

1. Clone o repositório:
```bash
git clone https://github.com/patriciajaques/multiclass_detection_suite.git
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

3. Configure os arquivos YAML conforme necessário

4. Execute o pipeline desejado:
```bash
# Para comportamentos
python main_behavior.py

# Para emoções
python main_emotion.py

# Para MNIST
python main_mnist.py
```

## Recursos Avançados

### LSTM para Sequências Temporais

- Modelo bidirecional com atenção
- Processamento de sequências temporais
- Balanceamento de classes por sequência

### Ensemble Learning

- Votação soft/hard
- Seleção automática dos melhores modelos
- Combinação de diferentes algoritmos

### Validação Cruzada Estratificada

- Estratificação por classe
- Estratificação por grupo
- Validação temporal

### Persistência de Modelos

- Salvamento automático de modelos
- Gerenciamento de versões
- Carregamento dinâmico

## Contribuindo

1. Fork o projeto
2. Crie uma branch para sua feature
3. Faça commit das mudanças
4. Push para a branch
5. Abra um Pull Request

## Licença

Este projeto está licenciado sob a Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0). Esta licença permite uso não comercial, incluindo pesquisa acadêmica, com atribuição apropriada. Para uso comercial, entre em contato com os autores.

Para mais detalhes, veja [Creative Commons BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).

## Citação

Se você usar este código em sua pesquisa, por favor cite:

```bibtex
@software{jaques2024multiclass,
  author = {Jaques, Patricia A. M.},
  title = {MultiClass Detection Suite},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/patriciajaques/multiclass_detection_suite}
}
```