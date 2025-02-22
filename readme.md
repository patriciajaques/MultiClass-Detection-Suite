# MultiClass Detection Suite

*Read this in other languages: [Portuguese](README.pt-br.md)*

A comprehensive Python framework for multiclass detection and classification, focusing on three main applications:

1. **Learning Behavior Detection** (`main_behavior.py`): Classifies different types of student behaviors during interactions with an Intelligent Tutoring System.
2. **Emotion Classification** (`main_emotion.py`): Identifies students' emotional states based on their logged actions.
3. **MNIST Image Classification** (`main_mnist.py`): Handwritten digit recognition using the MNIST dataset.

⚠️ **Note:** The behavior and emotion datasets are not included in this repository due to privacy and confidentiality restrictions. The MNIST dataset is available for testing.

## Features

### Supported Classification Algorithms

1. **Logistic Regression**
   - Linear model that estimates class probabilities using the sigmoid function
   - Efficient for linear problems and large datasets
   - Offers good coefficient interpretability

2. **Decision Trees**
   - Model based on hierarchical decision rules
   - Easy visual interpretation of classification logic
   - Capable of capturing non-linear relationships

3. **Random Forest**
   - Ensemble of decision trees
   - Reduces overfitting through random sampling
   - Robust to outliers and data noise

4. **Gradient Boosting**
   - Ensemble that sequentially combines weak models
   - Iteratively optimizes errors from previous models
   - Excellent performance on various data types

5. **SVM (Support Vector Machines)**
   - Finds optimal hyperplanes for class separation
   - Efficient in high-dimensional spaces
   - Robust on small to medium datasets

6. **KNN (K-Nearest Neighbors)**
   - Classification based on similarity to nearby examples
   - Non-parametric and intuitive
   - Good for data with complex decision boundaries

7. **XGBoost**
   - Optimized implementation of Gradient Boosting
   - Excellent performance and speed
   - Native support for missing data

8. **MLP (Multilayer Perceptron)**
   - Basic feed-forward neural network
   - Capable of learning complex non-linear patterns
   - Versatile for various data types

9. **Naive Bayes**
   - Probabilistic classifier based on Bayes' theorem
   - Fast and efficient on large datasets
   - Good for text classification

### Feature Selection Techniques

1. **PCA (Principal Component Analysis)**
   - Reduces dimensionality while maintaining maximum variance
   - Transforms features into orthogonal principal components
   - Useful for high-dimensional and correlated data

2. **RFE (Recursive Feature Elimination)**
   - Eliminates features recursively based on importance
   - Uses a base model to rank features
   - Allows fine control of desired number of features

3. **Random Forest-based Selection**
   - Uses Random Forest feature importance
   - Robust to outliers and non-linear data
   - Considers feature interactions

4. **MI (Mutual Information)**
   - Measures statistical dependence between features and target
   - Works well with non-linear relationships
   - Doesn't assume specific data distribution

### Hyperparameter Optimization

- **GridSearchCV**: Exhaustive grid search
  - Tests all possible combinations
  - Guarantees finding the best parameter set
  - Computationally intensive

- **Random Search**: Random search
  - Random sampling of parameter space
  - More efficient than GridSearch for large spaces
  - Good balance between exploration and time

- **Optuna**: Bayesian optimization
  - Uses machine learning to guide the search
  - More efficient than random searches
  - Adaptive and parallelizable

### Feature Engineering

- Automatic normalization and encoding
- Missing data handling
- Class balancing

### Evaluation and Reporting

- Balanced metrics
- Confusion matrices
- Classification reports
- Result visualization
- Metrics export

## Advanced Features

### Ensemble Learning

- Soft/hard voting
- Automatic model selection
- Different algorithm combinations
- Best model selection strategies

### Cross-Validation

- Class stratification
- Group-based validation
- Performance evaluation metrics

### Model Persistence

- Automatic model saving
- Version management
- Dynamic loading

### Education-Specific Features

- Learning behavior pattern detection
- Emotional state classification
- Educational log processing
- Student interaction analysis

## Behavior/Emotion Dataset

The main dataset contains:
- **5,525 instances**
- **372 features**
- **10 sessions** of 50 minutes
- **30 students**

Features include:
- System interactions
- Performance metrics
- Mathematical operations
- Behavioral/affective states
- Personality traits

## Project Structure

```
src/
├── behavior/              # Behavior detection module
├── emotion/              # Emotion classification module
├── mnist/               # MNIST classification module
└── core/                # Core components
    ├── config/         # Configuration management
    ├── evaluation/     # Model evaluation
    ├── feature_selection/  # Feature selection
    ├── models/         # Model definitions
    ├── preprocessors/  # Data processing
    ├── reporting/      # Report generation
    └── utils/          # Utilities
```

## Setup

### Requirements

- Python 3.x
- Main dependencies:
  - scikit-learn
  - pandas
  - numpy
  - matplotlib
  - optuna
  - PyYAML

### Project Configuration

Each new project requires two YAML files in the configuration folder:

1. **data_cleaning.yaml**: Defines columns to be removed
   ```yaml
   columns_to_remove:
     - "student_id"
     - "timestamp"
     - "session_id"
   ```

2. **training_settings.yaml**: Defines models and selectors to be used
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

## How to Use

1. Clone the repository:
```bash
git clone https://github.com/patriciajaques/multiclass_detection_suite.git
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure YAML files as needed

4. Run the desired pipeline:
```bash
# For behaviors
python main_behavior.py

# For emotions
python main_emotion.py

# For MNIST
python main_mnist.py
```

## Contributing

1. Fork the project
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## License

This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0). This license allows non-commercial use, including academic research, with appropriate attribution. For commercial use, please contact the authors.

For more details, see [Creative Commons BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).

## Permitted Use

This software is freely available for:
- Academic research
- Teaching and education
- Personal non-commercial use

For any commercial or production use, please contact the authors to obtain a commercial license.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{jaques2024multiclass,
  author = {Jaques, Patricia A. M.},
  title = {MultiClass Detection Suite},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/patriciajaques/multiclass_detection_suite}
}
```