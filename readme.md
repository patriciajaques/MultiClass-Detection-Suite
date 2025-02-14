# Multiclass Detection Suite

This repository contains a generic multiclass classifier designed to solve classification problems with more than two classes. It is used in three distinct applications, all based on logs of actions from an Intelligent Tutoring System:

1. **Learning Behavior Detection** (`main_behavior.py`): Classifies different types of student behavior during interaction.
2. **Emotion Classification** (`main_emotion.py`): Identifies emotions based on logged actions.
3. **Image Classification using the MNIST dataset** (`main_mnist.py`): Handwritten digit recognition.

The code structure is modular and reusable, with most classes and functions shared between the three problems.

⚠️ **Note:** The datasets used for behavior and emotion classification are **not provided** in this repository due to privacy and confidentiality restrictions. Only the MNIST dataset is available for testing.

## Features

- **Supported Models:** 
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Gradient Boosting
  - Support Vector Machines (SVM)
  - K-Nearest Neighbors (KNN)
  - XGBoost
  - Multilayer Perceptron (MLP)
  - Naive Bayes
  
- **Feature Selection Techniques:** 
  - None (default)
  - PCA (Principal Component Analysis)
  - RFE (Recursive Feature Elimination)
  - RF (Random Forest-based selection)
  - MI (Mutual Information)

- **Hyperparameter Optimization:** 
  - **GridSearchCV**
  - **Random Search**
  - **Optuna** (for more efficient Bayesian-based search).

- **Cross-Validation and Stratification:** 
  - Configurable cross-validation support.
  - Stratification by **class** and **group** (useful to maintain balanced distribution across subgroups).

## Project Organization and Configuration

Each new project must be configured by creating a new folder inside the main directory. For example, for the **mnist** project, create a folder named `mnist` containing two YAML configuration files:

1. **File for independent variables to be removed** (`columns_to_remove.yaml`): lists the features that should be excluded during training.

   Example:
   ```yaml
   columns_to_remove:
     - "student_id"
     - "timestamp"
     - "session_id"
   ```

2. **Configuration file for models and selectors** (`config.yaml`): defines which algorithms and selectors will be executed for the project.

   Example:
   ```yaml
   training_settings:
     models:
       - "Logistic Regression"
       - "Random Forest"
       - "XGBoost"
       - "MLP"
     selectors:
       - "none"
       - "pca"
       - "rfe"
   ```

⚠️ **Note:** You can comment or uncomment algorithms and selectors in the YAML file to customize which ones will be executed. The code will only consider the active lines (without `#`).

## Cross-Validation and Stratification

The code uses configurable cross-validation, allowing stratification by class and groups as needed. This ensures that each dataset split maintains the original proportion of classes and specific groups, avoiding biases during training.

The main validation strategies include:

- **Stratified Cross-Validation (Stratified K-Fold):** ensures each fold maintains the original class proportion.
- **Group Stratification:** useful when data has a natural grouping structure, such as different sessions for the same student.

## Dataset

The main dataset contains **5,525 instances** and **372 features**, collected during **10 sessions of 50 minutes with 30 students**. The features include:

- **System interactions** (clicks, checks, idle time)
- **Performance and progress** (number of correct steps, time spent, effectiveness)
- **Mathematical operations performed** (addition, subtraction, multiplication, simplification)
- **Behavior and affective state** (engagement, concentration, confusion, frustration, boredom)
- **Personality traits** (agreeableness, extroversion, neuroticism)

⚠️ **Note:** The behavior and emotion datasets are not available in this repository.

## How to Run

1. Clone the repository:
    ```bash
    git clone https://github.com/patriciajaques/multiclass_detection_suite.git
    ```

2. Navigate to the project directory:
    ```bash
    cd multiclass_detection_suite
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Configure the YAML files as explained in the previous section.

5. Run the desired script:
    ```bash
    python main_behavior.py
    ```

## Requirements

- Python 3.x
- Recommended libraries: `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `tensorflow` (for MNIST), `optuna`, `yaml`
