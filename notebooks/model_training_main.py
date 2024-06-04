# %%
import importlib

import training
import utils
import training_constants as tc
from preprocessing import encode_labels, load_data, split_train_test_data

# %%
# importlib.reload(training)
# importlib.reload(utils)

import os
import warnings
import numpy as np

# Configurar variáveis de ambiente
os.environ["MKL_CBWR"] = "AUTO"
os.environ["MKL_VERBOSE"] = "0"
os.environ["NUMPY_MKL_ERROR"] = "IGNORE"
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["MKL_DYNAMIC"] = "FALSE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_WARNINGS"] = "FALSE"

# Suprimir avisos de depreciação
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Redefinir a configuração do NumPy
np.__config__.show()


# %%
X, y = load_data("/Users/patricia/Documents/code/python/behavior-detection/data/new_logs_labels.csv")
y.head()

# %%
X_train, X_test, y_train, y_test = split_train_test_data(X, y)


# %%
y_train, label_encoder = encode_labels(y_train)
y_test = label_encoder.transform(y_test)


# %%

#importlib.reload(training)
trained_models = training.train_model(X_train, y_train, tc.BAYESIAN_OPTIMIZATION)

# %%
import evaluation
import importlib

#importlib.reload(evaluation)
dirpath = "/Users/patricia/Documents/code/python/behavior-detection/output/"

reports = evaluation.generate_reports(trained_models, X_train, y_train, X_test, y_test)
print(evaluation.print_reports(reports, dirpath))
evaluation.save_reports_to_csv(reports, dirpath)


