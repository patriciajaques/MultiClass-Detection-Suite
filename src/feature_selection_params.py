

def get_param_grid():
    return {
        'feature_selection__n_features_to_select': [10, 20, 30, 40, 50],  # Para RFE
        'feature_selection__n_components': [5, 10, 15, 20],  # Para PCA
    }
