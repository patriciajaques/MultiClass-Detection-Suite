import os
import pandas as pd
import numpy as np
import pytest
from src.data_exploration import vis_corr_cat

def test_vis_corr_cat():
    # Create sample data
    X = pd.DataFrame({
        'A': ['a', 'b', 'a', 'b', 'a'],
        'B': ['x', 'y', 'x', 'y', 'x'],
        'C': ['p', 'q', 'p', 'q', 'p']
    })
    y = pd.DataFrame({
        'Y': [1, 0, 1, 0, 1],
        'Z': [0, 1, 0, 1, 0]
    })

    # Call the function
    vis_corr_cat(X, y, output_dir='test_output', batch_size=2)

    # Check if the output files are generated
    assert os.path.exists('test_output/heatmap_batch_1.png')
    assert os.path.exists('test_output/heatmap_batch_2.png')

    # Clean up the generated files
    os.remove('test_output/heatmap_batch_1.png')
    os.remove('test_output/heatmap_batch_2.png')
    os.rmdir('test_output')