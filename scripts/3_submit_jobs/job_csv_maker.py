"""Job CSV Maker

This script creates a CSV using a parameter grid where each row is an experimental setting.

"""
import os
import csv
from sklearn.model_selection import ParameterGrid

def write_paramgrid(script, grid):
    """Writes parameter grid for family of experiments.

    Args:
        script (str): name of python script to run.
        grid (sklearn.ParameterGrid): Grid of all combos of parameter values.
    """
    for i, params in enumerate(grid):
        with open(f'./params/{script}_params.csv', 'a', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            if i ==0:
                writer.writerow(params.keys())
            vals = list(params.values())
            writer.writerow(vals)
            f.close()

if __name__ == "__main__":
    # Check if params folder exists:
    PARAMS_DIR = './params/'
    if not os.path.exists(PARAMS_DIR):
        os.makedirs(PARAMS_DIR)
    # Set up SL transformers
    script = 'SL_transformers'
    # Set up grid
    param_grid = {'method': ['SL'],
                'framework': ['TF'],
                'datadir': ['./data'],
                'dataset' : ['wiki', 'tweets'],
                'outdir': ['./results'],
                'transformer_model': ['./distilroberta-base'],
                'n_epochs': [3],
                'class_imbalance': [50,10,5],
                'train_n': [20000],
                'test_n': [5000],
                'run_n': [3]
                }
    grid = ParameterGrid(param_grid)
    write_paramgrid(script, grid)

    # Set up SL SVM
    script = 'SL_sklearn'
    # Set up grid
    param_grid = {'method': ['SL'],
                'framework': ['SK'],
                'datadir': ['./data'],
                'dataset' : ['wiki', 'tweets'],
                'outdir': ['./results'],
                'sklearn_model': ['ConfidenceEnhancedLinearSVC'],
                'class_imbalance': [50,10,5],
                'train_n': [20000],
                'test_n': [5000],
                'run_n': [3]
                }
    grid = ParameterGrid(param_grid)
    write_paramgrid(script, grid)

    # Set up AL transformers
    script = 'AL_transformers'
    # Set up grid
    param_grid = {'method': ['AL'],
                'framework': ['TF'],
                'datadir': ['./data'],
                'dataset' : ['wiki', 'tweets'],
                'outdir': ['./results'],
                'transformer_model': ['./distilroberta-base'],
                'n_epochs': [3],
                'class_imbalance': [50, 10, 5],
                'init_n': [20, 200],
                'cold_strategy': ['TrueRandom',
                                  'BalancedWeak'],
                'query_n': [50, 100, 500],
                'query_strategy': ['RandomSampling()',
                                  'LeastConfidence()',
                                  'EmbeddingKMeans(normalize=True)',
                                  'GreedyCoreset(normalize=True)'],
                'train_n': [20000],
                'test_n': [5000],
                'run_n': [3]
                }
    grid = ParameterGrid(param_grid)
    write_paramgrid(script, grid)

    # Set up AL SVM
    script = 'AL_sklearn'
    # Set up grid
    param_grid = {'method': ['AL'],
                'framework': ['SK'],
                'datadir': ['./data'],
                'dataset' : ['wiki', 'tweets'],
                'outdir': ['./results'],
                'sklearn_model': ['ConfidenceEnhancedLinearSVC'],
                'class_imbalance': [50,10,5],
                'init_n': [20, 200],
                'cold_strategy': ['TrueRandom',
                                  'BalancedWeak'],
                'query_n': [50,100, 500],
                'query_strategy': ['RandomSampling()',
                                  'LeastConfidence()'],
                'train_n': [20000],
                'test_n': [5000],
                'run_n': [3]
                }
    grid = ParameterGrid(param_grid)
    write_paramgrid(script, grid)
