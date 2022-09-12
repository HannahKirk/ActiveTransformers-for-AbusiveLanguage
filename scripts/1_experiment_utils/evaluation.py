"""Evaluation

This script defines the evaluation function for an experimental run.

"""
from sklearn.metrics import f1_score
import numpy as np

# Evaluation functions
def evaluate(active_learner, train, test_sets, labeled_indices):
    results_dict = {}
    pred_arrays = {}
    # Train score
    y_pred = active_learner.classifier.predict(train) # sort order of y_true y_pred
    train_score = f1_score(train.y, y_pred, average='macro', zero_division = 0)
    print(f'Train accuracy: {np.round(train_score,2)}')
    results_dict['train_f1'] = train_score
    results_dict['train_imbalance'] = np.mean(train.y)
    # Test scores
    for k in test_sets.keys():
        y_pred_test = active_learner.classifier.predict(test_sets[k])
        test_score = f1_score(test_sets[k].y, y_pred_test, average='macro', zero_division = 0)
        print(f'Test accuracy (pct = {k}): {np.round(test_score,2)}')
        results_dict[f'{k}_f1'] = test_score
        pred_arrays[f'{k}_preds'] = y_pred_test.tolist()
    print(f'num_examples: {len(labeled_indices)}')
    results_dict['num_examples'] = len(labeled_indices)
    print('---')
    return results_dict, pred_arrays
