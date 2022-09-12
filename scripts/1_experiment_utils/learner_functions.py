"""Learner Functions

This script contains the core workhorse functions for training and evaluating AL and SL processes.

"""
import logging
import random
from datetime import datetime
import json
import numpy as np
from small_text.active_learner import PoolBasedActiveLearner
from small_text.query_strategies import PoolExhaustedException, EmptyPoolException
from small_text.query_strategies.strategies import (QueryStrategy,
                                                    RandomSampling,
                                                    ConfidenceBasedQueryStrategy,
                                                    LeastConfidence,
                                                    EmbeddingBasedQueryStrategy,
                                                    EmbeddingKMeans)
from small_text.query_strategies.coresets import (greedy_coreset,
                                                  GreedyCoreset)                                                                                                    
from evaluation import evaluate

# Active Learning Functions
def initialize_learner(learner, train, test_sets, args):
    """Initalizes learner model for active learning or
    trains full model for supervised "passive" learning.
    """
    print('\n----Initalising----\n')
    iter_results_dict = {}
    iter_preds_dict = {}
    # Initialize the model (AL)
    if args.method == 'AL':
        # True Random Choice
        if args.cold_strategy =='TrueRandom':
            indices_neg_label = np.where(train.y == 0)[0]
            indices_pos_label = np.where(train.y == 1)[0]
            all_indices = np.concatenate([indices_neg_label, indices_pos_label])
            x_indices_initial = np.random.choice(all_indices,
                                                args.init_n,
                                                replace=False)
        # Balanced Random Choice Based on Known Class label
        elif args.cold_strategy == 'BalancedRandom': 
            indices_neg_label = np.where(train.y == 0)[0]
            indices_pos_label = np.where(train.y == 1)[0]
            selected_neg_label = np.random.choice(indices_neg_label,
                                                  int(args.init_n/2),
                                                  replace=False)
            selected_pos_label = np.random.choice(indices_pos_label,
                                                  int(args.init_n/2),
                                                  replace=False)
            x_indices_initial = np.concatenate([selected_neg_label, selected_pos_label])
        # Balanced Random Choice Based on Keywords (Weak label)
        elif args.cold_strategy == 'BalancedWeak': 
            indices_neg_label = np.where(train.yweak == 0)[0]
            indices_pos_label = np.where(train.yweak == 1)[0]
            if len(indices_pos_label) > int(args.init_n/2):
                selected_neg_label = np.random.choice(indices_neg_label,
                                                      int(args.init_n/2),
                                                      replace=False)
                selected_pos_label = np.random.choice(indices_pos_label,
                                                      int(args.init_n/2),
                                                      replace=False)
            # If limit reached, take as many positive as possible and pad with negatives
            else:
                selected_pos_label = np.random.choice(indices_pos_label,
                                                      len(indices_pos_label),
                                                      replace=False)
                selected_neg_label = np.random.choice(indices_neg_label,
                                                      int(args.init_n) - len(indices_pos_label),
                                                      replace=False)
            x_indices_initial = np.concatenate([selected_neg_label, selected_pos_label])
        else:
            print('Invalid Cold Start Policy')
        # Set x and y initial
        x_indices_initial = x_indices_initial.astype(int)
        y_initial = np.array([train.y[i] for i in x_indices_initial])
        print('y selected', train.y[x_indices_initial])
        print(f'Starting imbalance (train): {np.round(np.mean(y_initial),4)}')
        # Set validation indices for transformers framework
        if args.framework == 'TF':
            # After we initialize the learner, we know the label so can set val directly
            print('Setting val indices')
            indices_neg_label = np.where(train.y[x_indices_initial] == 0)[0]
            indices_pos_label = np.where(train.y[x_indices_initial] == 1)[0]
            print('idx_neg', indices_neg_label)
            print('idx_pos', indices_pos_label)
            # take 10% for init val
            val_neg_label = np.random.choice(indices_neg_label,
                                            int(0.1*(args.init_n/2)),
                                            replace=False)
            val_pos_label = np.random.choice(indices_pos_label,
                                            int(0.1*(args.init_n/2)),
                                            replace=False)
            print('val_neg', val_neg_label)
            print('val_pos', val_pos_label)
            val_indices = np.concatenate([val_neg_label, val_pos_label])
            print('val_indices', val_indices)
            y_val_initial = np.array([train.y[i] for i in x_indices_initial[val_indices]])
            print(f'Starting imbalance (val): {np.round(np.mean(y_val_initial),4)}')
        else:
            val_indices = None
        print('Initialising learner')
        learner.initialize_data(x_indices_initial, y_initial, x_indices_validation=val_indices)
        print('Learner initalized ok.')
    # SL Set-Up: for supervised args.init_n = total_n
    else:
        indices_neg_label = np.where(train.y == 0)[0]
        indices_pos_label = np.where(train.y == 1)[0]
        all_indices = np.concatenate([indices_neg_label, indices_pos_label])
        np.random.shuffle(all_indices)
        x_indices_initial = all_indices.astype(int)
        y_initial = np.array([train.y[i] for i in x_indices_initial])
        print(f'Starting imbalance: {np.round(np.mean(y_initial),2)}')
        if args.framework == 'TF':
            print('Setting val indices')
            val_indices = np.concatenate([np.random.choice(indices_pos_label, 
                                                          int(0.1*len(indices_pos_label)),
                                                          replace=False),
                                          np.random.choice(indices_neg_label,
                                                          int(0.1*len(indices_neg_label)),
                                                          replace=False)
                                          ])
        else:
            val_indices = None
        print('Initialising learner')
        learner.initialize_data(x_indices_initial, y_initial, x_indices_validation=val_indices)
        print('Learner initalized ok.')
    print('Evaluation step')
    iter_results_dict[int(0)], iter_preds_dict[int(0)] = evaluate(learner,
                                                                  train[x_indices_initial],
                                                                  test_sets,
                                                                  x_indices_initial)
    return learner, x_indices_initial, iter_results_dict, iter_preds_dict

def perform_active_learning(active_learner, train, test_sets, labeled_indices, iter_results_dict, iter_preds_dict, args):
    """Performs active learning for learner model until training budget is exhausted.
    Stores evaluation results at each iteration.
    """
    print('\n----Performing iterations----\n')
    print(f'Train total: {len(train)}')
    # Initialise i for storing results, 0 is populated by initialized model
    i = 1
    embedding_strategies = ['EmbeddingKMeans(normalize=True)', 'GreedyCoreset(normalize=True)']
    if str(active_learner.query_strategy) in embedding_strategies:
        print('Calculating embeddings')
        embeddings, proba = active_learner.classifier.embed(train, return_proba=True)
    while len(labeled_indices) + args.query_n < 2000:
        if str(active_learner.query_strategy) in embedding_strategies:
            print('Using embeddings')
            q_indices = active_learner.query(num_samples=args.query_n,
                                            query_strategy_kwargs = dict({'embeddings':embeddings}))
        else:
            q_indices = active_learner.query(num_samples=args.query_n)
        # Simulate user interaction here. Replace this for real-world usage.
        y = train.y[q_indices]
        # Return the label for the current query to the active learner.
        active_learner.update(y)
        labeled_indices = np.concatenate([q_indices, labeled_indices])
        print('Iteration #{:d} ({} samples)'.format(i, len(labeled_indices)))
        # Evaluate
        iter_results_dict[int(i)], iter_preds_dict[int(i)] = evaluate(active_learner,
                                                                    train[labeled_indices],
                                                                    test_sets,
                                                                    labeled_indices)
        # Update i
        i+=1
        print(f'Used indices: {len(labeled_indices)}')
        print(f'Remaining indices: {len(train)-len(labeled_indices)}')
    # Final batch of remaining indices
    query_n_final = 2000 - len(labeled_indices)
    print(f'Final round: {query_n_final}')
    q_indices = active_learner.query(num_samples=query_n_final)
    y = train.y[q_indices]
    active_learner.update(y)
    labeled_indices = np.concatenate([q_indices, labeled_indices])
    print('Iteration #{:d} ({} samples)'.format(i, len(labeled_indices)))
    # Evaluate
    iter_results_dict[i], iter_preds_dict[i] = evaluate(active_learner,
                                                        train[labeled_indices],
                                                        test_sets,
                                                        labeled_indices)
    # Update i
    i+=1
    print(f'Used indices: {len(labeled_indices)}')
    print(f'Remaining indices: {len(train)-len(labeled_indices)}')
    print('Budget is exhausted!')
    return active_learner, iter_results_dict, iter_preds_dict 

def run_AL(clf_factory, train, test_sets, args):
    """Runs full active learning process with initialisation and iterations."""
    active_learner = PoolBasedActiveLearner(clf_factory,eval(args.query_strategy),train)
    active_learner, labeled_indices, iter_results_dict, iter_preds_dict = initialize_learner(active_learner,
                                                                                            train,
                                                                                            test_sets,
                                                                                            args)
    active_learner, iter_results_dict, iter_preds_dict = perform_active_learning(active_learner,
                                                                                train,
                                                                                test_sets,
                                                                                labeled_indices,
                                                                                iter_results_dict,
                                                                                iter_preds_dict,
                                                                                args)
    return active_learner, iter_results_dict, iter_preds_dict

def run_SL(clf_factory, train, test_sets, args):
    """Runs full supervised "passive" learning process with initialisation as training."""
    supervised_learner = PoolBasedActiveLearner(clf_factory,RandomSampling(),train)
    supervised_learner, x_indices_initial, iter_results_dict, iter_preds_dict  = initialize_learner(supervised_learner, 
                                                                                                    train,
                                                                                                    test_sets,
                                                                                                    args)
    return supervised_learner, iter_results_dict, iter_preds_dict

def run_multiple_experiments(clf_factory, train, test_sets, matching_indexes, args):
    """Launches experiment for AL or SL, checking it hasn't already been run and logs results."""
    # Initialize output file to log experiment has started
    current_datetime = datetime.now()
    output = {}
    for arg in vars(args):
        output[arg] = getattr(args, arg)
    if args.method == 'SL':
        experiment_output_dir = f'{args.outdir}/{args.method}_{args.framework}_{args.dataset}_{args.class_imbalance}_{args.train_n}'
    elif args.method == 'AL':
        experiment_output_dir = f'{args.outdir}/{args.method}_{args.framework}_{args.dataset}_{args.class_imbalance}_{args.init_n}_{args.cold_strategy}_{args.query_strategy}_{args.query_n}'
    # Set up logging
    logging.basicConfig(filename=f"{experiment_output_dir}/log.txt",level=logging.DEBUG)
    logging.captureWarnings(True)
    logf = open(f"{experiment_output_dir}/err.log", "w")
    with open(f'{experiment_output_dir}/START_{current_datetime}.json', 'w') as fp:
        json.dump(output, fp)
    # Try running experiment
    try:
        results_dict = {}
        predictions_dict = {}
        # Run experiment n times
        for run in range(args.run_n):
            seed_value = run
            random.seed(seed_value)
            np.random.seed(seed_value)
            print(f'----RUN {run}: {args.method} LEARNER----')
            if args.method == 'SL':
                learner, run_results_dict, run_pred_dict = run_SL(clf_factory,
                                                                  train,
                                                                  test_sets,
                                                                  args)
            elif args.method == 'AL':
                learner, run_results_dict, run_pred_dict = run_AL(clf_factory,
                                                                  train,
                                                                  test_sets,
                                                                  args)
            # Save model and results
            learner.save(f'{experiment_output_dir}/model_run{run}_{current_datetime}.pkl')
            results_dict[f'run_{run}'] = run_results_dict
            current_datetime = datetime.now()
            with open(f'{experiment_output_dir}/results_run{run}_{current_datetime}.json', 'w') as fp:
                json.dump(results_dict, fp)
            predictions_dict[f'run_{run}'] = run_pred_dict
        # Save predictions and indexes to map back to original dataframe text
        predictions_dict['indexes'] = matching_indexes
        with open(f'{experiment_output_dir}/predictions.json', 'w') as fp:
            json.dump(predictions_dict, fp)
        # Log time
        current_datetime = datetime.now()
        # Save output
        output['results_dict'] = results_dict
        output['Error_Code'] = 'NONE'
        with open(f'{experiment_output_dir}/END_{current_datetime}.json', 'w') as fp:
            json.dump(output, fp)
        print('Finished with no errors!')
        logf.write("No errors!")
    # Catch errors
    except Exception as e:
        print(e)
        logf.write(f"time: {current_datetime}, error: {e}")
        # Reset params_dict
        output[arg] = getattr(args, arg)
        output['Error_Code'] = str(e)
        with open(f'{experiment_output_dir}/FAILED_{current_datetime}.json', 'w') as fp:
            json.dump(output, fp)
 