"""Data Loader

This script loads the datasets from file and prepares them for training and testing.

It requires you have already downloaded the files from Wulczyn2017 and Founta2018.

"""
import os
import re
import random
from cleaning_functions import (clean_text, 
                                drop_nans,
                                drop_duplicates,
                                drop_empty_text,
                                drop_url_emoji)
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import (confusion_matrix, 
                            accuracy_score,
                            precision_score,
                            recall_score,
                            f1_score)
from sklearn.model_selection import train_test_split
random.seed(123)

# FUNCTIONS
def load_wiki(folderpath):
    """Loads raw wiki data (Wulczyn2017) from folder, cleans text and returns train, test sets.
    See https://github.com/ewulczyn/wiki-detox/

    Args:
        folderpath (str): location of raw dataset.

    Returns:
        pd.DataFrame, pd.DataFrame: train and test sets as pd.Dataframe.
    """
    df = pd.read_csv(f'{folderpath}/attack_annotated_comments.tsv', sep = '\t', index_col = 0)
    annotations = pd.read_csv(f'{folderpath}/attack_annotations.tsv',  sep = '\t')
    # labels a comment as an atack if the majority of annoatators did so
    labels = annotations.groupby('rev_id')['attack'].mean() > 0.5
    # join binary labels to comments
    df['label'] = labels * 1
    # remove newline, tab tokens and ==
    df['comment'] = df['comment'].apply(lambda x: x.replace("NEWLINE_TOKEN", " "))
    df['comment'] = df['comment'].apply(lambda x: x.replace("TAB_TOKEN", " "))
    df['comment'] = df['comment'].apply(lambda x: x.replace("==", ""))
    # rename columns
    df = df.rename(columns = {'comment': 'text'})
    # clean data
    df = clean_data(df)
    # create train, test sets
    print('\n--Creating base test and train sets---')
    test = df[df['split']=='test']
    train = df[df['split']!='test']
    # keep cols
    test = test[['clean_text', 'label']]
    test = test.rename(columns = {'clean_text':'text'})
    abuse = len(test[test['label']==1])
    print(f'base_test:\nlen: {len(test)}, pct_abuse: {np.round(abuse/len(test),3)}')
    train = train[['clean_text', 'label']]
    train = train.rename(columns = {'clean_text':'text'})
    abuse = len(train[train['label']==1])
    print(f'base_train:\nlen: {len(train)}, pct_abuse: {np.round(abuse/len(train),3)}')
    return train, test

def load_tweets(folderpath):
    """Loads raw tweets data (Founta2018) from folder, cleans text and returns train, test sets.

    Args:
        folderpath (str): location of raw dataset.

    Returns:
        pd.DataFrame, pd.DataFrame: train and test sets as pd.Dataframe.
    """
    df = pd.read_csv(f'{folderpath}/hatespeech_text_label_vote.csv', sep = '\t',
        encoding='utf-8', header=None)
    df = df.rename(columns = {0:'tweet', 1:'label', 2:'vote'})
    # binarize labels
    df['binary_label'] = df['label'].map(lambda x: 0 if x in ['spam', 'normal'] else 1)
    # rename columns
    df = df.rename(columns = {'tweet':'text'})
    # clean data
    df = clean_data(df)
    # Split of 10% test set
    train, test = train_test_split(df, test_size=0.101, shuffle = True, random_state=42)
    # create train, test sets
    print('\n--Creating base test and train sets---')
    test = test[['clean_text', 'binary_label']]
    test = test.rename(columns = {'clean_text':'text', 'binary_label':'label'})
    abuse = len(test[test['label']==1])
    print(f'base_test:\nlen: {len(test)}, pct_abuse: {np.round(abuse/len(test),3)}')
    train = train[['clean_text', 'binary_label']]
    train = train.rename(columns = {'clean_text':'text', 'binary_label':'label'})
    abuse = len(train[train['label']==1])
    print(f'base_train:\nlen: {len(train)}, pct_abuse: {np.round(abuse/len(train),3)}')
    return train, test

def return_kw_matches(df, kw_list):
    """Searches for keywords in defined keyword list and returns all matches as list.

    Args:
        df (pd.DataFrame): dataframe with text column for matches.
        kw_list (list of str): list of keywords.

    Returns:
        pd.DataFrame: Modified df with additional 'matches' column.
    """
    regx = re.compile(r'\b(?:%s)\b' % '|'.join(kw_list))
    df['matches'] = df['text'].map(lambda x: re.findall(regx, x))
    return df

def calc_kw_label(df, threshold):
    """Weakly labels text examples based on their keyword density (at given threshold).

    Args:
        df (pd.DataFrame): dataframe with text and matches columns for weak labeling.
        threshold (float): pct of text tokens which have to be keyword matches for weak label.

    Returns:
        pd.DataFrame: Modified df with additional threshold column.
    """
    df['len_matches'] = df['matches'].map(lambda x: len(x))
    df['len_text'] = df['text'].map(lambda x: len(x.split(' ')))
    df['norm_kw'] = df['len_matches'] / df['len_text']
    df['weak_pos_keywords'] = df['norm_kw'].map(lambda x: 1 if x>threshold else 0)
    return df

def eval_kw_threshold(df, threshold):
    """Prints evaluation of how accurately keyword weak labels reflect true labels.

    Args:
        df (pd.DataFrame): dataframe with text column with weak labels.
        threshold (float): pct of text tokens which have to be keyword matches for weak label.
    """
    cm = confusion_matrix(df['label'], df['weak_pos_keywords'])
    accuracy = accuracy_score(df['label'], df['weak_pos_keywords'])
    precision = precision_score(df['label'], df['weak_pos_keywords'], average='macro')
    recall = recall_score(df['label'], df['weak_pos_keywords'])
    f1 = f1_score(df['label'], df['weak_pos_keywords'], average='macro')
    TN, FP, FN, TP = cm.ravel()
    # False positive rate
    FPR = FP/(FP+TN)
    # False negative rate
    FNR = FN/(TP+FN)
    print(f'''
    Threshold:{threshold};
    Accuracy={np.round(accuracy,2)};
    Recall={np.round(recall,2)};
    Precision={np.round(precision,2)}; 
    F1={np.round(f1,2)}; 
    FPR={np.round(FPR,2)}; 
    FNR:{np.round(FNR,2)}\n
    ''')

def clean_data(df):
    """Cleans data using functions from cleaning_functions.py.

    Args:
        df (pd.DataFrame): input dataframe.

    Returns:
        pd.DataFrame: output cleaned dataframe.
    """
    print('\n---Dropping NaNs---')
    df = drop_nans(df, subset_col = 'text', verbose = True)
    print('\n---Dropping duplicates---')
    df = drop_duplicates(df, subset_col = 'text', verbose = True)
    print('\n---Cleaning text---')
    df['clean_text'] = df['text'].apply(clean_text)
    print('\n---Dropping empty text entries---')
    df = drop_empty_text(df, subset_col = 'clean_text', verbose = True)
    print('\n---Dropping text entries with only URL + EMOJI---')
    df = drop_url_emoji(df, subset_col = 'clean_text', verbose = True)
    print('\n---Checking text length---')
    df['text_length'] = df['clean_text'].map(lambda x: len(x))
    print('Summary statistics of text length:')
    print(df['text_length'].describe())
    return df

def class_sampler_with_budget(DATASET, split, df, col='label', neg_pcts = [0.95, 0.9, 0.5],budget = 20000):
    """Takes random samples to artificially enforce class balance, saves to csv.

    Args:
        DATASET (str): name of dataset used to save csvs.
        split (str): string to identify train or test set.
        df (pd.DataFrame): input dataframe.
        col (str, optional): name of column with labels. Defaults to 'label'.
        neg_pcts (list, optional): list of artificial class imbalances for label == 0. Defaults to [0.95, 0.9, 0.5].
        budgets (int, optional): maximum budget enforced for training. Defaults to 20000.
    """
    print(f'\n----Creating {split} sets----\n')

    noabuse = df[df[col]==0]
    abuse = df[df[col]==1]
    print('total len', len(df))
    print('abuse len', len(abuse))
    print('no abuse len', len(noabuse))
    if len(noabuse) >= len(abuse):
        print('majority class no abuse')
    else:
        print('majority class abuse')
    for pct in neg_pcts:
        print(f'\npct (abuse): {np.round(1-pct,2)}')
        print('--------\n')
        noabuse_n = int(pct * budget)
        abuse_n = budget - noabuse_n
        print('abuse', abuse_n)
        print('no_abuse', noabuse_n)
        # len of output df should always be equal to total budget
        assert abuse_n + noabuse_n == budget
        abuse_sample = abuse.sample(abuse_n)
        noabuse_sample = noabuse.sample(noabuse_n)
        # concat
        output_df = pd.concat([abuse_sample, noabuse_sample])
        assert len(output_df) == budget
        #shuffle
        output_df = shuffle(output_df)
        print(f'value counts:\n{output_df[col].value_counts(normalize = True)}')
      # Save csv
        save_val = int(pct*100)
        save_val = 100 - save_val
        if split == 'train':
            output_df.to_csv(f'{DIR}/data/{DATASET}/{split}_{budget}_{save_val}.csv')
        else:
            output_df.to_csv(f'{DIR}/data/{DATASET}/{split}_{save_val}.csv')

def main():
    for DATASET in ['wiki', 'tweets']:
        directory = f'{DIR}/data/{DATASET}'
        if not os.path.exists(directory):
            os.makedirs(directory)
        # Load data
        if DATASET == 'wiki':
            base_train, base_test = load_wiki(f'{DIR}/data/Wulczyn2017')
        elif DATASET == 'tweets':
            base_train, base_test = load_tweets(f'{DIR}/data/Founta2018')
        # Load keywords
        print('\n---Loading Keywords---')
        f = open(f"{DIR}/data/keywords.txt", "r")
        content = f.read()
        kw_list = content.splitlines()
        f.close()
        kw_list = list(set(kw_list))
        print(f'Number of keywords: {len(kw_list)}')
        # Calculate keyword heuristics
        base_train = return_kw_matches(base_train, kw_list)
        # Check varied thresholds
        print('\n---Evaluating Thresholds---')
        for t in [0.01, 0.05, 0.1, 0.25]:
            base_train_t = calc_kw_label(base_train, t)
            eval_kw_threshold(base_train_t, t)
        # Use threshold of 0.05
        base_train = calc_kw_label(base_train, 0.05)
        # Save base datasets
        base_test.to_csv(f'{DIR}/data/{DATASET}/train_base.csv', index = True)
        base_train.to_csv(f'{DIR}/data/{DATASET}/train_base.csv', index = True)
        # Create artifical test sets
        class_sampler_with_budget(
            DATASET, 'test', base_test, 'label', neg_pcts = [0.95, 0.9, 0.5], budget = 5000
            )
        # Create artificial train sets
        class_sampler_with_budget(
            DATASET, 'train', base_train, 'label', neg_pcts = [0.95, 0.9, 0.5], budget = 20000
            )

if __name__ == "__main__":
    DIR = './ActiveTransformers-for-AbusiveLanguage'
    main()
