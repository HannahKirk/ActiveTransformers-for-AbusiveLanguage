"""Preprocessing

These functions processes train and test data from csv into modelling format.

"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from small_text.data import SklearnDataset
from TransformersDatasetWeak import TransformersDatasetWeak
from SklearnDatasetWeak import SklearnDatasetWeak


def data_loader(args):
    """Loads train and test data from csv.

    Args:
        args: CLI args for experimental parameters.

    Returns:
        pd.DataFrame: training dataset.
        dict: test datasets stored as dictionary for each class imbalance.
    """
    # train data at artificial class imbalance
    train_df = pd.read_csv(f'{args.datadir}/{args.dataset}/train_{args.train_n}_{args.class_imbalance}.csv', index_col = 0)
    # test data
    test_dfs = {}
    for test_set in ['base', '50', '10', '5']:
        df = pd.read_csv(f'{args.datadir}/{args.dataset}/test_{test_set}.csv', index_col = 0)
        df = df[0:args.test_n]
        exec(f"test_dfs['test_{test_set}'] = df")
    return train_df, test_dfs

def df_to_dict(split, input_df):
    """Takes pandas df as input and dicitonary returns dictionary of text data and target labels.

    Args:
        split (str): Identifies train or test split.
        input_df (pd.DataFrame): input dataframe with texts, labels, weak labels.

    Returns:
        dict: Dictionary of 'data', 'target' (true labels) and 'weak target' (keyword-based labels)
    """
    # Initialize and populate dicitonary
    ddict = {}
    ddict['data'] = input_df['text'].to_list()
    ddict['target'] = input_df['label'].to_numpy()
    if split == 'train':
        ddict['weak_target'] = input_df['weak_pos_keywords'].to_numpy()
    return ddict

def preprocess_data_sklearn_train(data, targets, weak_targets):
    """Preprocesses training data for SVM model with TF-IDF.

    Args:
        data (list): text entries.
        targets (np.array): true labels.
        weak_targets (np.array): weak labels.

    Returns:
        SklearnDatasetWeak: a small_text Dataset class.
        sklearn.TfidfVectorizer: the fitted vectorizer to be used on the test set.
    """
    vectorizer = TfidfVectorizer(stop_words='english')
    x_train = normalize(vectorizer.fit_transform(data))
    return SklearnDatasetWeak(x_train, targets, weak_targets), vectorizer

def preprocess_data_sklearn_test(data, targets, vectorizer):
    """Preprocesses test data for SVM model with TF-IDF fitted on train data.

    Args:
        data (list): text entries.
        targets (np.array): true labels.
        weak_targets (np.array): weak labels.

    Returns:
        SklearnDataset: a small_text Dataset class.
    """
    # for test, ignore weak labels
    x_test = normalize(vectorizer.transform(data))
    return SklearnDataset(x_test, targets)


def preprocess_data_transformers(split, tokenizer, data, labels, weak_labels=None, max_length=256):
    """Preprocesses train and test data for Transformers model with tokenizer.

    Args:
        split (str): identifies train or test set.
        tokenizer (_type_): the tokenizer.
        data (list): text entries.
        labels (np.array): true labels.
        weak_labels (np.array, optional): weak labels. Defaults to None.
        max_length (int, optional): Maximum encoded length. Defaults to 256.

    Returns:
        TransformersDatasetWeak: a small_text Dataset class.
    """
    data_out = []
    for i, doc in enumerate(data):
        encoded_dict = tokenizer.encode_plus(
            doc,
            add_special_tokens=True,
            padding='max_length',
            max_length=max_length,
            return_attention_mask=True,
            return_tensors='pt',
            truncation='longest_first'
        )
        if split == 'train':
            data_out.append(
              (encoded_dict['input_ids'], encoded_dict['attention_mask'], labels[i], weak_labels[i])
              )
        # for test, ignore weak labels
        else:
            data_out.append(
              (encoded_dict['input_ids'], encoded_dict['attention_mask'], labels[i], labels[i])
              )
    return TransformersDatasetWeak(data_out)
