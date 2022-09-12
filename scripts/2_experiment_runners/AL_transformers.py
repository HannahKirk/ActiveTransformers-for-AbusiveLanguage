"""AL transformers

This script runs a single experiment as part of a set of multiple experiments.
It runs experiments for active learning (AL) in the transformers framework.

"""
import argparse
from transformers import AutoTokenizer
from factories import TransformerBasedClassificationFactory
from small_text.integrations.transformers import TransformerModelArguments
from small_text.query_strategies import RandomSampling
from preprocess import data_loader, preprocess_data_transformers, df_to_dict
from learner_functions import run_multiple_experiments

# Arg parser
def parse_args():
    parser=argparse.ArgumentParser(description="Active Learning Experiment Runner with Transformers Integration")
    parser.add_argument('--method', type = str, metavar ="", default = 'AL', help="Supervised == SL or Active == AL")
    parser.add_argument('--framework', type = str, metavar ="", default = 'TF', help="Transformers == TF or SkLearn == SK")
    parser.add_argument('--datadir', type = str, metavar ="",default = './data/', help="Path to directory with data files")
    parser.add_argument('--dataset', type = str, metavar ="",default = 'attacks', help="Name of dataset")
    parser.add_argument('--outdir', type = str, metavar ="",default = './results/', help="Path to output directory for storing results")
    parser.add_argument('--transformer_model', type = str, metavar ="",default = 'distilroberta-base', help="Name of HuggingFace transformer model")
    parser.add_argument( '--n_epochs', type = int, metavar ="",default =  3, help = "Number of epochs for model training")
    parser.add_argument('--class_imbalance', type = int, metavar ="", default = 50, help = 'Class imbalance desired in train dataset')
    parser.add_argument('--init_n', type = int, metavar ="", default = 100, help = 'Initial batch size for training')
    parser.add_argument('--cold_strategy', metavar ="", default = 'TrueRandom', help = 'Method of cold start to select initial examples')
    parser.add_argument('--query_n', type = int, metavar ="", default = 100, help = 'Batch size per active learning query for training')
    parser.add_argument('--query_strategy', metavar ="", default = RandomSampling(), help = 'Method of active learning query for training')
    parser.add_argument('--train_n', type = int, metavar ="", default = 1000, help = 'Total number of training examples')
    parser.add_argument('--test_n', type = int, metavar ="", default = 1000, help = 'Total number of testing examples')
    parser.add_argument('--run_n', type = int, metavar ="", default = 5, help = 'Number of times to run each model')
    args=parser.parse_args()
    print("the inputs are:")
    for arg in vars(args):
        print("{} is {}".format(arg, getattr(args, arg)))
    return args

def main():
    # Get args
    args=parse_args()
    # Load data
    train_df, test_dfs = data_loader(args)
    # Prepare experiment model and data
    TRANSFORMER_MODEL = TransformerModelArguments(args.transformer_model)
    clf_factory = TransformerBasedClassificationFactory(TRANSFORMER_MODEL,
                                                    2,
                                                    args.n_epochs,
                                                    kwargs = dict({'device': 'cuda'}))
    tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL.model, cache_dir='.cache/')
    tokenizer.add_special_tokens(["[URL]", "[EMOJI]", "[USER]"])
    train_dict = df_to_dict('train', train_df)
    train = preprocess_data_transformers('train',
                                        tokenizer,
                                        train_dict['data'],
                                        train_dict['target'],
                                        train_dict['weak_target'])
    test_sets = {}
    matching_indexes = {}
    for j in test_dfs.keys():
        matching_indexes[j] = test_dfs[j].index.tolist()
        data_dict = df_to_dict('test', test_dfs[j])
        processed_data = preprocess_data_transformers('test',
                                                    tokenizer,
                                                    data_dict['data'],
                                                    data_dict['target'])
        test_sets[j] = processed_data
    # Run experiments
    run_multiple_experiments(clf_factory, train, test_sets, matching_indexes, args)

if __name__ == '__main__':
    main()
