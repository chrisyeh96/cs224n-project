"""
Calculates the percent of tokens in a data file that appear in a vocabulary.

Usage: python count_vocab_matches.py -v VOCAB_PATH -d DATA_PATH
- VOCAB_PATH: path to Pickle file containing a set or list of vocab words,
    or a dictionary whose keys are the vocab words
- DATA_PATH: path to tokenized data file such as the output of data_preprocess.py,
    has the following format where question1 and question2 have tokens separated
    by spaces
    
    question1
    question2
    label
    question1
    question2
    label
    ...
"""
from __future__ import print_function
import argparse
import pickle

DEFAULT_VOCAB_PATH = '../data/glove/glove.6B.200d.pkl'
DEFAULT_DATA_PATH = '../data/quora/tokenized_data.tsv'

def main(vocab_path, data_path):
    print('vocab_path: %s' % vocab_path)
    print('data_path: %s' % data_path)

    # convert input path to desired output format, write to temp file
    with open(vocab_path, 'rb') as vocab_pkl_file:
        vocab = pickle.load(vocab_pkl_file)

    num_total_tokens = 0
    num_tokens_in_vocab = 0

    with open(data_path, 'r') as data_file:
        for line_num, line in enumerate(data_file):
            if line_num % 3 == 2: # label line
                pass
            
            line = line.strip().split()
            for token in line:
                num_total_tokens += 1
                if token in vocab:
                    num_tokens_in_vocab += 1

    print('%d of the %d tokens appeared in the vocab.' % (num_tokens_in_vocab, num_total_tokens))
    fraction = num_tokens_in_vocab / float(num_total_tokens)
    print('Fraction: %f' % fraction)


if __name__ == "__main__":
    description = "Calculate the percent of tokens in a data file that appear in a vocabulary."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-v", "--vocab", required=False, help="path to vocabulary PKL fil")
    parser.add_argument("-d", "--data", required=False, help="path to tokenized data TSV file")
    args = parser.parse_args()

    vocab_path = DEFAULT_VOCAB_PATH
    if args.vocab is not None:
        vocab_path = args.vocab

    data_path = DEFAULT_DATA_PATH
    if args.data is not None:
        data_path = args.data
    
    main(vocab_path, data_path)