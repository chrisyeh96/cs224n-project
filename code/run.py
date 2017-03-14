from __future__ import print_function

import tensorflow as tf
import numpy as np
import sys, os, time, pickle
from similarity_model import SimilarityModel
import argparse

TRAIN_DATA_PATH = "../data/quora/train.tsv"
TEST_DATA_PATH = "../data/quora/test.tsv"
GLOVE_VECTORS_PATH = "../data/glove/glove.6B.200d.npy"
TOKENS_TO_INDEX_PATH = "../data/glove/glove.6B.200d.pkl"
MAX_LENGTH_PATH = "../data/quora/max_length.pkl"

class Config:
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    # each word just indexes into glove vectors
    dropout = 0.5
    # word vector dimensions
    embed_size = 50
    hidden_size = 100
    output_size = 50
    batch_size = 2048
    n_epochs = 20
    max_grad_norm = 10.
    lr = 0.001

    distance_measure = "l2" # one of ["l2", "cosine", "custom_coef"]
    cell = "rnn" # one of ["rnn", "gru"]
    regularization_constant = 0.1

def normalize(word):
    """
    Normalize words that are numbers or have casing.
    """
    if word.isdigit(): return NUM
    else: return word.lower()

def read_datafile(fstream):
    """
    Reads a input stream @fstream (e.g. output of `open(fname, 'r')`) in TSV file format.
    Input file is formatted as follows:
        QUESTION1
        QUESTION2
        LABEL
        ...
    where QUESTION1 and QUESTION2 are tab-delimited strings, and LABEL is an int.

    @returns a list of examples [([sentence1, sentence2], label)].
        @sentence1 and @sentence2 are lists of strings, @label is a boolean
    """
    examples = []
    sentence1 = sentence2 = label = None

    for line_num, line in enumerate(fstream):
        line = line.strip()
        if line_num % 3 == 0:
            sentence1 = line.split("\t")
        elif line_num % 3 == 1:
            sentence2 = line.split("\t")
        else:
            label = int(line)
            examples.append((sentence1, sentence2, label))

    return examples

def load_and_preprocess_data(train_file_path, dev_file_path, tokens_to_glove_index_path, max_length_path):
    """
    Reads the training and dev data sets from the given paths.
    TODO: should we have train/validation/test split instead of just train/dev?
    """
    print("Loading training data...")
    with open(train_file_path) as train_file:
        train = read_datafile(train_file)
    print("Done. Read %d sentences" % len(train))
    print("Loading dev data...")
    with open(dev_file_path) as dev_file:
        dev = read_datafile(dev_file)
    print("Done. Read %d sentences"% len(dev))

    # now process all the input data: turn words into the glove indices
    print("Converting words into glove vector indices...")
    helper = ModelHelper.load(tokens_to_glove_index_path, max_length_path)
    train_data = helper.vectorize(train)
    dev_data = helper.vectorize(dev)

    return helper, train_data, dev_data, train, dev

class ModelHelper(object):
    """
    This helper takes care of preprocessing data, constructing embeddings, etc.
    """
    def __init__(self, tok2id, max_length):
        self.tok2id = tok2id
        self.UNKNOWN_WORD_INDEX = len(tok2id)
        self.PADDING_WORD_INDEX = len(tok2id) + 1
        # TODO: If we can have different amounts of padding for training vs. testing data,
        # then we can just compute the max_length in the vectorize functions.
        # Otherwise, we should load in max_length from some saved PKL file
        # self.max_length = max_length
        self.max_length = 25 # TODO make constant or sth

    # add additional embeddings for unknown word and padding word 
    def add_additional_embeddings(self, embeddings):
        '''Creates additional embeddings for unknown words and the padding word
        Returns a (2, embed_size) numpy array:
        - 0th row is word vector for unknown word, average of some known words
        - 1st row is word vector for padding word, all zeros
        '''
        unknown_word_vector = np.mean(embeddings[:100, :], axis=0, dtype=np.float32) # vector for unknown word
        padding_word_vector = np.zeros(embeddings.shape[1], dtype=np.float32)
        self.additional_embeddings = np.stack([unknown_word_vector, padding_word_vector])

    def pad_or_truncate_sentence(self, sentence):
        """Ensures @sentence is of length self.max_length by padding it with
        self.PADDING_WORD_INDEX at the beginning of the sentence or by truncating the
        rest of the sentence.

        Args:
            sentence: a list of integers representing word indices
        Returns:
            an integer numpy array of length self.max_length representing the sentence
        """
        new_sentence = np.zeros(self.max_length, dtype=np.int32)
        initial_length = len(sentence)
        if initial_length < self.max_length:
            num_padding = self.max_length - initial_length
            new_sentence = [self.PADDING_WORD_INDEX]*num_padding + sentence
        elif initial_length >= self.max_length:
            new_sentence = sentence[0:self.max_length]
        return new_sentence

    def vectorize_example(self, example):
        s1, s2, label = example
        s1_vectorized = [self.tok2id.get(word, self.UNKNOWN_WORD_INDEX) for word in s1]
        s2_vectorized = [self.tok2id.get(word, self.UNKNOWN_WORD_INDEX) for word in s2]
        s1_vectorized = self.pad_or_truncate_sentence(s1_vectorized)
        s2_vectorized = self.pad_or_truncate_sentence(s2_vectorized)
        return (s1_vectorized, s2_vectorized, label)

    def vectorize(self, data):
        return [self.vectorize_example(example) for example in data]

    @classmethod
    def load(cls, tokens_to_glove_index_path, max_length_path):
        # Make sure the directory exists.
        assert os.path.exists(tokens_to_glove_index_path)
        with open(tokens_to_glove_index_path, 'rb') as f:
            tok2id = pickle.load(f)
        with open(max_length_path, 'rb') as f:
            max_length = pickle.load(f)
        return cls(tok2id, max_length)


if __name__ == "__main__":
    description = "Run the similarity_model"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-b", "--batch_size", type=int, required=False, help="number of examples for each minibatch")
    parser.add_argument("-c", "--cell", required=False, choices=["rnn", "gru"], help="model cell type")
    parser.add_argument("-d", "--distance_measure", required=False, choices=["l2", "cosine", "custom_coef"], help="distance measure")
    parser.add_argument("-r", "--reg_constant", type=float, required=False, help="regularization constant")
    args = parser.parse_args()

    config = Config()
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.cell is not None:
        config.cell = args.cell
    if args.distance_measure is not None:
        config.distance_measure = args.distance_measure
    if args.reg_constant is not None:
        config.regularization_constant = args.reg_constant

    print("Preparing data...")
    helper, train, dev, train_raw, dev_raw = load_and_preprocess_data(TRAIN_DATA_PATH, TEST_DATA_PATH, TOKENS_TO_INDEX_PATH, MAX_LENGTH_PATH)
    
    print("Load embeddings...")
    embeddings = np.load(GLOVE_VECTORS_PATH, mmap_mode='r')
    config.embed_size = embeddings.shape[1]

    # append unknown word and padding word vectors
    helper.add_additional_embeddings(embeddings)

    with tf.Graph().as_default():
        print("Building model...")
        start = time.time()
        model = SimilarityModel(helper, config, embeddings)
        print("took %.2f seconds" % (time.time() - start))

        init = tf.global_variables_initializer()
        saver = None

        with tf.Session() as session:
            session.run(init)
            model.fit(session, saver, train, dev)

