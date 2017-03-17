from __future__ import print_function

import tensorflow as tf
import numpy as np
import sys, os, time, pickle
from similarity_model import SimilarityModel
import argparse

DATA_PATH = "../data/quora/tokenized_data.tsv"
DATA_SPLIT_INDICES_PATH = "../data/quora/data_split_indices.npz"
GLOVE_VECTORS_PATH = "../data/glove/glove.6B.300d.npy"
TOKENS_TO_GLOVEID_PATH = "../data/glove/glove.6B.300d.pkl"
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
    embed_size = 300
    hidden_size = 250
    output_size = 50
    batch_size = 1024
    n_epochs = 10
    max_grad_norm = 10.
    lr = 0.001
    n_classes = 2
    max_length = 30

    distance_measure = "l2" # one of ["l2", "cosine", "custom_coef"]
    cell = "gru" # one of ["rnn", "gru"]
    regularization_constant = 0.0001

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
    where QUESTION1 and QUESTION2 are space-delimited strings, and LABEL is an int.

    @returns a list of examples [(sentence1, sentence2, label)].
        @sentence1 and @sentence2 are lists of strings, @label is a boolean
    """
    examples = []
    sentence1 = sentence2 = label = None

    for line_num, line in enumerate(fstream):
        line = line.strip()
        if line_num % 3 == 0:
            sentence1 = line.split()
        elif line_num % 3 == 1:
            sentence2 = line.split()
        else:
            label = int(line)
            examples.append((sentence1, sentence2, label))

    return examples

def load_and_preprocess_data(data_path, data_split_indices_path, tokens_to_gloveID_path, max_length):
    """
    Reads the training and dev data sets from the given paths.
    TODO: should we have train/validation/test split instead of just train/dev?
    """
    print("Loading all data...")
    with open(data_path, 'r') as data_file:
        data = read_datafile(data_file)
    print("Done. Read %d sentences" % len(data))

    # now process all the input data: turn words into the glove indices
    print("Converting words into glove vector indices...")
    helper = ModelHelper.load(tokens_to_gloveID_path, max_length)
    data_vectorized = helper.vectorize(data)

    # split into train, dev, and test sets
    with np.load(data_split_indices_path) as data_split_indices:
        train_indices = data_split_indices['train']
        dev_indices = data_split_indices['dev']
        test_indices = data_split_indices['test']

    train_data = [data_vectorized[i] for i in train_indices]
    dev_data = [data_vectorized[i] for i in dev_indices]
    test_data = [data_vectorized[i] for i in test_indices]

    return helper, train_data, dev_data, test_data

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
        self.max_length = max_length
        # self.max_length = 25 # TODO make constant or sth

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
    def load(cls, tokens_to_gloveID_path, max_length):
        # Make sure the directory exists.
        assert os.path.exists(tokens_to_gloveID_path)
        with open(tokens_to_gloveID_path, 'rb') as f:
            tok2id = pickle.load(f)
        # with open(max_length_path, 'rb') as f:
            # max_length = pickle.load(f)
        return cls(tok2id, max_length)


if __name__ == "__main__":
    description = "Run the similarity_model"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-b", "--batch_size", type=int, required=False, help="number of examples for each minibatch")
    parser.add_argument("-c", "--cell", required=False, choices=["rnn", "gru"], help="model cell type")
    parser.add_argument("-d", "--distance_measure", required=False, choices=["l2", "cosine", "custom_coef"], help="distance measure")
    parser.add_argument("-r", "--reg_constant", type=float, required=False, help="regularization constant")
    parser.add_argument("-hs", "--hidden_size", type=int, required=False, help="neural net hidden size")
    parser.add_argument("-ml", "--max_length", type=int, required=False, help="maximum length of sentences")
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
    if args.hidden_size is not None:
        config.hidden_size = args.hidden_size
    if args.max_length is not None:
        config.max_length = args.max_length

    with tf.Graph().as_default():
        print("Building model...")
        start = time.time()
        model = SimilarityModel(helper, config, embeddings)
        print("took %.2f seconds" % (time.time() - start))

        init = tf.global_variables_initializer()
        saver = None

        with tf.Session() as session:
            session.run(init)
            best_accuracy, best_f1 = model.fit(session, saver, train, dev)


    # accuracy_results = []
    # f1_results = []

    # for hs in (300, 350):

    #     print("hidden size is %d" % hs)
    #     config.hidden_size = hs

    #     print("Preparing data...")
    #     helper, train, dev, test = load_and_preprocess_data(DATA_PATH, DATA_SPLIT_INDICES_PATH, TOKENS_TO_GLOVEID_PATH, config.max_length)
    #     # helper.max_length = config.max_length

    #     print("Load embeddings...")
    #     embeddings = np.load(GLOVE_VECTORS_PATH, mmap_mode='r')
    #     config.embed_size = embeddings.shape[1]

    #     # append unknown word and padding word vectors
    #     helper.add_additional_embeddings(embeddings)

    #     with tf.Graph().as_default():
    #         print("Building model...")
    #         start = time.time()
    #         model = SimilarityModel(helper, config, embeddings)
    #         print("took %.2f seconds" % (time.time() - start))

    #         init = tf.global_variables_initializer()
    #         saver = None

    #         with tf.Session() as session:
    #             session.run(init)
    #             best_accuracy, best_f1 = model.fit(session, saver, train, dev)

    #         accuracy_results.append((hs, best_accuracy))
    #         f1_results.append((hs, best_f1))

    #         print("best accuracy: %f, f1: %f" % (best_accuracy, best_f1))

    # print("accuracy results:")
    # print(accuracy_results)
    # print("f1 results:")
    # print(f1_results)
