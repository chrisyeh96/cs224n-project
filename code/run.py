from __future__ import print_function

import tensorflow as tf
import numpy as np
import sys, os, time, pickle
from similarity_model import SimilarityModel
import argparse
import csv

DATA_PATH = "../data/quora/tokenized_data.tsv"
DATA_SPLIT_INDICES_PATH = "../data/quora/data_split_indices.npz"
# DATA_SPLIT_INDICES_PATH = "../data/quora/data_split_indices_ibm.npz"
GLOVE_VECTORS_PATH = "../data/glove/glove.6B.300d.npy"
TOKENS_TO_GLOVEID_PATH = "../data/glove/glove.6B.300d.pkl"
MAX_LENGTH_PATH = "../data/quora/max_length.pkl"

JACCARD_SIMILARITY_THRESH = 0.1

class Config:
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    augment_data = False
    save_params = False
    update_embeddings = True

    # each word just indexes into glove vectors
    dropout = 0.5
    embed_size = 300 # word vector dimensions
    output_size = 50
    n_epochs = 30
    max_grad_norm = 10.
    lr = 0.001
    n_classes = 2

    # parameters that can be set from the command-line:
    hidden_size = 250
    batch_size = 1024
    max_length = 30
    distance_measure = "concat_steroids" # one of ["l2", "cosine", "custom_coef", "concat", "concat_steroids"]
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

def load_and_preprocess_data(data_path, data_split_indices_path, tokens_to_gloveID_path, max_length, augment_data=False):
    """
    Reads the training and dev data sets from the given paths.
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

    if augment_data:
        print("Augmenting data...")
        helper.augment_data(train_data)

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

    def augment_data(self, data):
        num_examples = len(data)
        # augment with 50% more negative training examples
        rand_rows = np.random.randint(0, high=num_examples, size=(num_examples,2))
        rand_cols = np.random.randint(0, high=2, size=(num_examples,2))
        neg_count = 0
        order = np.arange(num_examples)
        np.random.shuffle(order)
        for i in order:
            if rand_rows[i,0] == rand_rows[i,1]:
                continue

            q1 = data[rand_rows[i,0]][rand_cols[i,0]]
            q2 = data[rand_rows[i,1]][rand_cols[i,1]]
            if q1 == q2:
                continue

            if self.jaccard_similarity(q1,q2) < JACCARD_SIMILARITY_THRESH:
                continue

            data.append((q1,q2,0))
            neg_count += 1
            if neg_count == num_examples/2:
                break
        print("Added %d negative examples to the training set" % neg_count)

        # augment with 25% more positive (flipped duplicate) training examples
        flipped_count = 0
        order = np.arange(num_examples)
        np.random.shuffle(order)
        for i in order:
            if data[i][2] == 1:
                q1 = data[i][0]
                q2 = data[i][1]
                data.append((q2,q1,1))
                flipped_count += 1

            if flipped_count == num_examples/4:
                break
        print("Added %d positive (flipped duplicate) examples to the training set" % flipped_count)

        # augment with 25% more positive (exact duplicate) training examples
        rand_rows = np.random.randint(0, high=num_examples, size=(num_examples,))
        rand_cols = np.random.randint(0, high=2, size=(num_examples,))
        exact_count = 0
        order = np.arange(num_examples)
        np.random.shuffle(order)
        for i in order:
            if data[rand_rows[i]][2] == 1:
                continue

            q = data[rand_rows[i]][rand_cols[i]]
            data.append((q,q,1))
            exact_count += 1
            if exact_count == num_examples/4:
                break

        print("Added %d positive (exact duplicate) examples to the training set" % exact_count)

    def jaccard_similarity(self,x,y):
        intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
        union_cardinality = len(set.union(*[set(x), set(y)]))
        return intersection_cardinality/float(union_cardinality)

    @classmethod
    def load(cls, tokens_to_gloveID_path, max_length):
        # Make sure the directory exists.
        assert os.path.exists(tokens_to_gloveID_path)
        with open(tokens_to_gloveID_path, 'rb') as f:
            tok2id = pickle.load(f)
        # with open(max_length_path, 'rb') as f:
            # max_length = pickle.load(f)
        return cls(tok2id, max_length)

def print_options(args, config):
    print("Running with options:")
    for key, value in vars(args).iteritems():
        print("\t%s: %s" % (key, value if value is not None else getattr(config, str(key))))

if __name__ == "__main__":
    description = "Run the similarity_model"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-a", "--augment_data", action="store_true", help="augment data with negative and positive samples")
    parser.add_argument("-b", "--batch_size", type=int, required=False, help="number of examples for each minibatch")
    parser.add_argument("-c", "--cell", required=False, choices=["rnn", "gru"], help="model cell type")
    parser.add_argument("-d", "--distance_measure", required=False, choices=["l2", "cosine", "custom_coef", "concat", "concat_steroids"], help="distance measure")
    parser.add_argument("-r", "--regularization_constant", type=float, required=False, help="regularization constant")
    parser.add_argument("-hs", "--hidden_size", type=int, required=False, help="neural net hidden size")
    parser.add_argument("-ml", "--max_length", type=int, required=False, help="maximum length of sentences")
    parser.add_argument("-s", "--save_params", action="store_true", help="save trained variables to a checkpoint file")
    args = parser.parse_args()

    config = Config()
    config.augment_data = args.augment_data
    config.save_params = args.save_params
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.cell is not None:
        config.cell = args.cell
    if args.distance_measure is not None:
        config.distance_measure = args.distance_measure
    if args.regularization_constant is not None:
        config.regularization_constant = args.regularization_constant
    if args.hidden_size is not None:
        config.hidden_size = args.hidden_size
    if args.max_length is not None:
        config.max_length = args.max_length


    print("Preparing data...")
    helper, train, dev, test = load_and_preprocess_data(DATA_PATH, DATA_SPLIT_INDICES_PATH, TOKENS_TO_GLOVEID_PATH, config.max_length, config.augment_data)

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

        print_options(args, config)

        init = tf.global_variables_initializer()
        saver = None
        if config.save_params:
            saver = tf.train.Saver()
        
        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True
        # sess_config.gpu_options.per_process_gpu_memory_fraction = 0.01

        # start a TensorFlow session, initialize all variables, then run model
        with tf.Session(config=sess_config) as session:
            session.run(init)
            best_dev_accuracy, dev_f1, test_accuracy, test_f1 = model.fit(session, saver, train, dev, test)
            print("best dev accuracy: %f, dev f1: %f, test accuracy: %f, test f1: %f" % (best_dev_accuracy, dev_f1, test_accuracy, test_f1))

    with open("../results/model_results.csv", 'a') as f:
        fieldnames = ["cell", "distance_measure", "augment_data", "regularization_constant", "hidden_size", \
            "max_length", "best_dev_accuracy", "dev_f1", "test_accuracy", "test_f1"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        hyperparams_and_results_dict = {
            "cell": config.cell,
            "distance_measure": config.distance_measure,
            "augment_data": config.augment_data,
            "regularization_constant": config.regularization_constant,
            "hidden_size": config.hidden_size,
            "max_length": config.max_length,
            "best_dev_accuracy": best_dev_accuracy,
            "dev_f1": dev_f1,
            "test_accuracy": test_accuracy,
            "test_f1": test_f1
        }
        writer.writerow(hyperparams_and_results_dict)


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
