import tensorflow as tf
import numpy as np
import sys, os, pickle
from glove import loadWordVectors
from similarity_model import SimilarityModel
import time

TRAIN_DATA_PATH = "../data/quora/train.tsv"
TEST_DATA_PATH = "../data/quora/test.tsv"
DEPENDENCIES_PATH = "dependencies"

class Config:
	"""Holds model hyperparams and data information.

	The config class is used to store various hyperparameters and dataset
	information parameters. Model objects are passed a Config() object at
	instantiation.
	"""
	# each word just indexes into glove vectors
	n_features = 1
	n_classes = 2
	dropout = 0.5
	# word vector dimensions
	embed_size = 50
	hidden_size = 300
	batch_size = 32
	n_epochs = 10
	max_grad_norm = 10.
	lr = 0.001

def normalize(word):
	"""
	Normalize words that are numbers or have casing.
	"""
	if word.isdigit(): return NUM
	else: return word.lower()

def load_embeddings(helper):
	embeddings = loadWordVectors(helper.tok2id)
	print "Initialized embeddings"
	return embeddings

def read_datafile(fstream):
	"""
	Reads a input stream @fstream (e.g. output of `open(fname, 'r')`) in CoNLL file format.
	@returns a list of examples [(tokens), (labels)]. @tokens and @labels are lists of string.
	"""
	ret = []
	sentence1 = sentence2 = label = None

	for line_num, line in enumerate(fstream):
		line = line.strip()
		if line_num % 3 == 0:
			sentence1 = line.split("\t")
		elif line_num % 3 == 1:
			sentence2 = line.split("\t")
		else:
			label = int(line)
			ret.append(([sentence1, sentence2], label))

	return ret

def load_and_preprocess_data(train_file, dev_file, deps_path):
	print "Loading training data..."
	with open(train_file) as tf:
	    train = read_datafile(tf)
	print "Done. Read %d sentences" % len(train)
	print "Loading dev data..."
	with open(dev_file) as df:
	    dev = read_datafile(df)
	print "Done. Read %d sentences"% len(dev)

	# helper = ModelHelper.build(train)
	helper = ModelHelper.load(deps_path)

	# now process all the input data.
	# turn words into the glove indices
	train_data = helper.vectorize(train)
	dev_data = helper.vectorize(dev)

	return helper, train_data, dev_data, train, dev

class ModelHelper(object):
	"""
	This helper takes care of preprocessing data, constructing embeddings, etc.
	"""
	def __init__(self, tok2id, max_length):
	    self.tok2id = tok2id
	    self.max_length = max_length

	def vectorize_example(self, sentences, label):
	    sentence1 = [self.tok2id.get(word, self.tok2id["UNK"]) for word in sentences[0]]
	    sentence2 = [self.tok2id.get(word, self.tok2id["UNK"]) for word in sentences[1]]

	    return [sentence1, sentence2], label

	def vectorize(self, data):
	    return [self.vectorize_example(sentences, labels) for sentences, labels in data]

	@classmethod
	def load(cls, path):
	    # Make sure the directory exists.
	    assert os.path.exists(path) and os.path.exists(os.path.join(path, "features.pkl"))
	    # Save the tok2id map.
	    with open(os.path.join(path, "features.pkl")) as f:
	        tok2id, max_length = pickle.load(f)
	    return cls(tok2id, max_length)


if __name__ == "__main__":
	print "Preparing data..."
	config = Config()
	helper, train, dev, train_raw, dev_raw = load_and_preprocess_data(TRAIN_DATA_PATH, TEST_DATA_PATH, DEPENDENCIES_PATH)
	print train_raw[0]
	print dev_raw[0]
	print train[0]
	print dev[0]
	embeddings = load_embeddings(helper)

	config.embed_size = embeddings.shape[1]

	print config.embed_size

	print "Building model..."
	start = time.time()
	model = SimilarityModel(helper, config, embeddings)
	print "took %.2f seconds" % (time.time() - start)

	# with tf.Graph().as_default():
	#     logger.info("Building model...",)
	#     start = time.time()
	#     model = RNNModel(helper, config, embeddings)
	#     logger.info("took %.2f seconds", time.time() - start)

	#     init = tf.global_variables_initializer()
	#     saver = None

	#     with tf.Session() as session:
	#         session.run(init)
	#         model.fit(session, saver, train, dev)

	# logger.info("Model did not crash!")
	# logger.info("Passed!")