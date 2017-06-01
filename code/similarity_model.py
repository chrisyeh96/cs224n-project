from __future__ import print_function

import tensorflow as tf
from model import Model
# from rnn_cell import RNNCell
from gru_cell import GRUCell
from util import Progbar, cosine_distance, norm
import numpy as np
import os
import pdb
import pickle
import csv

class SimilarityModel(Model):
    def __init__(self, helper, config, embeddings, report=None):
        self.helper = helper
        self.config = config
        self.pretrained_embeddings = embeddings
        self.report = report

        self.input_placeholder1 = None
        self.input_placeholder2 = None
        self.labels_placeholder = None
        self.dropout_placeholder = None

        self.build()

    def add_placeholders(self):
        """Generates placeholder variables to represent the input tensors

        These placeholders are used as inputs by the rest of the model building and will be fed
        data during training.  Note that when "None" is in a placeholder's shape, it's flexible
        (so we can use different batch sizes without rebuilding the model).

        Adds following nodes to the computational graph

        input_placeholder1: Input placeholder tensor of  shape (None, self.max_length), type tf.int32
        input_placeholder2: Input placeholder tensor of  shape (None, self.max_length), type tf.int32
        labels_placeholder: Labels placeholder tensor of shape (None, self.max_length), type tf.int32
        dropout_placeholder: Dropout value placeholder (scalar), type tf.float32

        TODO: Add these placeholders to self as the instance variables
            self.input_placeholder
            self.labels_placeholder
            self.mask_placeholder
            self.dropout_placeholder

        HINTS:
            - Remember to use self.max_length NOT Config.max_length

        (Don't change the variable names)
        """
        ### YOUR CODE HERE (~4-6 lines)
        self.input_placeholder1 = tf.placeholder(tf.int32, (None, self.helper.max_length))
        self.input_placeholder2 = tf.placeholder(tf.int32, (None, self.helper.max_length))

        self.labels_placeholder = tf.placeholder(tf.int32, (None,))
        self.dropout_placeholder = tf.placeholder(tf.float32)
        ### END YOUR CODE

    def create_feed_dict(self, inputs_batch1, inputs_batch2, labels_batch=None, dropout=1):
        """Creates the feed_dict for the dependency parser.

        A feed_dict takes the form of:

        feed_dict = {
                <placeholder>: <tensor of values to be passed for placeholder>,
                ....
        }

        Hint: The keys for the feed_dict should be a subset of the placeholder
                    tensors created in add_placeholders.
        Hint: When an argument is None, don't add it to the feed_dict.

        Args:
            inputs_batch: A batch of input data.
            mask_batch:   A batch of mask data.
            labels_batch: A batch of label data.
            dropout: The dropout rate.
        Returns:
            feed_dict: The feed dictionary mapping from placeholders to values.
        """
        ### YOUR CODE (~6-10 lines)
        feed_dict = {}

        feed_dict[self.input_placeholder1] = inputs_batch1
        feed_dict[self.input_placeholder2] = inputs_batch2

        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch
        if dropout is not None:
            feed_dict[self.dropout_placeholder] = dropout
        ### END YOUR CODE
        return feed_dict

    def add_embedding(self):
        """Adds an embedding layer that maps from input tokens (integers) to vectors and then
        concatenates those vectors:

        TODO:
            - Create an embedding tensor and initialize it with self.pretrained_embeddings.
            - Use the input_placeholder to index into the embeddings tensor, resulting in a
              tensor of shape (None, max_length, embed_size).
            - Concatenates the embeddings by reshaping the embeddings tensor to shape
              (None, max_length, embed_size).

        HINTS:
            - You might find tf.nn.embedding_lookup useful.
            - You can use tf.reshape to concatenate the vectors. See
              following link to understand what -1 in a shape means.
              https://www.tensorflow.org/api_docs/python/array_ops/shapes_and_shaping#reshape.

        Returns:
            embeddings: tf.Tensor of shape (None, max_length, embed_size)
        """
        # treat all word vectors as variables that we can update
        if self.config.update_embeddings:
            embeddings = tf.Variable(np.concatenate([self.pretrained_embeddings, self.helper.additional_embeddings]))

        # alternatively, only update the additional_embeddings (unknown / padding words)
        else:
            glove_vectors = tf.constant(self.pretrained_embeddings)
            additional_embeddings = tf.Variable(self.helper.additional_embeddings)
            embeddings = tf.concat(0, [glove_vectors, additional_embeddings])

        # look up values of input indices from pretrained embeddings
        # embeddings1 and embeddings2 will have shape (num_examples, max_length, embed_size)
        embeddings1 = tf.nn.embedding_lookup(embeddings, self.input_placeholder1)
        embeddings2 = tf.nn.embedding_lookup(embeddings, self.input_placeholder2)
        return embeddings1, embeddings2

    def add_prediction_op(self):
        """Adds the unrolled RNN:
            h_0 = 0
            for t in 1 to T:
                o_t, h_t = cell(x_t, h_{t-1})
                o_drop_t = Dropout(o_t, dropout_rate)
                y_t = o_drop_t U + b_2

        TODO: There a quite a few things you'll need to do in this function:
            - Define the variables U, b_2.
            - Define the vector h as a constant and inititalize it with
              zeros. See tf.zeros and tf.shape for information on how
              to initialize this variable to be of the right shape.
              https://www.tensorflow.org/api_docs/python/constant_op/constant_value_tensors#zeros
              https://www.tensorflow.org/api_docs/python/array_ops/shapes_and_shaping#shape
            - In a for loop, begin to unroll the RNN sequence. Collect
              the predictions in a list.
            - When unrolling the loop, from the second iteration
              onwards, you will HAVE to call
              tf.get_variable_scope().reuse_variables() so that you do
              not create new variables in the RNN cell.
              See https://www.tensorflow.org/versions/master/how_tos/variable_scope/
            - Concatenate and reshape the predictions into a predictions
              tensor.
        Hint: You will find the function tf.pack (similar to np.asarray)
              useful to assemble a list of tensors into a larger tensor.
              https://www.tensorflow.org/api_docs/python/array_ops/slicing_and_joining#pack
        Hint: You will find the function tf.transpose and the perms
              argument useful to shuffle the indices of the tensor.
              https://www.tensorflow.org/api_docs/python/array_ops/slicing_and_joining#transpose

        Remember:
            * Use the xavier initilization for matrices.
            * Note that tf.nn.dropout takes the keep probability (1 - p_drop) as an argument.
            The keep probability should be set to the value of self.dropout_placeholder

        Returns:
            pred: tf.Tensor of shape (batch_size, max_length, n_classes)
        """
        x1, x2 = self.add_embedding()
        dropout_rate = self.dropout_placeholder

        # choose cell type
        if self.config.cell == "rnn":
            cell = RNNCell(self.config.embed_size, self.config.hidden_size)
        elif self.config.cell == "gru":
            cell = GRUCell(self.config.embed_size, self.config.hidden_size)
        else:
            raise ValueError("Unsuppported cell type: " + self.config.cell)

        # Initialize hidden states to zero vectors of shape (num_examples, hidden_size)
        h1 = tf.zeros((tf.shape(x1)[0], self.config.hidden_size), tf.float32)
        h2 = tf.zeros((tf.shape(x2)[0], self.config.hidden_size), tf.float32)

        with tf.variable_scope("RNN") as scope:
            for time_step in range(self.helper.max_length):
                if time_step > 0:
                    scope.reuse_variables()

                o1_t, h1 = cell(x1[:, time_step, :], h1, scope)

                if time_step == 0:
                    scope.reuse_variables()

                o2_t, h2 = cell(x2[:, time_step, :], h2, scope)

        # h_drop1 = tf.nn.dropout(h1, dropout_rate)
        # h_drop2 = tf.nn.dropout(h2, dropout_rate)        
        
        # use L2-regularization: sum of squares of all parameters

        if self.config.distance_measure == "l2":
            # perform logistic regression on l2-distance between h1 and h2
            distance = norm(h1 - h2 + 0.000001)
            logistic_a = tf.Variable(0.0, dtype=tf.float32, name="logistic_a")
            logistic_b = tf.Variable(0.0, dtype=tf.float32, name="logistic_b")
            self.regularization_term = tf.square(logistic_a) + tf.square(logistic_b)
            preds = tf.sigmoid(logistic_a * distance + logistic_b)

        elif self.config.distance_measure == "cosine":
            # perform logistic regression on cosine distance between h1 and h2
            distance = cosine_distance(h1 + 0.000001, h2 + 0.000001)
            logistic_a = tf.Variable(1.0, dtype=tf.float32, name="logistic_a")
            logistic_b = tf.Variable(0.0, dtype=tf.float32, name="logistic_b")
            self.regularization_term = tf.square(logistic_a) + tf.square(logistic_b)
            preds = tf.sigmoid(logistic_a * distance + logistic_b)

        elif self.config.distance_measure == "custom_coef":
            # perform logistic regression on the vector |h1-h2|,
            # equivalent to logistic regression on the (scalar) weighted Manhattan distance between h1 and h2,
            # ie. weighted sum of |h1-h2|
            logistic_a = tf.get_variable("coef", [self.config.hidden_size], tf.float32, tf.contrib.layers.xavier_initializer())
            logistic_b = tf.Variable(0.0, dtype=tf.float32, name="logistic_b")
            self.regularization_term = tf.reduce_sum(tf.square(logistic_a)) + tf.square(logistic_b)
            preds = tf.sigmoid(tf.reduce_sum(logistic_a * tf.abs(h1 - h2), axis=1) + logistic_b)

        elif self.config.distance_measure == "concat":
            # use softmax for prediction
            U = tf.get_variable("U", (4 * self.config.hidden_size, self.config.n_classes), tf.float32, tf.contrib.layers.xavier_initializer())
            b = tf.get_variable("b", (self.config.n_classes,), tf.float32, tf.constant_initializer(0))
            v = tf.nn.relu(tf.concat(1, [h1, h2, tf.square(h1 - h2), h1 * h2]))
            self.regularization_term = tf.reduce_sum(tf.square(U)) + tf.reduce_sum(tf.square(b))
            preds = tf.matmul(v, U) + b

        elif self.config.distance_measure == "concat_steroids":
            # use softmax for prediction
            W1 = tf.get_variable("W1", (4 * self.config.hidden_size, self.config.hidden_size), tf.float32, tf.contrib.layers.xavier_initializer())
            b1 = tf.get_variable("b1", (self.config.hidden_size,), tf.float32, tf.constant_initializer(0))

            W2 = tf.get_variable("W2", (self.config.hidden_size, self.config.n_classes), tf.float32, tf.contrib.layers.xavier_initializer())
            b2 = tf.get_variable("b2", (self.config.n_classes,), tf.float32, tf.constant_initializer(0))

            v1 = tf.nn.relu(tf.concat(1, [h1, h2, tf.square(h1 - h2), h1 * h2]))
            v2 = tf.nn.relu(tf.matmul(v1, W1) + b1)

            self.regularization_term = tf.reduce_sum(tf.square(W1)) + tf.reduce_sum(tf.square(b1)) + tf.reduce_sum(tf.square(W2)) + tf.reduce_sum(tf.square(b2))
            preds = tf.matmul(v2, W2) + b2

        else:
            raise ValueError("Unsuppported distance type: " + self.config.distance_measure)
        
        return preds

    def add_loss_op(self, preds):
        """Adds Ops for the loss function to the computational graph.

        Args:
            preds: A tensor of shape (batch_size,) containing the output of the neural network
        Returns:
            loss: A 0-d tensor (scalar)
        """
        if self.config.distance_measure == "concat" or self.config.distance_measure == "concat_steroids": # Concatenated model
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(preds, self.labels_placeholder))
        else: # BASE MODELS
            loss = tf.reduce_mean(tf.square(preds - tf.to_float(self.labels_placeholder)))

        # add regularization term
        loss += self.config.regularization_constant * self.regularization_term
        return loss 

    def add_training_op(self, loss):
        """Sets up the training Ops.

        Creates an optimizer and applies the gradients to all trainable variables.
        The Op returned by this function is what must be passed to the
        `sess.run()` call to cause the model to train. See

        https://www.tensorflow.org/versions/r0.7/api_docs/python/train.html#Optimizer

        for more information.

        Use tf.train.AdamOptimizer for this model.
        Calling optimizer.minimize() will return a train_op object.

        Args:
            loss: Loss tensor, from cross_entropy_loss.
        Returns:
            train_op: The Op for training.
        """
        self.train_op = tf.train.AdamOptimizer(self.config.lr).minimize(loss)
        return self.train_op

    # rounds predictions to 0, 1
    def predict_on_batch(self, sess, inputs_batch1, inputs_batch2):
        feed = self.create_feed_dict(inputs_batch1, inputs_batch2)
        predictions = sess.run(self.pred, feed_dict=feed) # should return a list of 0s and 1s

        if self.config.distance_measure in ["concat", "concat_steroids"]:
            # predictions = array of size (num_examples, 2)
            # is the input into the softmax, but we just care about comparing the two values
            return (predictions[:, 1] > predictions[:, 0]).astype(int)
        else:
            # predictions = scalar output of logistic regression
            # => we need to round to nearest int (either 0 or 1)
            return np.round(predictions).astype(int)

    # evaluate model after training
    def evaluate(self, sess, examples):
        """
        Args:
            sess: a TFSession
            examples: [ numpy array (num_examples, max_length) of all sentence 1,
                        numpy array (num_examples, max_length) of all sentence 2,
                        numpy array (num_examples, ) of all labels ]
        Returns:
            fraction of correct predictions
            TODO: maybe return the actual predictions as well
        """
        correct_preds = 0.0
        tp = 0.0
        fp = 0.0
        fn = 0.0

        preds = []
        confusion_matrix = np.zeros((2,2), dtype=np.float64)

        num_examples = len(examples[0])
        num_batches = int(np.ceil(num_examples * 1.0 / self.config.batch_size))
        prog = Progbar(target=num_batches)

        for i, batch in enumerate(self.minibatch(examples, shuffle=False)):
            # Ignore labels
            sentence1_batch, sentence2_batch, labels_batch = batch
            preds_ = self.predict_on_batch(sess, sentence1_batch, sentence2_batch)
            preds += list(preds_)
            labels_batch = np.array(labels_batch)

            for j in range(preds_.shape[0]):
                confusion_matrix[labels_batch[j], preds_[j]] += 1

            prog.update(i+1)

        ## CONFUSION MATRIX (is indeed hella confusing)
        #            pred -   pred +
        # label -  |   tn   |   fp   |
        # label +  |   fn   |   tp   |
        tn = confusion_matrix[0,0]
        fp = confusion_matrix[0,1]
        fn = confusion_matrix[1,0]
        tp = confusion_matrix[1,1]
        correct_preds = tp + tn
        accuracy = correct_preds / num_examples
        precision = (tp)/(tp + fp) if tp > 0  else 0
        recall = (tp)/(tp + fn) if tp > 0  else 0
        
        print("\ntp: %f, fp: %f, fn: %f" % (tp, fp, fn))
        f1 = 2 * precision * recall / (precision + recall) if tp > 0  else 0

        return (preds, accuracy, precision, recall, f1)

    def minibatch(self, examples, shuffle=True):
        """
        Args:
            examples: [ numpy array (num_examples, max_length) of all sentence 1,
                        numpy array (num_examples, max_length) of all sentence 2,
                        numpy array (num_examples, ) of all labels ]
            batch_size: int
            shuffle: bool, whether or not to shuffle the examples before creating batches
        Yields: (sentence1_batch, sentence2_batch, labels_batch)
            sentence1_batch: numpy array with shape (batch_size, max_length)
            sentence2_batch: same idea as sentence1_batch
            labels_batch: (batch_size,) numpy array of labels for the batch
        """
        sent1, sent2, labels = examples
        num_examples = len(sent1)
        order = np.arange(num_examples)
        if shuffle:
            np.random.shuffle(order)

        batch_size = self.config.batch_size
        num_batches = int(np.ceil(num_examples * 1.0 / batch_size))
        for i in range(num_batches):
            start = i * batch_size
            end = min(i * batch_size + batch_size, num_examples)
            yield (sent1[order[start:end]], sent2[order[start:end]], labels[order[start:end]])

    def run_epoch(self, sess, train_examples, dev_set, test_set):
        """
        Args:
            sess: TFSession
            train_examples: [ numpy array (num_examples, max_length) of all sentence 1,
                        numpy array (num_examples, max_length) of all sentence 2,
                        numpy array (num_examples, ) of all labels ]
            dev_set: same as train_examples, except for the dev set
        Returns:
            avg loss across all minibatches
        """
        num_examples = len(train_examples[0])
        num_batches = int(np.ceil(num_examples * 1.0 / self.config.batch_size))
        prog = Progbar(target=num_batches)

        total_loss = 0.0
        
        for i, batch in enumerate(self.minibatch(train_examples, shuffle=True)):
            sentence1_batch, sentence2_batch, labels_batch = batch
            feed = self.create_feed_dict(sentence1_batch, sentence2_batch, labels_batch, dropout=self.config.dropout)
            _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
            total_loss += loss
            prog.update(i+1, [("train loss", loss)])
        print("")
        return total_loss / num_batches

    def preprocess_sequence_data(self, examples):
        """
        Args:
            examples: is list of tuples:
                [ 
                  (numpy array of sentence 1, numpy array of sentence2, int label),
                  ...
                ]
        Returns: (all_sent1, all_sent2, all_labels)
            all_sent1: numpy array of shape (num_examples, max_length)
            all_sent2: same as all_sent1, except for the sentence2's
            all_labels: numpy arrray of all labels, has shape (num_examples,)
        """
        # examples 
        all_sent1, all_sent2, all_labels = zip(*examples)
        all_sent1 = np.stack(all_sent1)
        all_sent2 = np.stack(all_sent2)
        all_labels = np.array(all_labels)
        return (all_sent1, all_sent2, all_labels)

    def fit(self, sess, saver, train_examples_raw, dev_set_raw, test_set_raw):
        """
        Args:
            sess: TFSession
            saver: tf.train.Saver, used to saves all variables after finding best model
                set to None if you do not want to save the variables
            train_examples_raw: list of training examples, each example is a
                tuple (s1, s2, label) where s1,s2 are padded/truncated sentences
            dev_set_raw: same as train_examples_raw, except for the dev set
        Returns:
            best training loss over the self.config.n_epochs of training
        """
        # unpack data
        train_examples = self.preprocess_sequence_data(train_examples_raw)
        dev_set = self.preprocess_sequence_data(dev_set_raw)
        test_set = self.preprocess_sequence_data(test_set_raw)

        splits = {
            "train" : train_examples,
            "dev" : dev_set,
            "test" : test_set
        }

        results = {}
        for split in splits:
            num_examples = len(splits[split][0])
            results[split] = {
                "preds" : np.zeros((self.config.n_epochs, num_examples)),
                "accuracy" : np.zeros(self.config.n_epochs),
                "precision" : np.zeros(self.config.n_epochs),
                "recall" : np.zeros(self.config.n_epochs),
                "f1" : np.zeros(self.config.n_epochs)
            }

        best_dev_accuracy = 0
        best_dev_accuracy_epoch = 0

        for epoch in range(self.config.n_epochs):
            print("Epoch %d out of %d" % (epoch + 1, self.config.n_epochs))
            self.run_epoch(sess, train_examples, dev_set, test_set)

            for split in splits:
                preds, accuracy, precision, recall, f1 = self.evaluate(sess, splits[split])
                results[split]["preds"][epoch] = preds
                results[split]["accuracy"][epoch] = accuracy
                results[split]["precision"][epoch] = precision
                results[split]["recall"][epoch] = recall
                results[split]["f1"][epoch] = f1

                for score in ["accuracy", "precision", "recall", "f1"]:
                    print("%s %s: %f" % (split, score, results[split][score][epoch]))
                print("")

            if results["dev"]["accuracy"][epoch] > best_dev_accuracy:
                best_dev_accuracy = results["dev"]["accuracy"][epoch]
                best_dev_accuracy_epoch = epoch
                print("New best accuracy on dev set!!")

                if saver is not None:
                    checkpoint_dir = "../saved_ckpts/"
                    if not os.path.exists(checkpoint_dir):
                        os.makedirs(checkpoint_dir)
                    filename = "model_b_%d_c_%s_d_%s_r_%g_hs_%d_ml_%d.ckpt" % (self.config.batch_size,
                        self.config.cell, self.config.distance_measure, self.config.regularization_constant,
                        self.config.hidden_size, self.config.max_length)
                    save_path = saver.save(sess, os.path.join(checkpoint_dir, filename))
                    print("Model saved in file: %s" % save_path)

        # save results to pickle
        results_dir = "../results/"
        filename = "model_a_%d_c_%s_d_%s_r_%g_hs_%d_ml_%d.pkl" % (int(self.config.augment_data), self.config.cell,
            self.config.distance_measure, self.config.regularization_constant, self.config.hidden_size, self.config.max_length)
        save_path = os.path.join(results_dir, filename)
        with open(save_path, 'wb') as f:
            pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

        # calculate other relevant scores
        dev_accuracy_f1 = results["dev"]["f1"][best_dev_accuracy_epoch]
        test_accuracy = results["test"]["accuracy"][best_dev_accuracy_epoch]
        test_f1 = results["test"]["f1"][best_dev_accuracy_epoch]

        return best_dev_accuracy, dev_accuracy_f1, test_accuracy, test_f1

    def test_time_fit(self, sess, saver, train_examples_raw):
        """
        Args:
            sess: TFSession
            saver: tf.train.Saver, used to saves all variables after finding best model
                set to None if you do not want to save the variables
            train_examples_raw: list of training examples, each example is a
                tuple (s1, s2, label) where s1,s2 are padded/truncated sentences
            dev_set_raw: same as train_examples_raw, except for the dev set
        Returns:
            best training loss over the self.config.n_epochs of training
        """
        # unpack data
        train_examples = self.preprocess_sequence_data(train_examples_raw)

        for epoch in range(self.config.n_epochs):
            print("Epoch %d out of %d" % (epoch + 1, self.config.n_epochs))
            loss = self.run_epoch(sess, train_examples, None, None)

        return loss

    def test_time_predict(self, sess, test_examples_raw):
        test_examples = self.preprocess_sequence_data(test_examples_raw)

        num_examples = len(test_examples[0])
        num_batches = int(np.ceil(num_examples * 1.0 / self.config.batch_size))
        prog = Progbar(target=num_batches)

        for i, batch in enumerate(self.minibatch(test_examples, shuffle=False)):
            # Ignore labels
            sentence1_batch, sentence2_batch, labels_batch = batch
            preds_ = self.predict_on_batch(sess, sentence1_batch, sentence2_batch)
            preds += list(preds_)

            prog.update(i+1)

        # here we have a list of predictions
        with open('../submission.csv', 'wb') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['test_id', 'is_duplicate'])
            for i in xrange(len(preds)):
                writer.writerow([str(i), preds[i]])
            print("Generated new submission.csv")
