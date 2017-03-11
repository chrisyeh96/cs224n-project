import tensorflow as tf
from model import Model
from rnn_cell import RNNCell
from util import Progbar, minibatches

def pad_sequences(data, max_length, padding_word_index):
    """Ensures each sentence in @data is of length @max_length by padding it with
    @padding_word_index at the beginning of the sentence or by truncating the rest
    of the sequence.

    Args:
        data: is a list of ([sentence1, sentence2], label) tuples.
            - sentence1, sentence2 are lists of integer indices representing words
            - label is a boolean
        max_length: the desired length for all sentences
    Returns:
        a new list of data points of the structure ([sentence1', sentence2'], label)
        where each of sentence1' and sentence2' are of length @max_length.
    """
    ret = []

    # Use this zero vector when padding sequences.
    for sentences, label in data:
        ### YOUR CODE HERE (~4-6 lines)
        new_sentences = np.zeros((2, max_length), dtype=np.int32)

        for i in [0,1]:
            sentence = sentences[i]
            initial_length = len(sentence)
            if initial_length < max_length:
                num_padding = max_length - initial_length
                new_sentences[i] = [padding_word_index]*num_padding + sentence
            elif initial_length >= max_length:
                new_sentences[i] = sentence[0:max_length]

        ret.append((new_sentences, label))
        ### END YOUR CODE ###
    return ret


class SimilarityModel(Model):
    def __init__(self, helper, config, embeddings, report=None):
        self.helper = helper
        self.config = config
        self.pretrained_embeddings = embeddings
        self.report = report

    def add_placeholders(self):
        """Generates placeholder variables to represent the input tensors

        These placeholders are used as inputs by the rest of the model building and will be fed
        data during training.  Note that when "None" is in a placeholder's shape, it's flexible
        (so we can use different batch sizes without rebuilding the model).

        Adds following nodes to the computational graph

        input_placeholder: Input placeholder tensor of  shape (None, self.max_length, n_features), type tf.int32
        labels_placeholder: Labels placeholder tensor of shape (None, self.max_length), type tf.int32
        mask_placeholder:  Mask placeholder tensor of shape (None, self.max_length), type tf.bool
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
        self.input_placeholder1 = tf.placeholder(tf.int32, (None, self.max_length, self.config.n_features))
        self.input_placeholder2 = tf.placeholder(tf.int32, (None, self.max_length, self.config.n_features))

        self.labels_placeholder = tf.placeholder(tf.int32, (None, self.max_length))
        self.mask_placeholder = tf.placeholder(tf.bool, (None, self.max_length))
        self.dropout_placeholder = tf.placeholder(tf.float32)
        ### END YOUR CODE

    def create_feed_dict(self, inputs_batch1, inputs_batch2, mask_batch, labels_batch=None, dropout=1):
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

        if mask_batch is not None:
            feed_dict[self.mask_placeholder] = mask_batch
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
              tensor of shape (None, max_length, n_features, embed_size).
            - Concatenates the embeddings by reshaping the embeddings tensor to shape
              (None, max_length, n_features * embed_size).

        HINTS:
            - You might find tf.nn.embedding_lookup useful.
            - You can use tf.reshape to concatenate the vectors. See
              following link to understand what -1 in a shape means.
              https://www.tensorflow.org/api_docs/python/array_ops/shapes_and_shaping#reshape.

        Returns:
            embeddings: tf.Tensor of shape (None, max_length, n_features*embed_size)
        """
        ### YOUR CODE HERE (~4-6 lines)
        embeddings = tf.Variable(self.pretrained_embeddings)
        # look up values of input indeces from pretrained embeddings
        embeddings1 = tf.nn.embedding_lookup([embeddings, self.helper.additional_embeddings], self.input_placeholder1)
        embeddings2 = tf.nn.embedding_lookup([embeddings, self.helper.additional_embeddings], self.input_placeholder2)
        # reshape the embeddings
        embeddings1 = tf.reshape(embeddings1, (-1, self.max_length, self.config.n_features * self.config.embed_size)) 
        embeddings2 = tf.reshape(embeddings2, (-1, self.max_length, self.config.n_features * self.config.embed_size)) 
        ### END YOUR CODE
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

        preds = 0 # Predicted total output

        # Use the cell defined below. For Q2, we will just be using the
        # RNNCell you defined, but for Q3, we will run this code again
        # with a GRU cell!
        # if self.config.cell == "rnn":
        cell = RNNCell(Config.n_features * Config.embed_size, Config.hidden_size)
        # elif self.config.cell == "gru":
            # cell = GRUCell(Config.n_features * Config.embed_size, Config.hidden_size)
        # else:
            # raise ValueError("Unsuppported cell type: " + self.config.cell)

        # Define U and b2 as variables.
        # Initialize state as vector of zeros.
        ### YOUR CODE HERE (~4-6 lines)
        U = tf.get_variable("U", (self.config.hidden_size, self.config.n_classes), tf.float32, tf.contrib.layers.xavier_initializer())
        b_2 = tf.get_variable("b_2", (self.config.n_classes,), tf.float32, tf.constant_initializer(0))

        h1 = tf.zeros((tf.shape(x1)[0], self.config.hidden_size), tf.float32)
        h2 = tf.zeros((tf.shape(x2)[0], self.config.hidden_size), tf.float32)
        ### END YOUR CODE
        o1_t, o2_t = 0, 0

        with tf.variable_scope("RNN") as scope:
            for time_step in range(self.max_length):
                ### YOUR CODE HERE (~6-10 lines)
                if time_step > 0:
                    scope.reuse_variables()

                o1_t, h1_t = cell(x1[:, time_step, :], tf.pack(h1), scope)
                o2_t, h2_t = cell(x2[:, time_step, :], tf.pack(h2), scope)
                # h[time_step] = h_t
                h1 = h1_t
                h2 = h2_t
                ### END YOUR CODE

        # Make sure to reshape @preds here.
        ### YOUR CODE HERE (~2-4 lines) 
        preds = tf.reduce_sum(tf.mul(h1, h2), axis=1) / tf.norm(h1, axis=1) / tf.norm(h2, axis=1)
        # preds = tf.transpose(preds)
        ### END YOUR CODE

        return preds

    def add_loss_op(self, preds):
        """Adds Ops for the loss function to the computational graph.

        Args:
            pred: A tensor of shape (batch_size, n_classes) containing the output of the neural
                  network before the softmax layer.
        Returns:
            loss: A 0-d tensor (scalar)
        """
        ### YOUR CODE HERE (~2-4 lines)
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(preds, self.labels_placeholder))
        ### END YOUR CODE
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
        ### YOUR CODE HERE (~1-2 lines)
        train_op = tf.train.AdamOptimizer(self.config.lr).minimize(loss)
        ### END YOUR CODE
        return train_op

    def train_on_batch(self, sess, inputs_batch, labels_batch):
        # split up inputs into inputs 1 and inputs 2
        print("Train on batch:")
        print("inputs_batch", inputs_batch)
        print("inputs_batch[0]", inputs_batch[0])
        print("labels_batch", labels_batch)

    def run_epoch(self, sess, train_examples, dev_set, train_examples_raw, dev_set_raw):
        prog = Progbar(target = 1 + int(len(train_examples) / self.config.batch_size))
        
        for i, batch in enumerate(minibatches(train_examples, self.config.batch_size)):
            loss = self.train_on_batch(sess, *batch)
            prog.update(i+1, [("train loss", loss)])
        print("")

    def preprocess_sequence_data(self, examples):
        return pad_sequences(examples, self.helper.max_length, self.helper.PADDING_WORD_INDEX)


    def fit(self, sess, saver, train_examples_raw, dev_set_raw):
        best_score = 0.

        # add padding
        train_examples = self.preprocess_sequence_data(train_examples_raw)
        dev_set = self.preprocess_sequence_data(dev_set_raw)

        for epoch in range(self.config.n_epochs):
            print("Epoch %d out of %d", epoch + 1, self.config.n_epochs)
            score = self.run_epoch(sess, train_examples, dev_set, train_examples_raw, dev_set_raw)

