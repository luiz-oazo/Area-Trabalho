import argparse
import numpy as np
import os
import pickle
import random
import tensorflow as tf
import time
import util
from collections import Counter
#from tensorflow.keras.optimizers.legacy import Adam

from datareader import DataReader
from tensorflow.keras import Model, layers


class Config:
    def __init__(self, vocab_size=50000, batch_size=140, embed_size=128, skip_window=1,
                 num_skips=2, num_sampled=64, lr=1.0, std_param=0.01,
                 init_param=(1.0, 1.0), num_steps=100001, show_step=2000,
                 verbose_step=10000, valid_size=16, valid_window=100):
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.embed_size = embed_size
        self.skip_window = skip_window
        self.num_skips = num_skips
        self.num_sampled = num_sampled
        self.lr = lr
        self.std_param = std_param
        self.init_param = init_param
        self.num_steps = num_steps
        self.show_step = show_step
        self.verbose_step = verbose_step
        self.valid_size = valid_size
        self.valid_window = valid_window
        self.valid_examples = np.array(random.sample(range(self.valid_window), self.valid_size))


class UserConfig(Config):
    def __init__(self, user_args):
        super().__init__(vocab_size=user_args.vocab_size,
                         batch_size=user_args.batch_size,
                         embed_size=user_args.embed_size,
                         skip_window=user_args.skip_window,
                         num_skips=user_args.num_skips,
                         num_sampled=user_args.num_sampled,
                         lr=user_args.learning_rate,
                         num_steps=user_args.num_steps,
                         show_step=user_args.show_step,
                         verbose_step=user_args.verbose_step,
                         valid_size=user_args.valid_size,
                         valid_window=user_args.valid_window)


class SkipGramModel:
    """
    The Skipgram model. This class only instatiates
    the tensorflow graph for the model.
    """
    def __init__(self, config):
        """
        :type config: Config
        """
        self.logdir = util.newlogname()
        self.config = config
        self.vocab_size = self.config.vocab_size
        self.embed_size = self.config.embed_size
        self.batch_size = self.config.batch_size
        self.num_sampled = self.config.num_sampled
        self.lr = self.config.lr
        self.std_param = self.config.std_param
        self.init_param = self.config.init_param
        self.valid_examples = self.config.valid_examples
        self.build_graph()

    def create_placeholders(self):
        """
        Create placeholder for the models graph
        """
        with tf.name_scope("words"):
            self.center_words = tf.compat.v1.placeholder(tf.int32,
                                               shape=[self.batch_size],
                                               name='center_words')
            self.targets = tf.compat.v1.placeholder(tf.int32,
                                          shape=[self.batch_size, 1],
                                          name='target_words')
            self.valid_dataset = tf.constant(self.valid_examples,
                                             dtype=tf.int32)

    def create_weights(self):
        """
        Create all the weights and biases for the model's graph,
        and store them in self.trainable_variables.
        """
        emshape = (self.vocab_size, self.embed_size)
        eminit = tf.random.uniform(emshape, -self.init_param[0], self.init_param[1])
        self.embeddings = tf.Variable(eminit, name="embeddings", trainable=True)

        with tf.name_scope("softmax"):
            Wshape = (self.vocab_size, self.embed_size)
            bshape = (self.vocab_size)
            std = 1.0 / (self.config.embed_size ** self.std_param)
            Winit = tf.random.truncated_normal(Wshape, stddev=std)
            binit = tf.zeros(bshape)

            self.weights = tf.Variable(Winit, name="weights", trainable=True)
            self.biases = tf.Variable(binit, name="biases", trainable=True)

        # Store the trainable variables
        self.trainable_variables = [self.embeddings, self.weights, self.biases]

    def create_loss(self):
        """
        Create the loss function of the model
        """
        with tf.name_scope("loss"):
            self.embed = tf.nn.embedding_lookup(self.embeddings,
                                                self.center_words,
                                                name='embed')
            self.loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(self.weights,
                                                                  self.biases,
                                                                  self.targets,
                                                                  self.embed,
                                                                  self.num_sampled,
                                                                  self.vocab_size))

    def create_optimizer(self):
        """
        Create the optimization of the model

        with tf.name_scope("train"):
            opt = tf.optimizers.Adam(self.lr)
            self.optimizer = opt.minimize(self.loss)
        """            
        with tf.name_scope("train"):
            opt = tf.optimizers.Adam(self.lr)
            grads = tf.gradients(self.loss, self.trainable_variables)
            self.optimizer = opt.apply_gradients(zip(grads, self.trainable_variables))
    

    def create_valid(self):
        """
        Create the valid vectors for comparison
        """
        norm = tf.sqrt(tf.reduce_sum(tf.square(self.embeddings),
                                     1, keepdims=True))
        self.normalized_embeddings = self.embeddings / norm
        valid_embeddings = tf.nn.embedding_lookup(self.normalized_embeddings,
                                                  self.valid_dataset)
        self.similarity = tf.matmul(valid_embeddings,
                                    tf.transpose(self.normalized_embeddings))

    def create_summaries(self):
        """
        Create the summary for the loss.
        """
        with tf.name_scope("summaries"):
            tf.summary.scalar("loss", self.loss, step=0)

    def build_graph(self):
        """
        Build the graph for our model
        """
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.create_placeholders()
            self.create_weights()
            self.create_loss()
            self.create_optimizer()
            self.create_valid()
            self.create_summaries()

'''
def run_training(model, data, config):
    optimizer = tf.optimizers.Adagrad(config.lr)
    average_loss = 0
    data_index = 0

    for step in range(config.num_steps):
        batch_data, batch_labels = data.batch_generator(config.batch_size, config.num_skips, config.skip_window, data_index)
        with tf.GradientTape() as tape:
            loss = model.compute_loss(batch_data, batch_labels)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        average_loss += loss.numpy()

        if step % config.show_step == 0:
            if step > 0:
                average_loss /= config.show_step
            print("Average loss at step {}: {:.4f}".format(step, average_loss))
            average_loss = 0

        if step % config.verbose_step == 0:
            sim = model.embeddings(np.array(config.valid_examples))
            sim = tf.matmul(sim, model.embeddings.embeddings, transpose_b=True).numpy()
            for i in range(config.valid_size):
                valid_word = data.index2word[config.valid_examples[i]]
                nearest = (-sim[i, :]).argsort()[1:9]
                log = "Nearest to {}:".format(valid_word)
                for k in range(8):
                    close_word = data.index2word[nearest[k]]
                    log = "{} {}, ".format(log, close_word)
                print(log)

    return model.embeddings.numpy()
'''
def run_training(model, data, verbose=True, visualization=True, debug=False):
    """
    Function to train the model. We use the parameter "verbose" to show
    some the words during training; "visualization" adds ternsorboard
    visualization; if "debug" is True then the return will be the duration of
    the training and the mean of the loss, and if "debug" is False this
    function returns the matrix of word embeddings.

    :type model: SkipGramModel
    :type data: Datareader
    :type verbose: boolean
    :type visualization: boolean
    :type debug: boolean
    :rtype duration: float
    :rtype avg_loss: float
    :rtype final_embeddings: np array -> [shape = (model.vocab_size,
                             model.embed_size), dtype=np.float32]
    """
    logdir = model.logdir
    batch_size = model.config.batch_size
    num_skips = model.config.num_skips
    skip_window = model.config.skip_window
    valid_examples = model.config.valid_examples
    num_steps = model.config.num_steps
    show_step = model.config.show_step
    verbose_step = model.config.verbose_step
    data_index = 0
    with tf.compat.v1.Session(graph=model.graph) as session:
        tf.compat.v1.global_variables_initializer().run()
        ts = time.time()
        print("Initialized")
        if visualization:
            print("\n&&&&&&&&& For TensorBoard visualization type &&&&&&&&&&&")
            print("\ntensorboard  --logdir={}\n".format(logdir))
            print("\n&&&&&&&&& And for the 3d embedding visualization type &&")
            print("\ntensorboard  --logdir=./processed\n")
        average_loss = 0
        total_loss = 0
        if visualization:
            writer = tf.compat.v1.summary.FileWriter(logdir, session.graph)
        for step in range(num_steps):
            data_index, batch_data, batch_labels = data.batch_generator(batch_size,
                                                                        num_skips,
                                                                        skip_window,
                                                                        data_index)
            feed_dict = {model.center_words: batch_data,
                         model.targets: batch_labels}
            _, l, summary = session.run([model.optimizer,
                                         model.loss,
                                         model.summary_op],
                                        feed_dict=feed_dict)
            average_loss += l
            total_loss += l
            if visualization:
                writer.add_summary(summary, global_step=step)
                writer.flush()
            if step % show_step == 0:
                if step > 0:
                    average_loss = average_loss / show_step
                    print("Average loss at step", step, ":", average_loss)
                    average_loss = 0
            if step % verbose_step == 0 and verbose:
                sim = model.similarity.eval()
                for i in range(model.config.valid_size):
                    valid_word = data.index2word[valid_examples[i]]
                    top_k = 8
                    nearest = (-sim[i, :]).argsort()[1:top_k+1]
                    log = "Nearest to %s:" % valid_word
                    for k in range(top_k):
                        close_word = data.index2word[nearest[k]]
                        log = "%s %s," % (log, close_word)
                    print(log)

        final_embeddings = model.normalized_embeddings.eval()
        if visualization:
            embedding_var = tf.Variable(final_embeddings[:1000],
                                        name='embedding')
            session.run(embedding_var.initializer)
            emconfig = projector.ProjectorConfig()
            summary_writer = tf.summary.FileWriter('processed')
            embedding = emconfig.embeddings.add()
            embedding.tensor_name = embedding_var.name
            embedding.metadata_path = 'processed/vocab_1000.tsv'
            projector.visualize_embeddings(summary_writer, emconfig)
            saver_embed = tf.train.Saver([embedding_var])
            saver_embed.save(session, 'processed/model3.ckpt', 1)

    te = time.time()
    duration = te - ts
    avg_loss = total_loss / num_steps
    if debug:
        return duration, avg_loss
    else:
        return final_embeddings

def create_argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("-f",
                        "--file",
                        type=str,
                        default='basic',
                        help="text file to apply the model (default=basic_pt.txt)")

    parser.add_argument("-s",
                        "--num_steps",
                        type=int,
                        default=100000,
                        help="number of training steps (default=100000)")

    parser.add_argument("-v",
                        "--vocab_size",
                        type=int,
                        default=50000,
                        help="vocab size (default=50000)")

    parser.add_argument("-b", "--batch_size", type=int,
                        default=140,
                        help="batch size (default=140)")

    parser.add_argument("-e",
                        "--embed_size",
                        type=int,
                        default=128,
                        help="embeddings size (default=128)")

    parser.add_argument("-k",
                        "--skip_window",
                        type=int,
                        default=1,
                        help="skip window (default=1)")

    parser.add_argument("-n",
                        "--num_skips",
                        type=int,
                        default=2,
                        help="""number of skips, number of times
                        a center word will be re-used (default=2)""")

    parser.add_argument("-S",
                        "--num_sampled",
                        type=int,
                        default=64,
                        help="number of negative samples(default=64)")

    parser.add_argument("-l",
                        "--learning_rate",
                        type=float,
                        default=1.0,
                        help="learning rate (default=1.0)")

    parser.add_argument("-w",
                        "--show_step",
                        type=int,
                        default=2000,
                        help="show result in multiples of this step (default=2000)")

    parser.add_argument("-B",
                        "--verbose_step",
                        type=int,
                        default=10000,
                        help="show similar words in multiples of this step (default=10000)")

    parser.add_argument("-V",
                        "--valid_size",
                        type=int,
                        default=16,
                        help="number of words to display similarity(default=16)")

    parser.add_argument("-W",
                        "--valid_window",
                        type=int,
                        default=100,
                        help="number of words from vocab to choose the words to display similarity (default=100)")

    return parser

def create_processed_dir():
    process_dir = 'processed/'
    if not os.path.exists(process_dir):
        os.makedirs(process_dir)

def process_text_data(file_path, vocab_size):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read().split()

    # Count frequency of each word
    word_counts = Counter(text)
    most_common = word_counts.most_common(vocab_size)

    # Create dictionaries
    word2index = {word: i for i, (word, _) in enumerate(most_common)}
    index2word = {i: word for word, i in word2index.items()}

    # Convert words to indices
    indexed_text = [word2index.get(word, 0) for word in text]

    # Create DataReader object (assuming it has a constructor that accepts these arguments)
    my_data = DataReader(indexed_text, word2index, index2word)

    return my_data

def main():
    parser = create_argument_parser()

    user_args = parser.parse_args()
    file_path = user_args.file

    if file_path == 'basic':
        file_path = util.get_path_basic_corpus()
        user_args.vocab_size = 500

    config = UserConfig(user_args)
    my_data = process_text_data(file_path, config.vocab_size)
    create_processed_dir()

    current_dir = os.path.dirname(__file__)
    old_vocab_path = os.path.join(current_dir, 'vocab_1000.tsv')
    new_vocab_path = os.path.join(current_dir, 'processed')
    new_vocab_path = os.path.join(new_vocab_path, 'vocab_1000.tsv')
    os.rename(old_vocab_path, new_vocab_path)

    my_model = SkipGramModel(config)
    embeddings = run_training(my_model, my_data)

    pickle_dir = 'pickles/'
    if not os.path.exists(pickle_dir):
        os.makedirs(pickle_dir)
    inverse = file_path[::-1][4:]
    number = -1
    for i, char in enumerate(inverse):
        if char == "/":
            number = i
            break
    if number == -1:
        filename = inverse[::-1] + '.pickle'
    else:
        filename = inverse[:number][::-1] + '.pickle'

    prefix = os.path.join(current_dir, 'pickles')
    filename = os.path.join(prefix, filename)

    f = open(filename, 'wb')
    di = {'word2index': my_data.word2index,
          'index2word': my_data.index2word,
          'embeddings': embeddings}
    pickle.dump(di, f)
    f.close()

    print("\n==========================================")
    print("""\nThe emmbedding vectors can be found in
      {}""".format(filename))

if __name__ == "__main__":
    main()

