import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
import tensorflow_hub as hub
import numpy as np
import tf_sentencepiece

# Some texts of different lengths.
english_sentences = ["dog", "Puppies are nice.", "I enjoy taking long walks along the beach with my dog."]
italian_sentences = ["cane", "I cuccioli sono carini.", "Mi piace fare lunghe passeggiate lungo la spiaggia con il mio cane."]
japanese_sentences = ["?", "???????", "?????????????????????"]

# Graph set up.
g = tf.Graph()
with g.as_default():
  text_input = tf.placeholder(dtype=tf.string, shape=[None])
  embed = hub.Module("https://www.kaggle.com/models/google/universal-sentence-encoder/TensorFlow1/multilingual/1")
  embedded_text = embed(text_input)
  init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
g.finalize()

# Initialize session.
session = tf.Session(graph=g)
session.run(init_op)

# Compute embeddings.
en_result = session.run(embedded_text, feed_dict={text_input: english_sentences})
it_result = session.run(embedded_text, feed_dict={text_input: italian_sentences})
ja_result = session.run(embedded_text, feed_dict={text_input: japanese_sentences})

# Compute similarity matrix. Higher score indicates greater similarity.
similarity_matrix_it = np.inner(en_result, it_result)
similarity_matrix_ja = np.inner(en_result, ja_result)

