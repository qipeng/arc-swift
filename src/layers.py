import tensorflow as tf
import numpy as np

class DenseLayer:
    def __init__(self, inputsize, outputsize, nl=tf.nn.relu, keepProb=1.0, stddev=None, no_bias=False):
        self.W = tf.Variable(tf.truncated_normal([inputsize, outputsize], stddev=np.sqrt(1.0/(inputsize)) if stddev is None else stddev))
        self.b = tf.constant(0.0) if no_bias else tf.Variable(tf.zeros([1, outputsize]))
        self.nl = nl if keepProb == 1 else lambda x: tf.nn.dropout(nl(x), keepProb)

    def __call__(self, input):
        return self.nl(tf.matmul(input, self.W) + self.b)

class MergeLayer:
    def __init__(self, input1size, input2size, outputsize, nl=tf.nn.relu, keepProb=1.0, no_bias = False, combination='affine'):
        assert combination in ['affine', 'bilinear', 'biaffine'], "combination type '%s' not supported. Must be one of 'affine', 'bilinear', 'biaffine'" % (combination)
        self.b = tf.constant(0.0) if no_bias else tf.Variable(tf.zeros([outputsize]))
        self.combination = combination

        if combination == "affine" or combination == "biaffine":
            self.W_inp1_to_out_affine = tf.Variable(tf.truncated_normal([input1size, outputsize], stddev=np.sqrt(2.0/(input1size+outputsize)), dtype=tf.float32))
            self.W_inp2_to_out_affine = tf.Variable(tf.truncated_normal([input2size, outputsize], stddev=np.sqrt(2.0/(input2size+outputsize)), dtype=tf.float32))

        if combination == "bilinear" or combination == "biaffine":
            self.W_bilinear = tf.Variable(tf.truncated_normal([input1size, input2size, outputsize], stddev=np.sqrt(2.0/(input1size*input2size+outputsize))), dtype=tf.float32)

        self.nl = nl if keepProb == 1 else lambda x: tf.nn.dropout(nl(x), keepProb)

    def __call__(self, input1, input2):
        out = self.b

        if self.combination == "affine" or self.combination == "biaffine":
            out = tf.add(out, tf.add(tf.matmul(input1, self.W_inp1_to_out_affine),
                                     tf.matmul(input2, self.W_inp2_to_out_affine)))

        if self.combination == "bilinear" or self.combination == "biaffine":
            out = tf.add(out, tf.einsum('ij,ik,jkl->il', input1, input2, self.W_bilinear))

        return self.nl(out)
