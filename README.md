Transformer Model from Scratch using TensorFlow

This repository contains an implementation of the Transformer model from scratch using TensorFlow and Keras. The Transformer architecture, introduced in the paper Attention Is All You Need, is widely used in sequence-to-sequence tasks, particularly in NLP applications like machine translation, text summarization, and more.

Table of Contents

Introduction

Architecture Overview

Encoder

Decoder

Implementation Details

Required Libraries

Positional Encoding

Multi-Head Attention

Feed-Forward Network

Transformer Blocks

Encoder Implementation

Decoder Implementation

Full Transformer Model

Training and Evaluation

Conclusion

Introduction

The Transformer architecture has revolutionized natural language processing and sequence modeling tasks. Unlike traditional models like RNNs and LSTMs, the Transformer does not rely on recurrence but instead utilizes a self-attention mechanism to capture long-range dependencies efficiently.

Key components of the Transformer model include:

Multi-Head Attention

Positional Encoding

Feed-Forward Networks

Layer Normalization

Residual Connections

This repository demonstrates a complete implementation of a Transformer model using TensorFlow and Keras.

Architecture Overview

Encoder

The encoder processes the input sequence and transforms it into a contextual representation for the decoder. It consists of:

Multi-Head Self-Attention: Captures relationships between words regardless of their distance in the sequence.

Feed-Forward Networks: Further refines the encoded representations for better feature extraction.

Decoder

The decoder generates the output sequence based on the encoder’s output. It consists of:

Masked Multi-Head Self-Attention: Ensures that each word only attends to previous words, maintaining autoregressive properties.

Encoder-Decoder Attention: Allows the decoder to focus on relevant parts of the input sequence.

Feed-Forward Networks: Similar to the encoder, processing features at each position independently.

Implementation Details

Required Libraries

To implement the Transformer model, we import the necessary libraries:

import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, Dropout, LayerNormalization
from tensorflow.keras.models import Model
import numpy as np

Positional Encoding

Since the Transformer model does not inherently encode sequential order, positional encoding is used to retain information about word positions.

def positional_encoding(position, d_model):
    angle_rads = np.arange(position)[:, np.newaxis] / np.power(10000, (2 * (np.arange(d_model) // 2)) / np.float32(d_model))
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return tf.cast(angle_rads[np.newaxis, ...], dtype=tf.float32)

Multi-Head Attention

The multi-head attention mechanism allows the model to attend to different positions in the sequence simultaneously.

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads
        self.wq = Dense(d_model)
        self.wk = Dense(d_model)
        self.wv = Dense(d_model)
        self.dense = Dense(d_model)
    
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, v, k, q, mask):
        q = self.split_heads(self.wq(q), tf.shape(q)[0])
        k = self.split_heads(self.wk(k), tf.shape(k)[0])
        v = self.split_heads(self.wv(v), tf.shape(v)[0])
        
        attention, _ = self.scaled_dot_product_attention(q, k, v, mask)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        return self.dense(tf.reshape(attention, (tf.shape(q)[0], -1, self.d_model)))

Feed-Forward Network

A simple feed-forward network processes each position independently after the attention mechanism:

class PositionwiseFeedforward(tf.keras.layers.Layer):
    def __init__(self, d_model, dff):
        super(PositionwiseFeedforward, self).__init__()
        self.dense1 = Dense(dff, activation='relu')
        self.dense2 = Dense(d_model)
    
    def call(self, x):
        return self.dense2(self.dense1(x))

Transformer Blocks

Each transformer block consists of multi-head attention, feed-forward networks, and layer normalization:

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFeedforward(d_model, dff)
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)
    
    def call(self, x, training, mask):
        attn_output = self.dropout1(self.att(x, x, x, mask), training=training)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.dropout2(self.ffn(out1), training=training)
        return self.layernorm2(out1 + ffn_output)

Full Transformer Model

The complete Transformer model integrates the encoder and decoder:

class Transformer(Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, pe_input, pe_target, rate=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size, pe_input, rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, pe_target, rate)
        self.final_layer = Dense(target_vocab_size)
    
    def call(self, inputs, targets, training, look_ahead_mask, padding_mask):
        enc_output = self.encoder(inputs, training, padding_mask)
        dec_output = self.decoder(targets, enc_output, training, look_ahead_mask, padding_mask)
        return self.final_layer(dec_output)

Conclusion

This repository provides a foundational implementation of the Transformer model from scratch using TensorFlow. The model can be extended for various NLP tasks like translation, text summarization, and question-answering.

