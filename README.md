# Transformer Model from Scratch using TensorFlow

## Introduction

The Transformer architecture has revolutionized natural language processing and sequence modeling tasks, providing a highly parallelizable structure with faster training and better performance than traditional models like RNNs or LSTMs. This repository demonstrates the complete implementation of the Transformer model using TensorFlow and Keras.

The model consists of an encoder-decoder architecture, each comprising several key components such as:
- **Multi-Head Attention**
- **Positional Encoding**
- **Feed-Forward Networks**
- **Layer Normalization**

---

## Architecture Overview

### Encoder
The **encoder** processes the input sequence and converts it into an internal representation that the decoder can use to generate an output sequence. The encoder consists of:
1. **Multi-Head Self-Attention Mechanism**: Captures dependencies between tokens.
2. **Feed-Forward Networks**: Processes each position of the sequence independently.

### Decoder
The **decoder** generates the output sequence based on the encoder's output. It is composed of:
1. **Masked Multi-Head Self-Attention**: Prevents attending to future tokens in the sequence.
2. **Encoder-Decoder Attention**: Allows the decoder to focus on relevant parts of the input sequence.
3. **Feed-Forward Networks**: Similar to the encoder's feed-forward mechanism.

---

## Implementation

### 1. Importing Required Libraries
To begin, we import the required libraries:

```python
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Embedding, Dropout, LayerNormalization
from tensorflow.keras.models import Model
import numpy as np
```

### 2. Positional Encoding
The Transformer model uses positional encodings to incorporate sequence order since it does not inherently account for token positions.

```python
def positional_encoding(position, d_model):
    angle_rads = np.arange(position)[:, np.newaxis] / np.power(10000, (2 * (np.arange(d_model) // 2)) / np.float32(d_model))
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return tf.cast(angle_rads[np.newaxis, ...], dtype=tf.float32)
```

### 3. Multi-Head Attention
The **multi-head attention** mechanism is used to capture dependencies between tokens in a sequence. Each attention head focuses on different parts of the sequence, allowing the model to learn various representations.

```python
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

    def scaled_dot_product_attention(self, q, k, v, mask):
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        return tf.matmul(attention_weights, v), attention_weights
```

### 4. Feed-Forward Network
A fully connected feed-forward network processes each position independently after the attention mechanism:

```python
class PositionwiseFeedforward(tf.keras.layers.Layer):
    def __init__(self, d_model, dff):
        super(PositionwiseFeedforward, self).__init__()
        self.dense1 = Dense(dff, activation='relu')
        self.dense2 = Dense(d_model)

    def call(self, x):
        return self.dense2(self.dense1(x))
```

### 5. Transformer Block
Each **transformer block** combines multi-head attention and a feed-forward network with layer normalization and dropout:

```python
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
```

### 6. Encoder Implementation
The **encoder** consists of a stack of transformer blocks:

```python
class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, maximum_position_encoding, dropout_rate=0.1):
        super(Encoder, self).__init__()
        self.embedding = Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)
        self.enc_layers = [TransformerBlock(d_model, num_heads, dff, dropout_rate) for _ in range(num_layers)]
        self.dropout = Dropout(dropout_rate)

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]
        x = self.embedding(x) + self.pos_encoding[:, :seq_len, :]
        return tf.reduce(lambda x, layer: layer(x, training, mask), self.enc_layers, x)
```

### 7. Decoder Implementation
The **decoder** mirrors the encoder with an additional self-attention mechanism to prevent attending to future tokens:

```python
class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, maximum_position_encoding, dropout_rate=0.1):
        super(Decoder, self).__init__()
        self.embedding = Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)
        self.dec_layers = [TransformerBlock(d_model, num_heads, dff, dropout_rate) for _ in range(num_layers)]
        self.dropout = Dropout(dropout_rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        x = self.embedding(x) + self.pos_encoding[:, :seq_len, :]
        return tf.reduce(lambda x, layer: layer(x, training, look_ahead_mask), self.dec_layers, x)
```

### 8. Transformer Model
The complete **Transformer model** integrates the encoder and decoder:

```python
class Transformer(Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, pe_input, pe_target, rate=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size, pe_input, rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, pe_target, rate)
        self.final_layer = Dense(target_vocab_size)

    def call(self

, inputs, targets, training, look_ahead_mask, padding_mask):
        enc_output = self.encoder(inputs, training, padding_mask)
        dec_output = self.decoder(targets, enc_output, training, look_ahead_mask, padding_mask)
        return self.final_layer(dec_output)
```

### 9. Training and Testing
Define model parameters and test the forward pass with dummy data:

```python
# Model Parameters
num_layers = 4
d_model = 128
num_heads = 8
dff = 512
input_vocab_size = 8500
target_vocab_size = 8000
pe_input = 1000
pe_target = 1000
dropout_rate = 0.1

# Create the Transformer model
transformer = Transformer(num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, pe_input, pe_target, dropout_rate)

# Dummy Input for Forward Pass
inputs = tf.random.uniform((64, 50), dtype=tf.int64, minval=0, maxval=input_vocab_size)
targets = tf.random.uniform((64, 50), dtype=tf.int64, minval=0, maxval=target_vocab_size)
output = transformer(inputs, targets, training=True, look_ahead_mask=None, padding_mask=None)
print(output.shape)  # Expected output: (64, 50, target_vocab_size)
```

---

## Conclusion
This repository demonstrates the step-by-step implementation of a Transformer model from scratch using TensorFlow. The Transformer is highly effective for tasks such as machine translation, text summarization, and other NLP applications.

#   T r a n s f o r m e r - M o d e l - f r o m - S c r a t c h - u s i n g - T e n s o r F l o w 
 
 
