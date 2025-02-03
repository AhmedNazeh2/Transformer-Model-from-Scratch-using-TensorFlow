# Transformer Model from Scratch using TensorFlow

This repository implements a Transformer model from scratch using TensorFlow. 
The Transformer architecture is designed for sequence-to-sequence tasks and relies entirely on a mechanism called **self-attention** to capture dependencies between input and output.

---

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
