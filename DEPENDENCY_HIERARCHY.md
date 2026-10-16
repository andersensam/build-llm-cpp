# Complete Dependency Hierarchy

This document provides a comprehensive view of all dependencies, including abstract classes and interfaces.

## Full Class Inheritance Tree

```mermaid
graph TD
    subgraph "Storage Layer"
        storage[Storage<T>]
    end

    subgraph "Abstract Tensor Interface"
        abstract_tensor[AbstractTensor<T>]
    end

    subgraph "Tensor Implementations"
        tensor[Tensor<T>]
        list_slice[ListTensorSlice<T>]
    end

    subgraph "Abstract Embedding Interface"
        abstract_embedding[AbstractEmbedding<T>]
    end

    subgraph "Embedding Implementations"
        embedding[Embedding<T>]
        abs_pos[AbsolutePositionalEmbedding<T>]
    end

    subgraph "Tokenizer Interface"
        tokenizer[Tokenizer]
    end

    subgraph "Tokenizer Implementations"
        bpe[BytePairEncoding<T>]
    end

    subgraph "Model Interface"
        model[Model<T>]
    end

    subgraph "Model Implementations"
        gpt[GPTModel]
    end

    subgraph "Layer Interface"
        layer[Layer<T>]
    end

    subgraph "Layer Implementations"
        norm[NormalizationLayer<T>]
        dropout[DropoutLayer<T>]
        gelu[GELULayer<T>]
        linear[LinearLayer<T>]
    end

    subgraph "Attention Interface"
        attention[Attention<T>]
    end

    subgraph "Attention Implementations"
        mha[MultiHeadAttention<T>]
    end

    subgraph "FeedForward"
        feedforward[FeedForward<T>]
    end

    subgraph "Transformer Block"
        transformer[TransformerBlock<T>]
    end

    %% Storage dependencies
    storage -->|owns memory| tensor
    storage -->|used by| abstract_embedding

    %% Tensor hierarchy
    abstract_tensor -->|inherited by| tensor
    abstract_tensor -->|inherited by| list_slice
    abstract_tensor -->|used by| all other components

    %% Embedding hierarchy
    abstract_embedding -->|inherited by| embedding
    abstract_embedding -->|inherited by| abs_pos
    embedding -->|uses| abstract_tensor
    abs_pos -->|uses| abstract_tensor

    %% Tokenizer hierarchy
    tokenizer -->|inherited by| bpe

    %% Model hierarchy
    model -->|implemented by| gpt
    gpt -->|uses| transformer
    gpt -->|uses| embedding
    gpt -->|uses| abs_pos

    %% Layer hierarchy
    layer -->|inherited by| norm
    layer -->|inherited by| dropout
    layer -->|inherited by| gelu
    layer -->|inherited by| linear
    layer -->|inherited by| attention
    layer -->|inherited by| feedforward
    layer -->|inherited by| transformer

    %% Attention hierarchy
    attention -->|inherited by| mha
    attention -->|inherited by| layer

    %% FeedForward composition
    feedforward -->|has| linear
    feedforward -->|has| gelu
    feedforward -->|implements| layer

    %% MultiHeadAttention composition
    mha -->|uses| tensor
    mha -->|uses| tensor_matmul
    mha -->|implements| attention

    %% TransformerBlock composition
    transformer -->|has| norm
    transformer -->|has| mha
    transformer -->|has| dropout
    transformer -->|has| feedforward
    transformer -->|implements| layer

    %% GPTModel composition
    gpt -->|has| embedding
    gpt -->|has| abs_pos
    gpt -->|has| dropout
    gpt -->|has| norm
    gpt -->|has| transformer
```

## Interface Dependencies (Abstract Classes)

```
┌─────────────────────────────────────────────────────────────────┐
│                    ABSTRACT INTERFACES                          │
└─────────────────────────────────────────────────────────────────┘

Storage<T> (concrete template struct)
    └── owned by Tensor<T> via std::shared_ptr

AbstractTensor<T> (interface)
    ├── Tensor<T> (concrete implementation)
    │   ├── used by all tensor operations
    │   └── has Storage<T> via shared_ptr
    └── ListTensorSlice<T> (slice implementation)

AbstractEmbedding<T> (interface)
    ├── Embedding<T> (token embedding layer)
    └── AbsolutePositionalEmbedding<T> (position embedding layer)

Tokenizer (interface)
    └── BytePairEncoding<T> (BPE tokenizer implementation)

Model<T> (interface)
    └── GPTModel (GPT-style language model)

Layer<T> (interface)
    ├── NormalizationLayer<T>
    ├── DropoutLayer<T>
    ├── GELULayer<T>
    ├── LinearLayer<T>
    ├── Attention<T> (interface)
    │   └── MultiHeadAttention<T>
    ├── FeedForward<T>
    └── TransformerBlock<T>

Attention<T> (interface)
    └── MultiHeadAttention<T> (MHA implementation)
```

## Concrete Instantiation Chain (GPTModel)

```
┌─────────────────────────────────────────────────────────────────┐
│              GPTModel Instantiation Path                        │
└─────────────────────────────────────────────────────────────────┘

main.cpp
    │
    ├── creates BytePairEncoding (tokenizer)
    │   └── implements Tokenizer interface
    │
    ├── creates DataLoader
    │   └── uses BytePairEncoding for tokenization
    │
    └── creates GPTModel
            │
            ├── m_tok_emb: Embedding<float>
            │   └── implements AbstractEmbedding<float>
            │
            ├── m_pos_emb: AbsolutePositionalEmbedding<float>
            │   └── implements AbstractEmbedding<float>
            │
            ├── m_drop: DropoutLayer<float>
            │   └── implements Layer<float>
            │
            ├── m_norm: NormalizationLayer<float>
            │   └── implements Layer<float>
            │
            └── m_layers: vector<TransformerBlock<float>> × N
                    │
                    └── each TransformerBlock contains:
                            ├── NormalizationLayer<float>
                            ├── MultiHeadAttention<float>
                            │   ├── implements Attention<float>
                            │   ├── Q/K/V weight tensors
                            │   └── uses TensorMatmul for projections
                            ├── DropoutLayer<float>
                            ├── NormalizationLayer<float>
                            ├── FeedForward<float>
                            │   ├── LinearLayer<float> (up-projection)
                            │   ├── GELULayer<float>
                            │   └── LinearLayer<float> (down-projection)
                            └── DropoutLayer<float>
```

## Dependency Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA FLOW                                    │
└─────────────────────────────────────────────────────────────────┘

Text Data
    ↓
BytePairEncoding (tokenize)
    ↓
std::vector<size_t> (token IDs)
    ↓
DataLoader (batch creation)
    ↓
Tensor<size_t> [batch_size, seq_len]
    ↓
GPTModel.forward()
    ↓
    ├── Embedding: Tensor<size_t> → Tensor<float> [batch, seq, emb_dim]
    ├── AbsolutePositionalEmbedding: add position embeddings
    ├── DropoutLayer: apply dropout
    │
    └── TransformerBlock × N:
            ├── NormalizationLayer
            ├── MultiHeadAttention (QKV projections + causal masking)
            ├── Dropout + residual connection
            ├── NormalizationLayer
            ├── FeedForward (Linear → GELU → Linear)
            └── Dropout + residual connection
    ↓
    └── NormalizationLayer (final)
    ↓
    └── LinearLayer (output projection)
    ↓
Tensor<float> [batch_size, seq_len, vocab_size] (logits)
```

## Complete File Dependencies

```
main.cpp
├── GPTModel.hpp
│   ├── Model.hpp
│   ├── TransformerBlock.hpp
│   │   ├── Layer.hpp
│   │   ├── MultiHeadAttention.hpp
│   │   │   ├── Attention.hpp
│   │   │   ├── MultiHeadAttention.hpp (self-reference)
│   │   │   └── TensorMatmul.hpp
│   │   ├── FeedForward.hpp
│   │   │   ├── Layer.hpp
│   │   │   ├── LinearLayer.hpp
│   │   │   └── GELULayer.hpp
│   │   ├── NormalizationLayer.hpp
│   │   └── DropoutLayer.hpp
│   ├── Embedding.hpp
│   │   └── AbstractEmbedding.hpp
│   ├── AbsolutePositionalEmbedding.hpp
│   ├── DropoutLayer.hpp
│   └── NormalizationLayer.hpp
├── BytePairEncoding.hpp
│   ├── Tokenizer.hpp
│   └── SliceConfig.hpp
├── DataLoader.hpp
│   ├── BytePairEncoding.hpp
│   └── SliceConfig.hpp
├── Tensor.hpp
│   ├── AbstractTensor.hpp
│   ├── Storage.hpp
│   ├── Log.hpp
│   ├── Numerics.hpp
│   ├── TensorMath.hpp
│   │   └── Numerics.hpp
│   ├── TensorMatmul.hpp
│   └── TensorUtils.hpp
└── Log.hpp

All headers depend on:
├── AbstractTensor.hpp (via tensor operations)
├── Storage.hpp (via tensor memory management)
└── Numerics.hpp (via math utilities)
```

## Summary

### Core Architecture
1. **Storage Layer**: `Storage<T>` manages heap-allocated memory blocks
2. **Tensor Layer**: `Tensor<T>` inherits from `AbstractTensor<T>` with `Storage<T>` for memory
3. **Layer Interface**: All neural network layers implement `Layer<T>`
4. **Attention Interface**: `Attention<T>` extends `Layer<T>` for attention mechanisms
5. **Model Interface**: `Model<T>` defines the forward pass contract

### GPTModel Components (Instantiated)
- `Embedding<T>` - Token embeddings
- `AbsolutePositionalEmbedding<T>` - Position embeddings
- `DropoutLayer<T>` - Regularization
- `NormalizationLayer<T>` - Layer normalization
- `TransformerBlock<T>` - Transformer block (repeated)
  - `MultiHeadAttention<T>` - Self-attention
  - `FeedForward<T>` - MLP with GELU activation
- `LinearLayer<T>` - Output projection

### Key Dependencies
- `Tensor<T>` - Core data structure throughout
- `TensorMatmul<T>` - Matrix multiplication kernels
- `TensorMath<T>` - Mathematical operations
- `Storage<T>` - Memory management (via shared_ptr)
