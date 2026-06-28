# Linear Attention Transformer: Architecture and Mathematical Foundations

This document outlines the design and mathematical underpinnings of a Linear Attention Transformer, tailored for processing sequences exceeding 1 million words while adhering to a single-file, zero-dependency Python implementation (using only the standard library).

## 1. Core Principles for Long Contexts

Traditional Transformer models suffer from quadratic complexity (O(n^2)) with respect to sequence length (n) due to the softmax attention mechanism. This makes them impractical for very long sequences. Linear Attention mechanisms address this by reformulating the attention calculation to achieve **linear complexity (O(n))** in both time and memory, making 1M+ word contexts feasible.

## 2. Mathematical Foundations (Standard Library Implementation)

All mathematical operations will be implemented from scratch using Python's built-in types (lists, numbers) and the `math` module. Key operations include:

### 2.1 Vector and Matrix Operations

*   **Vector Addition/Subtraction**: Element-wise operations.
    `C[i] = A[i] + B[i]`
*   **Vector-Scalar Multiplication**: `C[i] = A[i] * s`
*   **Dot Product (Vector-Vector)**: `sum(A[i] * B[i] for i in range(len(A)))`
*   **Matrix Multiplication (A @ B)**: Given `A` (m x k) and `B` (k x n), `C` (m x n) where `C[i][j] = sum(A[i][p] * B[p][j] for p in range(k))`.
*   **Matrix Transposition**: Swapping rows and columns.

### 2.2 Activation Functions

*   **ReLU (Rectified Linear Unit)**: `max(0, x)`
*   **GELU (Gaussian Error Linear Unit)**: Approximated as `0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x**3)))` or a simpler approximation if `tanh` is too complex to implement from scratch.

### 2.3 Normalization

*   **Layer Normalization**: For a given input vector `x`:
    `mean = sum(x) / len(x)`
    `variance = sum((val - mean)**2 for val in x) / len(x)`
    `std_dev = sqrt(variance + epsilon)`
    `normalized_x[i] = (x[i] - mean) / std_dev`
    Then, `y = gamma * normalized_x + beta`, where `gamma` and `beta` are learnable parameters.

### 2.4 Softmax (for output layer, not attention)

*   `softmax(x_i) = exp(x_i) / sum(exp(x_j) for all j)`

## 3. Linear Attention Mechanism

The core idea is to replace the `softmax(Q @ K.T) @ V` operation with a linear approximation or reordering. A common approach involves using a kernel function `phi` such that `softmax(Q @ K.T)` can be approximated by `phi(Q) @ phi(K).T`. Then, the attention can be computed as `(phi(Q) @ (phi(K).T @ V))`. The inner product `phi(K).T @ V` can be computed efficiently.

*   **Query (Q), Key (K), Value (V) Projections**: Linear transformations of input embeddings.
*   **Kernel Function (phi)**: A non-negative function applied element-wise, e.g., `phi(x) = exp(x)` or `phi(x) = ReLU(x) + 1`. For simplicity and standard library compatibility, `phi(x) = ReLU(x) + 1` is a good candidate.
*   **Linear Attention Calculation**: 
    1.  Apply `phi` to Q and K: `Q' = phi(Q)`, `K' = phi(K)`.
    2.  Compute `D = K'.T @ V` (sum over sequence length, resulting in `d_model x d_value` matrix).
    3.  Compute `Attention_Output = Q' @ D` (resulting in `seq_len x d_value` matrix).
    4.  Normalization (optional, but often used): `Attention_Output / (Q' @ sum(K', axis=1))`.

## 4. Architectural Components

### 4.1 Tokenizer

*   **Character-level Tokenizer**: Simple mapping of characters to integer IDs and vice-versa. This avoids external dependencies and handles any input text.

### 4.2 Embedding Layer

*   **Token Embeddings**: A lookup table (list of lists) where each row is the embedding vector for a token ID.
*   **Positional Embeddings**: Learnable or fixed sinusoidal embeddings added to token embeddings. For simplicity, a fixed sinusoidal positional encoding can be generated using `math.sin` and `math.cos`.

### 4.3 Linear Attention Block

Each block will consist of:

1.  **Linear Attention Sublayer**: 
    *   Input `x` goes through Q, K, V linear projections.
    *   Linear Attention calculation as described above.
    *   Output of attention is passed through a final linear projection.
    *   Add & Norm: `LayerNorm(x + Attention_Output)`.

2.  **Feed-Forward Sublayer**: 
    *   Input `y` goes through two linear layers with an activation function (e.g., GELU) in between.
    *   Add & Norm: `LayerNorm(y + FFN_Output)`.

### 4.4 Transformer Stack

*   Multiple Linear Attention Blocks stacked sequentially.

### 4.5 Output Layer

*   A final linear layer projecting the output of the last Transformer block to the vocabulary size.
*   Softmax activation to get probability distribution over vocabulary.

## 5. Model Parameters

*   `vocab_size`: Number of unique characters.
*   `d_model`: Dimension of embeddings and hidden states.
*   `n_layers`: Number of Linear Attention Blocks.
*   `d_ff`: Dimension of the inner layer in the Feed-Forward Network.

This design provides a solid foundation for building a functional, self-contained LLM with linear attention using only Python's standard library. The next phase will involve implementing these components.

