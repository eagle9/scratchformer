# Project: [Insert Project Name Here]
> A ground-up implementation of the Transformer architecture to master the mechanics of modern NLP.

## 🎯 Project Goals
- Build a functional Decoder-only Transformer (GPT-style).
- Understand tensor manipulation and dimensionality.
- Implement training and inference loops from scratch.

---

## 🏗️ The Roadmap

### Phase 1: The Core Mechanism (Attention)
*Focus: Mastering the "Attention is All You Need" math.*
- [ ] **Task 1.1: Scaled Dot-Product Attention** Implement the fundamental formula:  
  $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
- [ ] **Task 1.2: Multi-Head Attention (MHA)** Implement head splitting, parallel calculation, and concatenation. Pay close attention to `reshape` and `transpose` operations.

### Phase 2: The Supporting Cast
*Focus: Structural components that ensure stability and order.*
- [ ] **Task 2.1: Positional Encoding** Add temporal information using sine/cosine functions so the model knows word order.
- [ ] **Task 2.2: Feed-Forward Network (FFN)** Implement the position-wise linear layers: `Linear -> ReLU/GeLU -> Linear`.
- [ ] **Task 2.3: Layer Normalization & Residuals** Implement "Add & Norm" blocks to allow for deep gradient flow.

### Phase 3: Assembly (Decoder-Only Architecture)
*Focus: Building the "Brain" using the modules from Phase 1 & 2.*
- [ ] **Task 3.1: The Transformer Block** Create a single class that wraps MHA, FFN, and LayerNorm.
- [ ] **Task 3.2: Causal Masking** Create a triangular mask to prevent the model from "seeing the future" during training.
- [ ] **Task 3.3: The Full Model Shell** Combine an Embedding layer, a stack of Transformer blocks, and a final Linear head.

### Phase 4: Data & Training
*Focus: Teaching the model to talk.*
- [ ] **Task 4.1: Character-Level Tokenizer** Create a simple mapping of unique characters to integers.
- [ ] **Task 4.2: The Training Loop** Write the standard PyTorch boilerplate: `optimizer.zero_grad()`, `loss.backward()`, `optimizer.step()`.
- [ ] **Task 4.3: Greedy & Random Sampling** Write a `generate` function to produce text from a starting prompt.

### Phase 5: Deepening & Optimization
*Focus: Going from "working" to "efficient."*
- [ ] **Weight Tying:** Share weights between input embeddings and output linear layer.
-
