import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

print("=== Part 1: Understanding Attention from Scratch ===")

# Simple example - 4 tokens, each with 8-dimensional embedding
# In real BERT: hundreds of tokens, 768 dimensions
# We use small numbers so you can see what's happening

SEQUENCE_LENGTH = 4   # 4 tokens: [def, add, (, a]
EMBEDDING_DIM = 8     # 8 numbers per token (768 in real BERT)

# Fake token embeddings (in real BERT these come from embedding lookup)
torch.manual_seed(42)  # makes random numbers same every run
tokens = torch.randn(SEQUENCE_LENGTH, EMBEDDING_DIM)

print(f"Token embeddings shape: {tokens.shape}")
print(f"  -> {SEQUENCE_LENGTH} tokens, each with {EMBEDDING_DIM} numbers")
print(f"\nToken embeddings:")
print(tokens.round(decimals=3))

print("\n=== Part 2: Create Q, K, V matrices ===")

# Weight matrices that transform embeddings into Q, K, V
# These are LEARNED during training
HEAD_DIM = 8  # dimension of each Q, K, V vector

W_Q = torch.randn(EMBEDDING_DIM, HEAD_DIM)  # Query weights
W_K = torch.randn(EMBEDDING_DIM, HEAD_DIM)  # Key weights  
W_V = torch.randn(EMBEDDING_DIM, HEAD_DIM)  # Value weights

print(f"Weight matrix shapes: {W_Q.shape}")
print(f"  -> transforms {EMBEDDING_DIM}-dim embedding to {HEAD_DIM}-dim Q/K/V")

# Create Q, K, V by multiplying tokens with weight matrices
Q = tokens @ W_Q  # @ is matrix multiplication in Python
K = tokens @ W_K
V = tokens @ W_V

print(f"\nQuery (Q) matrix shape: {Q.shape}")
print(f"Key   (K) matrix shape: {K.shape}")
print(f"Value (V) matrix shape: {V.shape}")
print(f"  → Each of our {SEQUENCE_LENGTH} tokens now has its own Q, K, V vector")

print("\n=== Part 3: Calculate Attention Scores ===")

# Step 1: Dot product between Q and K
# Q (4×8) @ K.T (8×4) → scores (4×4)
# K.T means K transposed (rows become columns)
scores = Q @ K.transpose(-2, -1)

print(f"Raw attention scores shape: {scores.shape}")
print(f"  → {SEQUENCE_LENGTH}×{SEQUENCE_LENGTH} matrix")
print(f"  → scores[i][j] = how much token i attends to token j")
print(f"\nRaw scores:")
tokens.round(decimals=3)

# Step 2: Scale by square root of head dimension
# WHY? Without scaling, dot products get very large
# with high dimensions, making softmax too sharp
# (one token gets ~100% attention, others get ~0%)
# Dividing by sqrt(dim) keeps scores in reasonable range
scale = math.sqrt(HEAD_DIM)
scaled_scores = scores / scale

print(f"\nScaled scores (divided by sqrt({HEAD_DIM})={scale:.2f}):")
print(scaled_scores.round(decimals=3))

print("\n=== Part 4: Softmax — Convert Scores to Probabilities ===")

# Apply softmax along last dimension (across keys for each query)
attention_weights = F.softmax(scaled_scores, dim=-1)

print(f"Attention weights (after softmax):")
print(attention_weights.round(decimals=3))
print(f"\nEach row sums to 1.0:")
print(attention_weights.sum(dim=-1).round(decimals=3))
print(f"\nInterpretation:")
print(f"  Row 0 = how much token 0 (def) attends to each other token")
print(f"  Row 1 = how much token 1 (add) attends to each other token")
print(f"  Higher number = stronger attention = more influence")

print("\n=== Part 5: Weighted Sum of Values ===")

# Final attention output
# attention_weights (4×4) @ V (4×8) → output (4×8)
attention_output = attention_weights @ V

print(f"Attention output shape: {attention_output.shape}")
print(f"  → Same shape as input! Each token now has")
print(f"     context-aware representation")
print(f"\nAttention output:")
print(attention_output.round(decimals=3))
print(f"\nEach token's new representation is a WEIGHTED MIX")
print(f"of all other tokens' values, based on attention weights")

print("\n=== Part 6: Full Self-Attention as a Class ===")

class SelfAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        # Linear layers to create Q, K, V
        self.W_Q = nn.Linear(embedding_dim, embedding_dim)
        self.W_K = nn.Linear(embedding_dim, embedding_dim)
        self.W_V = nn.Linear(embedding_dim, embedding_dim)
        
        # Output projection — combines all heads
        self.W_O = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        # Create Q, K, V
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)
        
        # Reshape for multi-head attention
        # Split embedding_dim into num_heads × head_dim
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        Q = Q.transpose(1, 2)  # (batch, heads, seq, head_dim)
        
        K = K.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        K = K.transpose(1, 2)
        
        V = V.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        V = V.transpose(1, 2)

        # Attention scores
        scale = math.sqrt(self.head_dim)
        scores = Q @ K.transpose(-2, -1) / scale
        weights = F.softmax(scores, dim=-1)
        
        # Weighted sum of values
        attended = weights @ V  # (batch, heads, seq, head_dim)
        
        # Recombine heads
        attended = attended.transpose(1, 2)  # (batch, seq, heads, head_dim)
        attended = attended.reshape(batch_size, seq_len, embed_dim)
        
        # Final projection
        output = self.W_O(attended)
        return output
    
    # Test it
EMBED_DIM = 64
NUM_HEADS = 4
BATCH = 1
SEQ_LEN = 10

attention = SelfAttention(EMBED_DIM, NUM_HEADS)
fake_input = torch.randn(BATCH, SEQ_LEN, EMBED_DIM)
output = attention(fake_input)

print(f"Input shape:  {fake_input.shape}")
print(f"Output shape: {output.shape}")
print(f"  → Same shape! But each token now has context from all others")
print(f"\n✅ Self-Attention working correctly!")

print("\n=== Part 7: Mini Transformer Block ===")

class TransformerBlock(nn.Module):
    """One complete transformer layer — like one of BERT's 12 layers"""
    
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        
        # Self attention
        self.attention = SelfAttention(embed_dim, num_heads)
        
        # Feed forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),              # smoother version of ReLU used in BERT
            nn.Linear(ff_dim, embed_dim)
        )
        
        # Layer normalization — keeps values stable
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Attention with residual connection
        attended = self.attention(x)
        x = self.norm1(x + attended)

        # Feed forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x

# Test transformer block
block = TransformerBlock(embed_dim=64, num_heads=4, ff_dim=256)
fake_input = torch.randn(1, 10, 64)
output = block(fake_input)

print(f"Transformer block input:  {fake_input.shape}")
print(f"Transformer block output: {output.shape}")
print(f"  → Shape preserved, meaning enriched")
print(f"\n✅ Full transformer block working!")

print("\n=== Part 8: Mini BERT for Code ===")

class MiniBERT(nn.Module):
    """
    Simplified BERT — same architecture as CodeBERT
    but much smaller. Real CodeBERT has:
    - 768 embedding dim (we use 64)
    - 12 attention heads (we use 4)
    - 12 layers (we use 3)
    - 125M parameters (we have ~200K)
    """
    
    def __init__(self, vocab_size, embed_dim, num_heads, 
                 num_layers, ff_dim, max_seq_len, num_classes):
        super().__init__()
        
        # Token embeddings — lookup table: token_id → vector
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Positional embeddings — learned position representations
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        
        # Stack of transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim)
            for _ in range(num_layers)
        ])
        
        # Classification head — takes [CLS] token output
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, num_classes)
        )
        
        self.dropout = nn.Dropout(0.1)

    def forward(self, token_ids):
        batch_size, seq_len = token_ids.shape
        
        # Create position indices [0, 1, 2, ..., seq_len-1]
        positions = torch.arange(seq_len).unsqueeze(0)
        # unsqueeze(0) adds batch dimension: (seq_len,) → (1, seq_len)
        
        # Get embeddings
        token_embeds = self.token_embedding(token_ids)
        pos_embeds = self.position_embedding(positions)
        
        # Combine token + position embeddings
        x = self.dropout(token_embeds + pos_embeds)
        
        # Pass through all transformer layers
        for block in self.transformer_blocks:
            x = block(x)
        
        # Take [CLS] token output (position 0)
        cls_output = x[:, 0, :]
        # x[:, 0, :] means: all batches, position 0, all dimensions
        
        # Classify
        logits = self.classifier(cls_output)
        return logits
    

    # Create mini BERT
mini_bert = MiniBERT(
    vocab_size=1000,    # small vocab for testing
    embed_dim=64,       # 768 in real BERT
    num_heads=4,        # 12 in real BERT
    num_layers=3,       # 12 in real BERT
    ff_dim=256,         # 3072 in real BERT
    max_seq_len=128,    # 512 in real BERT
    num_classes=2       # good or bad
)

# Count parameters
total_params = sum(p.numel() for p in mini_bert.parameters())
print(f"Mini BERT parameters: {total_params:,}")
print(f"Real CodeBERT parameters: 125,000,000")
print(f"Ratio: {125_000_000 // total_params}x bigger")

# Test forward pass
fake_tokens = torch.randint(0, 1000, (1, 20))  # 1 sample, 20 tokens
output = mini_bert(fake_tokens)
print(f"\nInput token IDs shape: {fake_tokens.shape}")
print(f"Output logits shape: {output.shape}")
print(f"Output: {output}")
print(f"\n✅ Mini BERT working! Tomorrow we load the REAL CodeBERT!")