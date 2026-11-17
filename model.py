import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    """
    Transformer Block with Pre-LayerNorm, Causal Multi-Head Self-Attention,
    and MLP feed-forward network.
    """
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        self.n_head = n_head
        self.dropout = dropout

        # Attention sub-layer
        self.ln1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

        # Feed-forward sub-layer
        self.ln2 = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, 4 * d_model, bias=False)
        self.fc_proj = nn.Linear(4 * d_model, d_model, bias=False)

        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        # 1. Self-Attention with Pre-LN & PyTorch Flash Attention
        q, k, v = self.qkv(self.ln1(x)).split(C, dim=2)
        q, k, v = [t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v)]

        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.drop(self.proj(y))

        # 2. MLP with Pre-LN & GELU activation
        mlp_out = self.fc_proj(F.gelu(self.fc(self.ln2(x))))
        x = x + self.drop(mlp_out)

        return x


class MiniGPT(nn.Module):
    """
    Decoder-only autoregressive Transformer language model (~27M parameters).
    Features:
    - Byte-Pair Encoding (BPE) subword token embeddings
    - Learned positional embeddings
    - Scaled Dot-Product Attention with Flash Attention support
    - Pre-LayerNorm residual architecture
    - Tied input/output weights
    - Scaled initialization on residual projections
    """
    def __init__(
        self,
        vocab_size: int = 4096,
        block_size: int = 256,
        n_layer: int = 8,
        n_head: int = 8,
        n_embd: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd

        # Token & Positional Embeddings
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(d_model=n_embd, n_head=n_head, dropout=dropout)
            for _ in range(n_layer)
        ])

        # Final LayerNorm & Language Modeling Head
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

        # Weight tying (Press & Wolf 2017)
        self.head.weight = self.wte.weight

        # Parameter initialization
        self.apply(self._init_weights)

        # Special scaled init for residual projections (following GPT-2 paper)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("fc_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        """
        Forward pass for training or inference.
        Args:
            idx: Tensor of shape (B, T) with token IDs.
            targets: Optional tensor of shape (B, T) with target token IDs.
        Returns:
            logits: (B, T, vocab_size)
            loss: Cross-entropy scalar loss (if targets is provided)
        """
        B, T = idx.shape
        assert T <= self.block_size, f"Cannot forward sequence of length {T}, block size is {self.block_size}"

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.drop(self.wte(idx) + self.wpe(pos))

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    def get_num_params(self, non_embedding: bool = False) -> int:
        """Return the number of parameters in the model."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.wpe.weight.numel()
        return n_params
