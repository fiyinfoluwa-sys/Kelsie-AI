import torch
import torch.nn as nn

from tokenizer import NUM_BUCKETS


class IntentClassifier(nn.Module):
    """Hashed n-gram embeddings -> masked mean -> small MLP -> intent logits."""

    def __init__(self, num_labels: int, num_buckets: int = NUM_BUCKETS, dim: int = 96,
                 hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        self.embed = nn.Embedding(num_buckets, dim, padding_idx=0)
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, num_labels)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:  # ids: [batch, seq] int64
        # Row 0 of the table (padding) is always zero (padding_idx gets no gradient and AdamW's
        # decoupled decay keeps it at 0), so padded positions add nothing to the sum. That lets us
        # skip multiplying a [batch, seq, dim] mask -- the top op in the profile before this change.
        emb = self.embed(ids)
        count = (ids != 0).sum(1, keepdim=True).clamp(min=1).to(emb.dtype)
        pooled = emb.sum(1) / count
        return self.fc2(self.drop(self.act(self.fc1(self.norm(pooled)))))
