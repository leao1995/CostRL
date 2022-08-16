import torch
import torch.nn as nn
import torch.jit as jit

class EmbeddingPool(jit.ScriptModule):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()

        self.pool = nn.ModuleList([
            nn.Embedding(n, embedding_dim) for n in num_embeddings
        ])
        self.output_dim = embedding_dim * len(num_embeddings)

    @jit.script_method
    def forward(self, x):        
        embed = []
        for i, module in enumerate(self.pool):
            embed.append(module(x[...,i]))
        embed = torch.cat(embed, dim=-1)

        return embed