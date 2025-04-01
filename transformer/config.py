from dataclasses import dataclass

import torch


@dataclass
class Config:
    num_heads: int = 8
    d_model: int = 512
    data_path: str = "transformer/data/input.txt"
    vocab_size: int = 5001
    eos_token: str = "<EOS>"
    eos_token_id: int = 5001
    seq_length: int = 128
    batch_size: int = 100
    num_layers: int = 10
    d_ff: int = 100
    n_x: int = 1
    dropout: float = 0.1
    learning_rate: float = 0.001
    num_epochs: int = 50
    model_path: str = "models/model.pth"
    config_path: str = "models/config.json"
    device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    beta_1: float = 0.9
    beta_2: float = 0.98
    epsilon: float = 1e-9
