import torch

from chef.config import ChefConfig
from chef.model import ChefLM


def _tiny_config():
    # Small enough to instantiate/forward quickly in CI, but keeps the
    # same shape relationships as the real config (see test_config.py).
    return ChefConfig(
        vocab_size=64,
        max_seq_len=16,
        d_model=32,
        n_layers=2,
        n_heads=4,
        ffn_hidden=64,
        dropout=0.0,
    )


def test_forward_pass_output_shape():
    config = _tiny_config()
    model = ChefLM(config)
    batch_size, seq_len = 2, 8
    idx = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    logits, loss = model(idx)

    assert logits.shape == (batch_size, seq_len, config.vocab_size)
    assert loss is None


def test_forward_pass_with_targets_computes_loss():
    config = _tiny_config()
    model = ChefLM(config)
    batch_size, seq_len = 2, 8
    idx = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    logits, loss = model(idx, targets)

    assert loss is not None
    assert loss.item() > 0
