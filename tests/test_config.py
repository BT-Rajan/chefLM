from chef.config import ChefConfig, TrainConfig


def test_config_defaults_are_consistent():
    config = ChefConfig()
    assert config.d_model % config.n_heads == 0, (
        "d_model must be divisible by n_heads for multi-head attention to split evenly"
    )
    assert config.vocab_size > 0
    assert config.max_seq_len > 0
    assert config.pad_id != config.bos_id != config.eos_id


def test_train_config_defaults():
    train_config = TrainConfig()
    assert train_config.max_steps > train_config.warmup_steps
    assert 0.0 < train_config.learning_rate
    assert train_config.min_lr < train_config.learning_rate
