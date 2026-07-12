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


def test_vocab_size_matches_prepare_data_constant():
    # ChefConfig.vocab_size and prepare_data.VOCAB_SIZE are two separate
    # constants that MUST stay equal — one sizes the model's embedding
    # table + lm_head, the other sizes the tokenizer actually trained and
    # loaded at inference time. If they drift apart, shapes won't line up.
    from chef.prepare_data import VOCAB_SIZE
    assert ChefConfig().vocab_size == VOCAB_SIZE


def test_lang_tag_ids_are_distinct_from_core_special_tokens():
    config = ChefConfig()
    ids = {config.pad_id, config.bos_id, config.eos_id, config.lang_en_id, config.lang_ar_id}
    assert len(ids) == 5, "special token ids must all be distinct"
