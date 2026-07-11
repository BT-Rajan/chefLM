"""ChefLM configuration."""

from dataclasses import dataclass


@dataclass
class ChefConfig:
    # Matches the 768-token tokenizer. Tempting to bump this up for the
    # bigger 1000-sample Phase 2 dataset, but Phase 1 already tested that
    # exact tradeoff (1024 vs 768 on the 350-sample set) and found the
    # bigger vocab spread training signal too thin and produced worse,
    # more garbled output -- see prepare_data.py's VOCAB_SIZE comment.
    # Left unchanged here since scaling the *data* is the change being
    # tested this round, not vocab_size; if you do want to test a bigger
    # vocab on the 1000-sample set, change this AND prepare_data.py's
    # VOCAB_SIZE together (they're two separate constants that must
    # match), regenerate data, and compare output quality directly
    # rather than assuming bigger is better.
    vocab_size: int = 768
    max_seq_len: int = 128
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    ffn_hidden: int = 768
    dropout: float = 0.1

    # Special tokens
    pad_id: int = 0
    bos_id: int = 1           # <|im_start|>
    eos_id: int = 2           # <|im_end|>


@dataclass
class TrainConfig:
    # Phase 1 (110 -> 350 samples) initially tuned max_steps down to 500,
    # reasoning that eval loss (on a 17-sample eval set) bottomed out
    # around step 300. That was the wrong signal to optimize: eval loss
    # rising just means the model is memorizing rather than generalizing,
    # which is *expected and fine* for a small hand-curated dataset like
    # this one (see README Dataset section) -- the actual goal is
    # accurate recall, not held-out generalization. Stopping at step
    # 300-500 left train loss around 2.2-2.5 (nowhere near converged) and
    # produced genuinely broken output ("i only really, especially than
    # about milkshakes."). Retrain to real convergence instead (train
    # loss plateaus around 0.3-0.5) and just accept eval loss rising —
    # that's the signal working as intended for this dataset.
    #
    # Phase 2 (350 -> 1000 samples): scaled max_steps/warmup_steps up
    # roughly proportionally with the ~3x data increase, so the model
    # still gets a comparable number of effective passes over the data
    # before considering it converged. Training this long in one sitting
    # is slow -- use train.py's --resume/--extra-steps flags to split it
    # across multiple sessions if needed.
    batch_size: int = 8
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    warmup_steps: int = 200
    max_steps: int = 5300
    eval_interval: int = 250
    save_interval: int = 400
    grad_clip: float = 1.0
    device: str = "auto"
    seed: int = 42
    data_dir: str = "data"
    output_dir: str = "checkpoints"
