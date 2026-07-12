"""Prepare training data for ChefLM."""

import json
import os
import random

random.seed(42)

DATA_DIR = "data"
# Phase 1: 350-sample milkshake dataset (up from the original 110). Tried
# 1024 first (2x the original 512, matching the ~3.2x growth in sample
# count) but that badly overfit/underfit in practice — with only ~30K
# characters of text, a 1024-slot BPE vocab spreads training signal too
# thin across rare subword tokens and produces noticeably more garbled
# output than the smaller vocab. 768 is the empirically better tradeoff:
# still bigger than the original 512 to cover Phase 1's larger word list,
# without diluting per-token training signal as much.
#
# Phase 4 (English/Arabic bilingual): bumped to 1536. This is a genuinely
# different situation from the 768-vs-1024 English-only experiment above —
# we're not just adding more English data, we're adding an entire second
# script (Arabic) that shares zero subword units with the Latin-alphabet
# merges the old 768-token vocab learned. Keeping 768 would force Arabic
# text through mostly single/double-byte fallback tokens (no learned
# merges for it at all), which both bloats sequence length past
# max_seq_len's budget and gives the model far less signal per Arabic
# token. 1536 is a starting point, not a proven-optimal number the way 768
# was for Phase 1 — if generation quality suffers on either language after
# training, that's the first knob to revisit (alongside dataset size, per
# the same logic as the note above).
VOCAB_SIZE = 1536

SPECIAL_TOKENS = [
    "<pad>",         # 0
    "<|im_start|>",  # 1
    "<|im_end|>",    # 2
    "<|lang_en|>",   # 3 — forces an English reply, placed right after the
                     #     assistant preamble (see data_utils.format_sample).
    "<|lang_ar|>",   # 4 — forces an Arabic reply. Both tags condition the
                     #     *output* language directly, independent of
                     #     whatever script the user's message happened to
                     #     be in — that's what lets a UI toggle reliably
                     #     force one language regardless of input.
]


def train_tokenizer(texts, save_path, vocab_size=VOCAB_SIZE):
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
        min_frequency=2,
    )

    print(f"Training BPE tokenizer (vocab_size={vocab_size}) on {len(texts)} texts...")
    tokenizer.train_from_iterator(texts, trainer)
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    tokenizer.save(save_path)
    print(f"Tokenizer saved to {save_path} ({tokenizer.get_vocab_size()} tokens)")
    return tokenizer


def prepare(data_dir=DATA_DIR, n_samples=350, eval_ratio=0.05):
    os.makedirs(data_dir, exist_ok=True)

    # 1. Generate data
    # NOTE: wired to the milkshake experiment dataset (350 hand-written
    # samples as of Phase 1). The original fish dataset generator
    # (generate_data.py) was removed as orphaned code once this dataset
    # replaced it.
    print(f"Generating {n_samples} samples...")
    from .milkshake_data import generate_dataset
    generate_dataset(n_samples, eval_ratio)

    # 2. Read back all samples for tokenizer training
    texts = []
    for name in ["data/train.jsonl", "data/eval.jsonl"]:
        if os.path.exists(name):
            with open(name) as f:
                for line in f:
                    texts.append(json.loads(line)["text"])

    # 3. Train tokenizer
    tokenizer_path = os.path.join(data_dir, "tokenizer.json")
    tokenizer = train_tokenizer(texts, tokenizer_path)

    # Quick test
    test = "<|im_start|>user\nwhat is your favorite milkshake<|im_end|>"
    ids = tokenizer.encode(test).ids
    decoded = tokenizer.decode(ids)
    print(f"\nTokenizer test:")
    print(f"  Input:   {test}")
    print(f"  Tokens:  {len(ids)} ids")
    print(f"  Decoded: {decoded}")

    # Language-tag test — confirms <|lang_en|>/<|lang_ar|> round-trip as
    # single special tokens (ids 3/4) rather than getting shredded by BPE,
    # and that real Arabic text survives encode/decode intact.
    for tag, sample in [
        ("<|lang_en|>", "hi"),
        ("<|lang_ar|>", "مرحبا"),
    ]:
        lang_test = f"<|im_start|>assistant\n{tag}{sample}<|im_end|>"
        lang_ids = tokenizer.encode(lang_test).ids
        lang_decoded = tokenizer.decode(lang_ids, skip_special_tokens=False)
        ok = lang_decoded.strip() == lang_test.strip()
        print(f"  Lang tag test ({tag}): {'PASS' if ok else 'FAIL'} — decoded: {lang_decoded!r}")


if __name__ == "__main__":
    prepare()
