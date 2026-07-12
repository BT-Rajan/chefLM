"""ChefLM inference — simple chat."""

import json
import os
import time
import uuid

import torch
from tokenizers import Tokenizer

from .config import ChefConfig
from .model import ChefLM
from .grammar import correct_grammar
from .persona import Persona, NONE_PERSONA
from .guardrails import is_on_topic, looks_degenerate, best_match, FALLBACK


class ChefInference:
    def __init__(self, checkpoint_path, tokenizer_path, device="cpu"):
        self.device = torch.device(device)
        self.tokenizer = Tokenizer.from_file(tokenizer_path)

        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # Load config.json from same directory as the model file
        config_dir = os.path.dirname(os.path.abspath(checkpoint_path))
        config_path = os.path.join(config_dir, "config.json")

        # Extract state_dict — handle both legacy and standard formats
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            state_dict = ckpt

        # Load config — try config.json first, fall back to embedded config
        if os.path.exists(config_path):
            with open(config_path) as f:
                cfg = json.load(f)
            # train.py writes {"model": {...}, "train": {...}}; unwrap that
            # nested structure if present instead of silently falling back
            # to defaults (which used to happen unconditionally here).
            if "model" in cfg and isinstance(cfg["model"], dict):
                cfg = cfg["model"]
            # Support both HF standard keys and our own keys
            self.config = ChefConfig(
                vocab_size=cfg.get("vocab_size", 4096),
                max_seq_len=cfg.get("max_position_embeddings", cfg.get("max_seq_len", 128)),
                d_model=cfg.get("hidden_size", cfg.get("d_model", 384)),
                n_layers=cfg.get("num_hidden_layers", cfg.get("n_layers", 6)),
                n_heads=cfg.get("num_attention_heads", cfg.get("n_heads", 6)),
                ffn_hidden=cfg.get("intermediate_size", cfg.get("ffn_hidden", 768)),
                dropout=cfg.get("hidden_dropout_prob", cfg.get("dropout", 0.1)),
                pad_id=cfg.get("pad_token_id", cfg.get("pad_id", 0)),
                bos_id=cfg.get("bos_token_id", cfg.get("bos_id", 1)),
                eos_id=cfg.get("eos_token_id", cfg.get("eos_id", 2)),
            )
        elif isinstance(ckpt, dict) and "config" in ckpt:
            valid_fields = {f.name for f in ChefConfig.__dataclass_fields__.values()}
            self.config = ChefConfig(**{k: v for k, v in ckpt["config"].items() if k in valid_fields})
        else:
            print("Warning: No config found, using defaults")
            self.config = ChefConfig()

        self.model = ChefLM(self.config).to(self.device)
        filtered = {k: v for k, v in state_dict.items() if k in self.model.state_dict()}
        self.model.load_state_dict(filtered)
        self.model.eval()

        total, _ = self.model.param_count()
        print(f"ChefLM loaded: {total/1e6:.1f}M params")

    def chat_completion(self, messages, temperature=0.4, max_tokens=64,
                        top_k=10, check_grammar=True, persona=None, lang="en",
                        use_guardrails=True, topic_threshold=1,
                        retrieval_threshold=0.6, **kwargs):
        """Chat completion — takes messages, returns response.

        lang: "en" or "ar". Forces the reply language by feeding the
        matching <|lang_en|>/<|lang_ar|> tag as the last token of the
        prompt (see data_utils.format_sample for the training-side half of
        this) — the model was trained to always continue that tag with
        text in the matching language, so this works regardless of what
        script the user's own message was written in. Also selects which
        LanguageTool language grammar-check runs (see grammar.py).
        check_grammar: run the reply through LanguageTool before returning
        (see grammar.py). Set False to skip it (e.g. for fast eval loops).
        persona: optional persona.Persona instance, applied AFTER grammar
        check (see persona.py) or after a retrieval hit. Rewrites already-
        generated (or retrieved) text — never sent to the model, never
        changes what it computes. Leave None (or pass persona.NONE_PERSONA)
        for the unmodified output. Persona rewrites are English-specific
        word swaps; they harmlessly no-op on Arabic output rather than
        corrupting it.
        use_guardrails: (see guardrails.py) try a direct retrieval match
        first, skip generation entirely for messages that don't resemble
        anything in the training data, and swap in a fallback if
        generation still comes out empty/looping/malformed. Set False to
        get the raw, unguarded model behavior (e.g. for eval scripts that
        want to measure it directly).
        topic_threshold: passed to guardrails.is_on_topic — minimum number
        of domain-vocabulary words (see guardrails.py) the message needs
        to contain before it's treated as on-topic. Default 1.
        retrieval_threshold: passed to guardrails.best_match — minimum
        word-overlap similarity (0-1) to a training question before its
        stored answer is returned directly instead of generating. Default
        0.6. Lower it to retrieve more aggressively (faster, more literal,
        more brittle to paraphrasing); raise it to lean on generation more.
        """
        if use_guardrails:
            last_user = next(
                (m.get("content", "") for m in reversed(messages) if m.get("role") == "user"),
                "",
            )

            retrieved, _score = best_match(last_user, lang=lang, min_similarity=retrieval_threshold)
            if retrieved is not None:
                resp_text = retrieved
                if persona is not None and persona is not NONE_PERSONA:
                    resp_text = persona.apply(resp_text)
                return {
                    "choices": [{"message": {"role": "assistant", "content": resp_text}}],
                    "guardrail": "retrieved",
                }

            if not is_on_topic(last_user, lang=lang, threshold=topic_threshold):
                return {
                    "choices": [{"message": {"role": "assistant", "content": FALLBACK.get(lang, FALLBACK["en"])}}],
                    "guardrail": "off_topic",
                }

        prompt = self._format_prompt(messages, lang=lang)
        input_ids = self.tokenizer.encode(prompt).ids
        prompt_tokens = len(input_ids)
        input_t = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        output_t, _ = self.model.generate(input_t, max_tokens, temperature, top_k)
        output_text = self.tokenizer.decode(output_t[0].tolist()[prompt_tokens:])
        # Truncate at first <|im_end|> — don't let the model leak into the next turn
        if "<|im_end|>" in output_text:
            output_text = output_text.split("<|im_end|>")[0]
        # Also strip any <|im_start|> fragments
        if "<|im_start|>" in output_text:
            output_text = output_text.split("<|im_start|>")[0]
        resp_text = output_text.strip()

        if use_guardrails and looks_degenerate(resp_text):
            # Empty / looping / leaked-special-token output — don't bother
            # grammar-checking or persona-styling garbage, just fall back.
            return {
                "choices": [{"message": {"role": "assistant", "content": FALLBACK.get(lang, FALLBACK["en"])}}],
                "guardrail": "degenerate_output",
            }

        if check_grammar:
            resp_text = correct_grammar(resp_text, lang=lang)

        if persona is not None and persona is not NONE_PERSONA:
            resp_text = persona.apply(resp_text)

        return {
            "choices": [{
                "message": {"role": "assistant", "content": resp_text},
            }],
        }

    def _format_prompt(self, messages, lang="en"):
        lang_tag = "<|lang_ar|>" if lang == "ar" else "<|lang_en|>"
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content") or ""
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        parts.append(f"<|im_start|>assistant\n{lang_tag}")
        return "\n".join(parts)


def main():
    import argparse
    p = argparse.ArgumentParser(description="Chat with Chef")
    p.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    p.add_argument("--tokenizer", default="data/tokenizer.json")
    p.add_argument("--device", default="cpu")
    p.add_argument("--prompt", "-p", help="Single prompt mode: ask one question and exit")
    p.add_argument("--lang", default="en", choices=["en", "ar"],
                    help="Reply language: en or ar (default: en). Forces the reply "
                         "language regardless of what script the prompt itself is in.")
    p.add_argument("--no-grammar-check", action="store_true",
                    help="Skip LanguageTool grammar correction (faster, no Java/network needed)")
    p.add_argument("--no-guardrails", action="store_true",
                    help="Disable the topic gate and output sanity check (see guardrails.py) "
                         "and get the model's raw, unguarded output.")
    p.add_argument("--persona", default="none", choices=["none", "indian"],
                    help="Output-side persona layer (see persona.py). Default: none.")
    p.add_argument("--persona-intensity", type=float, default=0.4,
                    help="0.0-1.0, how often the persona layer tags/rewrites a sentence (default 0.4)")
    p.add_argument("--temperature", type=float, default=0.4,
                    help="Sampling temperature (default 0.4). Lower (e.g. 0.2-0.4) makes replies "
                         "more deterministic and more likely to match the closest trained example "
                         "instead of drifting to a topically-similar-but-wrong answer.")
    p.add_argument("--top-k", type=int, default=10,
                    help="Top-k sampling cutoff (default 10). Lower values (e.g. 5-10) also push "
                         "toward more deterministic, closer-to-training-data replies.")
    args = p.parse_args()

    engine = ChefInference(args.checkpoint, args.tokenizer, args.device)
    check_grammar = not args.no_grammar_check
    persona = Persona(name=args.persona, intensity=args.persona_intensity) if args.persona != "none" else None

    if args.prompt:
        result = engine.chat_completion([{"role": "user", "content": args.prompt}],
                                         check_grammar=check_grammar, persona=persona, lang=args.lang,
                                         temperature=args.temperature, top_k=args.top_k,
                                         use_guardrails=not args.no_guardrails)
        print(result["choices"][0]["message"]["content"])
        return

    print("\nChef Chat (type 'quit' to exit)")
    while True:
        inp = input("\nYou> ").strip()
        if inp.lower() in ("quit", "exit", "q"):
            break
        result = engine.chat_completion([{"role": "user", "content": inp}],
                                         check_grammar=check_grammar, persona=persona, lang=args.lang,
                                         temperature=args.temperature, top_k=args.top_k,
                                         use_guardrails=not args.no_guardrails)
        msg = result["choices"][0]["message"]
        if msg.get("content"):
            print(f"Chef> {msg['content']}")


if __name__ == "__main__":
    main()
