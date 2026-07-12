---
license: mit
language:
  - en
tags:
  - character
  - tiny-llm
  - text-generation
  - from-scratch
pipeline_tag: text-generation
---

<p align="center">
  <img src="assets/chef.png" alt="ChefLM" width="300"/>
</p>

<p align="center">
  <a href="https://github.com/BT-Rajan/chefLM"><img src="https://img.shields.io/badge/GitHub-chefLM-181717?logo=github" alt="GitHub"/></a>&nbsp;
  <a href="https://colab.research.google.com/github/BT-Rajan/chefLM/blob/main/use_chef.ipynb"><img src="https://img.shields.io/badge/Open_in-Colab-F9AB00?logo=googlecolab" alt="Colab"/></a>
  <br/><br/>
  <a href="https://bt-rajan.github.io/chef-rajan/"><img src="https://img.shields.io/badge/Try_in-Browser-64ffda?style=for-the-badge&logo=webassembly&logoColor=white" alt="Browser Demo"/></a>
</p>

# ChefLM — 7M Parameter Milkshake Chat Model

A ~7M parameter LLM trained from scratch that talks like a milkshake-obsessed chef named Chef.

This project exists to show that training your own language model is not magic. One Colab notebook, a few minutes, and you have a working LLM built from scratch.

## Example

```
You> what is your favorite milkshake flavor
Chef> chocolate. it always wins.

You> how do you make a milkshake
Chef> blend milk and ice cream together until smooth.

You> is a milkshake healthy
Chef> not really, it's a sweet treat, best in moderation.

You> what toppings go well on a milkshake
Chef> whipped cream, a cherry, or chocolate syrup.
```

## Architecture

| | |
|---|---|
| **Parameters** | 7.3M |
| **Type** | Vanilla transformer (from scratch) |
| **Layers** | 6 |
| **Hidden dim** | 384 |
| **Heads** | 6 |
| **FFN** | 768 (ReLU) |
| **Vocab** | 512 (BPE) |
| **Max sequence** | 128 tokens |
| **Norm** | LayerNorm |
| **Position** | Learned embeddings |
| **LM head** | Weight-tied with embeddings |

No GQA, no RoPE, no SwiGLU, no early exit. As simple as it gets.

## Training

- **Data:** 1205 hand-written single-turn conversations across 14 topics (flavor, ingredients, howto, recipe, topping, temperature, ordering, health, nutrition, comparison, opinion, funfact, redirect, banter) — 1059 English + 146 Arabic, bilingual via a `<|lang_en|>`/`<|lang_ar|>` tag that forces the reply language independent of the prompt's own script
- **Steps:** 800
- **Optimizer:** AdamW (cosine LR schedule)
- **Hardware:** CPU (~3 min), faster on a T4 GPU
- **No system prompt** — personality is baked into the weights

Note: with 1059 English training samples, the model reliably reproduced trained answers
(verified 23/25 on a random exact-match test, including the `banter` personality category)
but recipe questions were sensitive to exact phrasing per flavor, and densely-packed
categories (`health`, `comparison`, `opinion`) occasionally blended answers even on exact
matches. That figure predates the bilingual retrain (146 Arabic samples added, vocab_size
768 -> 1536) — re-run the exact-match check after training on the new data rather than
assuming it still holds unchanged. See the README's Dataset section for details.

## Usage

```python
from inference import ChefInference

engine = ChefInference('checkpoints/best_model.pt', 'data/tokenizer.json')
r = engine.chat_completion([{'role': 'user', 'content': 'what is your favorite milkshake flavor'}])
print(r['choices'][0]['message']['content'])
# chocolate. it always wins.
```

## Links

- **Repo:** [github.com/BT-Rajan/chefLM](https://github.com/BT-Rajan/chefLM)

## License

MIT
