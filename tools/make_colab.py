"""Generate the ChefLM Colab training notebook."""

import json
import os
import re

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_file(path):
    with open(path) as f:
        return f.read()


def read_for_colab(path):
    """Read a Python file and flatten relative imports for Colab."""
    content = read_file(path)
    content = re.sub(r'from \.(\w+) import', r'from \1 import', content)
    return content


def cell(source, cell_type="code"):
    lines = source.split("\n")
    formatted = [line + "\n" if i < len(lines) - 1 else line for i, line in enumerate(lines)]
    base = {"cell_type": cell_type, "metadata": {}, "source": formatted}
    if cell_type == "code":
        base["outputs"] = []
        base["execution_count"] = None
    return base


def md(text):
    return cell(text, "markdown")


def code(text):
    return cell(text, "code")


# Source files to embed in the notebook
FILES = [
    ("config.py",         "chef/config.py"),
    ("model.py",          "chef/model.py"),
    ("dataset.py",        "chef/dataset.py"),
    ("train.py",          "chef/train.py"),
    ("inference.py",      "chef/inference.py"),
    ("data_utils.py",     "chef/data_utils.py"),
    ("milkshake_data.py", "chef/milkshake_data.py"),
]


def build():
    cells = []

    # ══════════════════════════════════════════════════════════════════
    #  HEADER
    # ══════════════════════════════════════════════════════════════════

    cells.append(md(
        "# ChefLM — Your Friendly Milkshake Chef\n"
        "\n"
        "Train a ~7M parameter LLM that talks like a milkshake-obsessed chef.\n"
        "\n"
        "**What this notebook does:**\n"
        "1. Generates a 100-sample milkshake conversation dataset\n"
        "2. Trains a BPE tokenizer on the data\n"
        "3. Trains a 6-layer vanilla transformer (~7.3M params)\n"
        "4. Tests the model with sample conversations\n"
        "\n"
        "**Architecture:** 6 layers, 384 dim, 6 heads, ReLU FFN, LayerNorm, 512 vocab\n"
        "\n"
        "**Runtime:** ~20-25 min on CPU, faster on a T4 GPU\n"
        "\n"
        "**Result:** A chef that answers simple questions about milkshakes."
    ))

    # ══════════════════════════════════════════════════════════════════
    #  1. SETUP
    # ══════════════════════════════════════════════════════════════════

    cells.append(md(
        "## 1. Setup\n"
        "\n"
        "Install dependencies and create a clean working directory."
    ))

    cells.append(code(
        "!pip install -q torch tokenizers tqdm numpy datasets huggingface_hub\n"
        "\n"
        "import torch\n"
        "print(f'PyTorch {torch.__version__}')\n"
        "print(f'CUDA: {torch.cuda.is_available()}')\n"
        "if torch.cuda.is_available():\n"
        "    print(f'GPU: {torch.cuda.get_device_name(0)}')"
    ))

    cells.append(code(
        "import os, shutil\n"
        "\n"
        "# Start fresh — removes stale files from previous runs\n"
        "if os.path.exists('/content/chef'):\n"
        "    shutil.rmtree('/content/chef')\n"
        "os.makedirs('/content/chef')\n"
        "os.chdir('/content/chef')\n"
        "print(f'Working dir: {os.getcwd()}')"
    ))

    # ══════════════════════════════════════════════════════════════════
    #  2. SOURCE FILES
    # ══════════════════════════════════════════════════════════════════

    cells.append(md(
        "## 2. Source Files\n"
        "\n"
        "Write the model code to disk. These are the only files needed:\n"
        "- `config.py` — model and training hyperparameters\n"
        "- `model.py` — transformer architecture\n"
        "- `dataset.py` — data loading and batching\n"
        "- `train.py` — training loop\n"
        "- `inference.py` — chat interface"
    ))

    for display_name, src_path in FILES:
        full_path = os.path.join(PROJECT_ROOT, src_path)
        content = read_for_colab(full_path)
        cells.append(code(f"%%writefile {display_name}\n{content}"))

    # ══════════════════════════════════════════════════════════════════
    #  3. PREPARE DATA
    # ══════════════════════════════════════════════════════════════════

    cells.append(md(
        "## 3. Prepare Data\n"
        "\n"
        "Generate the milkshake conversation dataset and train a BPE tokenizer.\n"
        "\n"
        "The dataset has 100 hand-written single-turn conversations across 10 topics:\n"
        "flavor, ingredients, howto, topping, temperature, ordering, health, comparison, opinion, funfact.\n"
        "\n"
        "Each sample is formatted as ChatML:\n"
        "```\n"
        "<|im_start|>user\n"
        "what is your favorite milkshake flavor<|im_end|>\n"
        "<|im_start|>assistant\n"
        "chocolate. it always wins.<|im_end|>\n"
        "```"
    ))

    cells.append(code(
        "import json, os\n"
        "from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors\n"
        "from milkshake_data import generate_dataset\n"
        "\n"
        "# ── Generate the milkshake dataset locally (100 hand-written samples) ──\n"
        "generate_dataset()\n"
        "\n"
        "with open('data/train.jsonl') as f:\n"
        "    texts = [json.loads(line)['text'] for line in f]\n"
        "with open('data/eval.jsonl') as f:\n"
        "    texts += [json.loads(line)['text'] for line in f]\n"
        "\n"
        "# ── Train BPE tokenizer on the data ──\n"
        "tokenizer = Tokenizer(models.BPE())\n"
        "tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)\n"
        "tokenizer.decoder = decoders.ByteLevel()\n"
        "\n"
        "trainer = trainers.BpeTrainer(\n"
        "    vocab_size=768,\n"
        "    special_tokens=['<pad>', '<|im_start|>', '<|im_end|>'],\n"
        "    min_frequency=2,\n"
        "    show_progress=True,\n"
        ")\n"
        "tokenizer.train_from_iterator(texts, trainer)\n"
        "tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)\n"
        "tokenizer.save('data/tokenizer.json')\n"
        "print(f'  Tokenizer: {tokenizer.get_vocab_size()} tokens')\n"
        "\n"
        "# ── Preview ──\n"
        "with open('data/train.jsonl') as f:\n"
        "    sample = json.loads(f.readline())\n"
        "print(f'\\nSample ({sample[\"category\"]}):\\n{sample[\"text\"]}')"
    ))

    # ══════════════════════════════════════════════════════════════════
    #  4. VERIFY ARCHITECTURE
    # ══════════════════════════════════════════════════════════════════

    cells.append(md(
        "## 4. Verify Architecture\n"
        "\n"
        "Quick sanity check — build the model, print param count, run a dummy forward pass."
    ))

    cells.append(code(
        "from config import ChefConfig\n"
        "from model import ChefLM\n"
        "import torch\n"
        "\n"
        "config = ChefConfig()\n"
        "model = ChefLM(config)\n"
        "print(model.param_summary())\n"
        "print(f'  Layers: {config.n_layers}, Heads: {config.n_heads}, FFN: {config.ffn_hidden}')\n"
        "print(f'  Vocab: {config.vocab_size}, Max seq: {config.max_seq_len}')\n"
        "\n"
        "# Dummy forward pass\n"
        "x = torch.randint(0, config.vocab_size, (2, 32))\n"
        "logits, _ = model(x)\n"
        "print(f'  Forward pass: {x.shape} -> {logits.shape} OK')\n"
        "del model"
    ))

    # ══════════════════════════════════════════════════════════════════
    #  5. TRAIN
    # ══════════════════════════════════════════════════════════════════

    cells.append(md(
        "## 5. Train\n"
        "\n"
        "5000 steps with cosine LR schedule. Takes ~20-25 min on CPU, faster on a T4. Long enough that you may want the --resume flag if running locally without a GPU.\n"
        "\n"
        "The model learns to:\n"
        "- Respond in short, lowercase sentences\n"
        "- Stay in character as a milkshake-obsessed chef\n"
        "- Cover 10 different conversation topics\n"
        "- Stop generating at the right time (learn the `<|im_end|>` token)"
    ))

    cells.append(code("from train import train\ntrain()"))

    # ══════════════════════════════════════════════════════════════════
    #  6. TEST
    # ══════════════════════════════════════════════════════════════════

    cells.append(md(
        "## 6. Test\n"
        "\n"
        "Chat with the trained model. Each message is independent (single-turn)."
    ))

    cells.append(code(
        "from inference import ChefInference\n"
        "import torch\n"
        "\n"
        "engine = ChefInference(\n"
        "    'checkpoints/best_model.pt', 'data/tokenizer.json',\n"
        "    device='cuda' if torch.cuda.is_available() else 'cpu'\n"
        ")\n"
        "\n"
        "def chat(prompt):\n"
        "    r = engine.chat_completion([{'role': 'user', 'content': prompt}], max_tokens=64)\n"
        "    return r['choices'][0]['message'].get('content', '').strip()\n"
        "\n"
        "# Test across different topics\n"
        "tests = [\n"
        "    ('what is your favorite milkshake flavor', 'flavor'),\n"
        "    ('what ingredients go in a milkshake',      'ingredients'),\n"
        "    ('how do you make a milkshake',              'howto'),\n"
        "    ('what toppings go well on a milkshake',     'topping'),\n"
        "    ('should a milkshake be very cold',          'temperature'),\n"
        "    ('what size milkshake should i order',       'ordering'),\n"
        "    ('is a milkshake healthy',                    'health'),\n"
        "    ('is a frappe the same as a milkshake',      'comparison'),\n"
        "    ('do you think milkshakes are a good dessert', 'opinion'),\n"
        "    ('when were milkshakes invented',            'funfact'),\n"
        "]\n"
        "\n"
        "print(f'{\"Topic\":<12s}  {\"You\":<40s}  Chef')\n"
        "print('=' * 100)\n"
        "for prompt, topic in tests:\n"
        "    reply = chat(prompt)\n"
        "    print(f'{topic:<12s}  {prompt:<40s}  {reply[:128]}')\n"
    ))

    # ══════════════════════════════════════════════════════════════════
    #  7. EXPORT & UPLOAD TO HUGGINGFACE
    # ══════════════════════════════════════════════════════════════════

    cells.append(md(
        "## 7. Export & Upload to HuggingFace\n"
        "\n"
        "Export the model in both PyTorch and ONNX (quantized uint8, ~9 MB) formats,\n"
        "then upload everything to HuggingFace in one go.\n"
        "\n"
        "Set your token and repo below."
    ))

    cells.append(code(
        "!pip install -q onnx onnxruntime onnxscript\n"
        "\n"
        "from huggingface_hub import HfApi, login\n"
        "import torch, json, os, shutil\n"
        "from config import ChefConfig\n"
        "from model import ChefLM\n"
        "\n"
        "HF_TOKEN = os.environ.get('HF_TOKEN', '')  # Or paste your token here\n"
        "HF_REPO = os.environ.get('HF_REPO', 'BT-Rajan/chef-9m')  # placeholder \u2014 create this repo first\n"
        "\n"
        "# Load checkpoint\n"
        "ckpt = torch.load('checkpoints/best_model.pt', map_location='cpu', weights_only=False)\n"
        "cfg = ckpt['config']\n"
        "os.makedirs('hf_export', exist_ok=True)\n"
        "\n"
        "# ── PyTorch format ──\n"
        "torch.save(ckpt['model_state_dict'], 'hf_export/pytorch_model.bin')\n"
        "\n"
        "with open('hf_export/config.json', 'w') as f:\n"
        "    json.dump({\n"
        "        'model_type': 'chef',\n"
        "        'architectures': ['ChefLM'],\n"
        "        'vocab_size': cfg['vocab_size'],\n"
        "        'max_position_embeddings': cfg['max_seq_len'],\n"
        "        'hidden_size': cfg['d_model'],\n"
        "        'num_hidden_layers': cfg['n_layers'],\n"
        "        'num_attention_heads': cfg['n_heads'],\n"
        "        'intermediate_size': cfg['ffn_hidden'],\n"
        "        'hidden_dropout_prob': cfg.get('dropout', 0.1),\n"
        "        'pad_token_id': cfg['pad_id'],\n"
        "        'bos_token_id': cfg['bos_id'],\n"
        "        'eos_token_id': cfg['eos_id'],\n"
        "    }, f, indent=2)\n"
        "\n"
        "shutil.copy('data/tokenizer.json', 'hf_export/tokenizer.json')\n"
        "print(f'pytorch_model.bin: {os.path.getsize(\"hf_export/pytorch_model.bin\")/1e6:.1f} MB')\n"
        "\n"
        "# ── ONNX format (quantized uint8) ──\n"
        "valid_fields = {f.name for f in ChefConfig.__dataclass_fields__.values()}\n"
        "config = ChefConfig(**{k: v for k, v in cfg.items() if k in valid_fields})\n"
        "model = ChefLM(config)\n"
        "model.load_state_dict(ckpt['model_state_dict'])\n"
        "model.eval()\n"
        "\n"
        "dummy = torch.randint(0, config.vocab_size, (1, 32))\n"
        "fp32_path = 'hf_export/model_fp32.onnx'\n"
        "torch.onnx.export(\n"
        "    model, (dummy,), fp32_path,\n"
        "    input_names=['input_ids'], output_names=['logits'],\n"
        "    dynamic_axes={'input_ids': {0: 'batch', 1: 'seq_len'},\n"
        "                  'logits': {0: 'batch', 1: 'seq_len'}},\n"
        "    opset_version=17,\n"
        ")\n"
        "\n"
        "from onnxruntime.quantization import quantize_dynamic, QuantType\n"
        "quantize_dynamic(fp32_path, 'hf_export/model.onnx', weight_type=QuantType.QUInt8)\n"
        "os.remove(fp32_path)\n"
        "print(f'model.onnx:       {os.path.getsize(\"hf_export/model.onnx\")/1e6:.1f} MB (uint8)')\n"
        "\n"
        "# ── Upload ──\n"
        "if HF_TOKEN:\n"
        "    login(token=HF_TOKEN)\n"
        "    api = HfApi()\n"
        "    api.create_repo(HF_REPO, exist_ok=True)\n"
        "    api.upload_folder(folder_path='hf_export', repo_id=HF_REPO, repo_type='model')\n"
        "    print(f'Done! https://huggingface.co/{HF_REPO}')\n"
        "else:\n"
        "    print('No HF_TOKEN — exported locally to hf_export/')"
    ))

    # ══════════════════════════════════════════════════════════════════
    #  8. DOWNLOAD
    # ══════════════════════════════════════════════════════════════════

    cells.append(md(
        "## 8. Download\n"
        "\n"
        "Or download the model locally as a tar.gz."
    ))

    cells.append(code(
        "import os\n"
        "\n"
        "!cd /content && tar czf chef.tar.gz \\\n"
        "    chef/checkpoints/best_model.pt \\\n"
        "    chef/checkpoints/config.json \\\n"
        "    chef/data/tokenizer.json \\\n"
        "    chef/model.py \\\n"
        "    chef/config.py \\\n"
        "    chef/inference.py \\\n"
        "    chef/hf_export/model.onnx\n"
        "\n"
        "sz = os.path.getsize('/content/chef.tar.gz') / 1e6\n"
        "print(f'Package: /content/chef.tar.gz ({sz:.1f} MB)')\n"
        "\n"
        "try:\n"
        "    from google.colab import files\n"
        "    files.download('/content/chef.tar.gz')\n"
        "except ImportError:\n"
        "    print('Not in Colab — download manually from the file browser.')"
    ))

    # ══════════════════════════════════════════════════════════════════

    return {
        "nbformat": 4, "nbformat_minor": 0,
        "metadata": {
            "colab": {"provenance": [], "gpuType": "T4", "name": "ChefLM — Train"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "cells": cells,
    }


def build_use():
    """Build the use_chef notebook — download model from HF and chat."""
    cells = []

    cells.append(md(
        "# ChefLM — Chat with Chef\n"
        "\n"
        "Download a pre-trained ~7M parameter milkshake-chef LLM and chat with it. Just run all cells.\n"
        "\n"
        "**Model:** `BT-Rajan/chef-9m` (placeholder — create and upload this repo first, "
        "or set `HF_REPO` to wherever you've published your trained checkpoint)"
    ))

    cells.append(code(
        "# Setup + Download\n"
        "!pip install -q torch tokenizers huggingface_hub\n"
        "import os, shutil\n"
        "if os.path.exists('/content/chef'): shutil.rmtree('/content/chef')\n"
        "os.makedirs('/content/chef'); os.chdir('/content/chef')\n"
        "\n"
        "HF_REPO = os.environ.get('HF_REPO', 'BT-Rajan/chef-9m')  # placeholder \u2014 create this repo first\n"
        "\n"
        "from huggingface_hub import snapshot_download\n"
        "snapshot_download(repo_id=HF_REPO, local_dir='.')\n"
        "print('Model downloaded.')"
    ))

    cells.append(code(
        "# Load model\n"
        "from inference import ChefInference\n"
        "import torch\n"
        "\n"
        "engine = ChefInference('pytorch_model.bin', 'tokenizer.json',\n"
        "                        device='cuda' if torch.cuda.is_available() else 'cpu')\n"
        "\n"
        "def chat(prompt):\n"
        "    return engine.chat_completion(\n"
        "        [{'role': 'user', 'content': prompt}], max_tokens=64\n"
        "    )['choices'][0]['message'].get('content', '').strip()\n"
        "\n"
        "# Quick test\n"
        "for p in ['what is your favorite milkshake flavor', 'how do you make a milkshake',\n"
        "          'is a milkshake healthy', 'what toppings go well on a milkshake']:\n"
        "    print(f'You> {p}\\nChef> {chat(p)}\\n')"
    ))

    cells.append(code(
        "# Interactive chat — type your messages\n"
        "while True:\n"
        "    try:\n"
        "        p = input('You> ').strip()\n"
        "    except (KeyboardInterrupt, EOFError):\n"
        "        break\n"
        "    if not p or p.lower() in ('quit', 'exit', 'q'):\n"
        "        print(\"Chef> bye. i'll be here whenever you want to talk milkshakes.\"); break\n"
        "    print(f'Chef> {chat(p)}\\n')"
    ))

    return {
        "nbformat": 4, "nbformat_minor": 0,
        "metadata": {
            "colab": {"provenance": [], "name": "ChefLM — Chat"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
        },
        "cells": cells,
    }


def write_notebook(nb, filename):
    out = os.path.join(PROJECT_ROOT, filename)
    with open(out, "w") as f:
        json.dump(nb, f, indent=1)
    n = len(nb["cells"])
    sz = os.path.getsize(out) / 1024
    print(f"Generated {out}: {n} cells, {sz:.1f} KB")


if __name__ == "__main__":
    write_notebook(build(), "train_chef.ipynb")
    write_notebook(build_use(), "use_chef.ipynb")
