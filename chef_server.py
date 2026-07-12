"""
chef_server.py — tiny local web UI for ChefLM.

Wraps chef.inference.ChefInference in a minimal Flask app so the model can
be used from a browser instead of the terminal. Serves a single-page chat
UI (static/index.html) with an English/Arabic toggle, and a JSON endpoint
the page calls into.

Not part of the original chef/ package on purpose — keeps this UI layer
separate from the core CLI tool, so `python -m chef chat` still works
exactly as before.
"""

import argparse
import os

from flask import Flask, jsonify, request, send_from_directory

from chef.inference import ChefInference

app = Flask(__name__, static_folder="static", static_url_path="")
engine = None  # set in main()
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_K = 50


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True) or {}
    message = (data.get("message") or "").strip()
    lang = data.get("lang", "en")
    if lang not in ("en", "ar"):
        lang = "en"
    if not message:
        return jsonify({"error": "empty message"}), 400

    temperature = float(data.get("temperature", DEFAULT_TEMPERATURE))
    top_k = int(data.get("top_k", DEFAULT_TOP_K))

    result = engine.chat_completion(
        [{"role": "user", "content": message}],
        check_grammar=True,
        lang=lang,
        temperature=temperature,
        top_k=top_k,
    )
    reply = result["choices"][0]["message"]["content"]
    return jsonify({"reply": reply})


def main():
    global engine
    p = argparse.ArgumentParser(description="Local web UI server for ChefLM")
    p.add_argument("--checkpoint", default="checkpoints/final_model.pt")
    p.add_argument("--tokenizer", default="data/tokenizer.json")
    p.add_argument("--device", default="cpu")
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--lang", default="en", choices=["en", "ar"],
                    help="Default reply language when the page first loads")
    p.add_argument("--temperature", type=float, default=0.7,
                    help="Default sampling temperature (default 0.7). Lower values (0.2-0.4) "
                         "make replies more deterministic and less likely to drift off-topic.")
    p.add_argument("--top-k", type=int, default=50,
                    help="Default top-k sampling cutoff (default 50). Lower (5-10) also pushes "
                         "toward more deterministic replies.")
    args = p.parse_args()

    global DEFAULT_TEMPERATURE, DEFAULT_TOP_K
    DEFAULT_TEMPERATURE = args.temperature
    DEFAULT_TOP_K = args.top_k

    print("Loading ChefLM...")
    engine = ChefInference(args.checkpoint, args.tokenizer, args.device)

    # Stash the default language in an env var so index.html's inline
    # script can read it without a template engine.
    os.environ["CHEF_DEFAULT_LANG"] = args.lang

    print(f"ChefLM web UI ready at http://127.0.0.1:{args.port}")
    app.run(host="127.0.0.1", port=args.port, debug=False)


@app.route("/api/default-lang")
def default_lang():
    return jsonify({"lang": os.environ.get("CHEF_DEFAULT_LANG", "en")})


if __name__ == "__main__":
    main()
