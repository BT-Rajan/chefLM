"""
installer.py — interactive browser-based installer/launcher for ChefLM.

Deliberately uses ONLY the Python standard library (http.server, threading,
subprocess, webbrowser, json, socket). This has to run *before* venv/pip/
flask exist, so it can't depend on any of them. Once setup is done, it
hands off to chef_server.py (which does use Flask) as a child process and
the installer page redirects the browser there.

Flow:
  1. Starts a tiny HTTP server on INSTALLER_PORT, opens the browser to it.
  2. Serves static/installer.html - a page with a language picker and an
     "Install & Launch" button, plus a live progress/log view.
  3. On "start", runs setup (venv, pip installs, prepare, train - each
     skipped if already done) in a background thread, streaming step
     status + log lines back to the page via polling.
  4. Once done, launches chef_server.py in the venv as a child process and
     tells the page to redirect to it.
"""

import json
import os
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

WEBUI_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(WEBUI_DIR)  # webui/ lives one level below the repo root
INSTALLER_PORT = 8000
CHAT_PORT = 5000

IS_WINDOWS = os.name == "nt"
VENV_DIR = os.path.join(REPO_ROOT, "venv")
VENV_PY = os.path.join(VENV_DIR, "Scripts" if IS_WINDOWS else "bin",
                        "python.exe" if IS_WINDOWS else "python")

STEPS = [
    ("venv", "Setting up Python environment"),
    ("deps", "Installing dependencies"),
    ("grammar", "Installing grammar-check support"),
    ("data", "Preparing dataset and tokenizer"),
    ("train", "Training the model"),
    ("launch", "Starting chat server"),
]

# ---------------------------------------------------------------------------
# Shared state, guarded by a lock. Polled by the browser via /api/status.
# ---------------------------------------------------------------------------
state_lock = threading.Lock()
STATE = {
    "started": False,
    "current_step": None,      # step key currently running
    "step_status": {k: "pending" for k, _ in STEPS},  # pending|running|done|error
    "log": [],                 # rolling list of log lines
    "error": None,
    "ready": False,            # chat server is up
    "chat_url": f"http://127.0.0.1:{CHAT_PORT}",
}


def log(line):
    with state_lock:
        STATE["log"].append(line)
        STATE["log"] = STATE["log"][-400:]  # keep it bounded
    print(line, flush=True)


def set_step(key, status):
    with state_lock:
        STATE["current_step"] = key
        STATE["step_status"][key] = status


def run_cmd(cmd, cwd=REPO_ROOT):
    """Run a command, streaming its output into the shared log. Returns True on success."""
    log(f"$ {' '.join(cmd)}")
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    proc = subprocess.Popen(
        cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, encoding="utf-8", errors="replace", env=env,
    )
    for line in proc.stdout:
        log(line.rstrip())
    proc.wait()
    return proc.returncode == 0


def port_open(port, host="127.0.0.1"):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def run_setup(lang):
    try:
        # 1. venv
        set_step("venv", "running")
        if not os.path.exists(VENV_PY):
            if not run_cmd([sys.executable, "-m", "venv", "venv"]):
                raise RuntimeError("Failed to create virtual environment.")
        else:
            log("Virtual environment already exists, skipping.")
        set_step("venv", "done")

        # 2. dependencies
        set_step("deps", "running")
        run_cmd([VENV_PY, "-m", "pip", "install", "--upgrade", "pip"])
        req = os.path.join(REPO_ROOT, "requirements.txt")
        if os.path.exists(req):
            if not run_cmd([VENV_PY, "-m", "pip", "install", "-r", "requirements.txt"]):
                raise RuntimeError("pip install -r requirements.txt failed.")
        if not run_cmd([VENV_PY, "-m", "pip", "install", "flask"]):
            raise RuntimeError("Failed to install flask.")
        set_step("deps", "done")

        # 3. grammar support (best-effort, never fatal)
        set_step("grammar", "running")
        ok = run_cmd([VENV_PY, "-m", "pip", "install", "language-tool-python"])
        if ok:
            log("language-tool-python installed. Note: grammar checking also "
                "needs a Java runtime on PATH (https://adoptium.net) - if it's "
                "missing, chat still works, just without grammar correction.")
        else:
            log("Could not install language-tool-python - chat will still "
                "work, just without grammar correction.")
        set_step("grammar", "done")

        # 4. prepare data
        set_step("data", "running")
        tokenizer_path = os.path.join(REPO_ROOT, "data", "tokenizer.json")
        if not os.path.exists(tokenizer_path):
            if not run_cmd([VENV_PY, "-m", "chef", "prepare"]):
                raise RuntimeError("Data preparation failed.")
        else:
            log("Prepared data already found, skipping.")
        set_step("data", "done")

        # 5. train
        set_step("train", "running")
        final_ckpt = os.path.join(REPO_ROOT, "checkpoints", "final_model.pt")
        if not os.path.exists(final_ckpt):
            log("No trained checkpoint found - training from scratch "
                "(runs on CPU, a few minutes).")
            if not run_cmd([VENV_PY, "-m", "chef", "train"]):
                raise RuntimeError("Training failed.")
        else:
            log("Existing checkpoint found at checkpoints/final_model.pt, skipping training.")
        set_step("train", "done")

        # 6. launch chat server
        set_step("launch", "running")
        log(f"Starting chat server on port {CHAT_PORT} (lang={lang})...")
        subprocess.Popen(
            [VENV_PY, os.path.join(WEBUI_DIR, "chef_server.py"),
             "--checkpoint", "checkpoints/final_model.pt",
             "--lang", lang, "--temperature", "0.4",
             "--port", str(CHAT_PORT)],
            cwd=REPO_ROOT,
        )
        for _ in range(60):
            if port_open(CHAT_PORT):
                break
            time.sleep(0.5)
        else:
            raise RuntimeError("Chat server did not come up in time.")
        set_step("launch", "done")

        with state_lock:
            STATE["ready"] = True
        log("Ready! Redirecting to chat...")

    except Exception as e:
        with state_lock:
            STATE["error"] = str(e)
        if STATE["current_step"]:
            set_step(STATE["current_step"], "error")
        log(f"[ERROR] {e}")


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------
STATIC_DIR = os.path.join(WEBUI_DIR, "static")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # keep the console clean; setup log already prints

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/" or self.path == "/installer.html":
            path = os.path.join(STATIC_DIR, "installer.html")
            self._serve_file(path, "text/html")
        elif self.path == "/api/status":
            with state_lock:
                self._json(dict(STATE))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/api/start":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length) if length else b"{}"
            try:
                payload = json.loads(body or b"{}")
            except json.JSONDecodeError:
                payload = {}
            lang = payload.get("lang", "en")
            if lang not in ("en", "ar"):
                lang = "en"

            with state_lock:
                already_started = STATE["started"]
                if not already_started:
                    STATE["started"] = True

            if not already_started:
                threading.Thread(target=run_setup, args=(lang,), daemon=True).start()

            self._json({"ok": True})
        else:
            self.send_response(404)
            self.end_headers()

    def _serve_file(self, path, content_type):
        try:
            with open(path, "rb") as f:
                body = f.read()
        except OSError:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    server = ThreadingHTTPServer(("127.0.0.1", INSTALLER_PORT), Handler)
    url = f"http://127.0.0.1:{INSTALLER_PORT}"
    print(f"ChefLM installer running at {url}")
    threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
