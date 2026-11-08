"""Start the backend and the dashboard together with one command.

    uv run run.py                # start both and open the dashboard in the browser
    uv run run.py --no-browser   # don't open the browser
    uv run run.py --reload       # restart the backend when Python files change

Settings are read from .env in the project root (copied from .env.example on
the first run). Press Ctrl+C to stop both.
"""

import argparse
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
UI_DIR = ROOT / "ui"
BACKEND_PORT = 8000
UI_PORT = 5173
BACKEND_URL = f"http://127.0.0.1:{BACKEND_PORT}"
UI_URL = f"http://localhost:{UI_PORT}"
WINDOWS = os.name == "nt"


def log(message):
    print(f"[run] {message}", flush=True)


def fail(message):
    log(f"ERROR: {message}")
    sys.exit(1)


def port_in_use(port):
    for family, address in ((socket.AF_INET, "127.0.0.1"), (socket.AF_INET6, "::1")):
        try:
            with socket.socket(family, socket.SOCK_STREAM) as sock:
                sock.settimeout(0.5)
                if sock.connect_ex((address, port)) == 0:
                    return True
        except OSError:
            pass
    return False


def responds(url):
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            return response.status < 500
    except Exception:
        return False


def check_setup():
    env_file, example = ROOT / ".env", ROOT / ".env.example"
    if not env_file.exists() and example.exists():
        shutil.copyfile(example, env_file)
        log("Created .env from .env.example. Edit it to change the camera or location.")

    sys.path.insert(0, str(ROOT / "src"))
    from backend.config import Settings

    model_path = Settings().model_path
    if not model_path.is_file():
        fail(f"Pose pipeline config is missing: {model_path}")

    check_database()

    node = shutil.which("node")
    if not node:
        fail("Node.js was not found. Install it from https://nodejs.org and open a new terminal.")
    if not (UI_DIR / "node_modules" / "vite").exists():
        npm = shutil.which("npm")
        if not npm:
            fail("npm was not found. It comes with Node.js: https://nodejs.org")
        log("Installing dashboard packages (first run only)...")
        if subprocess.run([npm, "install"], cwd=UI_DIR).returncode != 0:
            fail("npm install failed; see the output above.")

    for port, name in ((BACKEND_PORT, "backend"), (UI_PORT, "dashboard")):
        if port_in_use(port):
            fail(
                f"Port {port} ({name}) is already in use. Another copy is probably "
                "running; stop it first (close its terminal or end it in Task Manager)."
            )
    return node


def check_database():
    """Fail early, with a hint, if PostgreSQL can't be reached with the .env settings."""
    sys.path.insert(0, str(ROOT / "src"))
    from sqlalchemy.engine import make_url

    from backend.config import database_url_from_env  # reads .env
    from backend.db import create_db_engine

    url = database_url_from_env()
    engine = create_db_engine(url)
    try:
        with engine.connect():
            pass
    except Exception as error:
        reason = str(getattr(error, "orig", error)).strip().splitlines()[0]
        where = make_url(url).render_as_string(hide_password=True)
        fail(
            f"Could not connect to the database at {where}: {reason}\n"
            "      Check FALL_DATABASE_URL in .env (the team's Supabase connection) and the\n"
            "      internet connection. See docs/database.md."
        )
    finally:
        engine.dispose()


def start(name, command, cwd):
    """Start a process and copy its output to ours, prefixed with its name."""
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    process = subprocess.Popen(
        command, cwd=cwd, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
    )

    def pipe():
        for line in process.stdout:
            print(f"[{name}] {line}", end="", flush=True)

    threading.Thread(target=pipe, daemon=True).start()
    return process


def stop(processes, interrupted, timeout=10):
    """Ask each process to exit, then force it after `timeout` seconds.

    After Ctrl+C the children received it too (they share our console), so
    they only need time to shut down cleanly. Otherwise they must be told:
    with SIGINT on macOS/Linux, or ended right away on Windows, which has no
    equivalent signal for a process in the same console.
    """
    if not interrupted:
        for process in processes:
            if process.poll() is None:
                if WINDOWS:
                    _kill_tree(process)
                else:
                    process.send_signal(signal.SIGINT)
    deadline = time.time() + timeout
    for process in processes:
        try:
            process.wait(max(0.1, deadline - time.time()))
        except subprocess.TimeoutExpired:
            pass
        if process.poll() is None:
            if WINDOWS:
                _kill_tree(process)
            else:
                process.kill()


def _kill_tree(process):
    # /T also ends child processes (e.g. uvicorn's --reload worker).
    subprocess.run(
        ["taskkill", "/T", "/F", "/PID", str(process.pid)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def wait_until_ready(name, url, processes, timeout):
    deadline = time.time() + timeout
    while time.time() < deadline:
        for process in processes:
            if process.poll() is not None:
                raise RuntimeError(f"a process exited while starting the {name}; see the output above")
        if responds(url):
            return
        time.sleep(0.5)
    raise RuntimeError(f"the {name} did not start within {timeout} s")


def main():
    parser = argparse.ArgumentParser(description="Start the backend and the dashboard.")
    parser.add_argument("--no-browser", action="store_true", help="don't open the browser")
    parser.add_argument("--reload", action="store_true", help="restart the backend on code changes")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    node = check_setup()

    backend_command = [
        sys.executable, "-m", "uvicorn", "backend.main:app",
        "--app-dir", "src", "--host", "127.0.0.1", "--port", str(BACKEND_PORT),
    ]
    if args.reload:
        backend_command += ["--reload", "--reload-dir", "src"]
    ui_command = [node, "node_modules/vite/bin/vite.js", "--port", str(UI_PORT), "--strictPort"]

    processes = []
    interrupted = False
    try:
        log("Starting backend...")
        processes.append(start("backend", backend_command, ROOT))
        wait_until_ready("backend", f"{BACKEND_URL}/api/health", processes, timeout=120)

        log("Starting dashboard...")
        processes.append(start("dashboard", ui_command, UI_DIR))
        wait_until_ready("dashboard", UI_URL, processes, timeout=60)

        log("")
        log(f"Dashboard:  {UI_URL}")
        log(f"API docs:   {BACKEND_URL}/docs")
        log("Press Ctrl+C to stop both.")
        log("")
        if not args.no_browser:
            webbrowser.open(UI_URL)

        # Run until Ctrl+C, or until either process stops on its own.
        while all(process.poll() is None for process in processes):
            time.sleep(0.5)
        log("A process stopped unexpectedly; shutting down the other one.")
        return 1
    except KeyboardInterrupt:
        interrupted = True
        log("Stopping...")
        return 0
    except RuntimeError as error:
        log(f"ERROR: {error}")
        return 1
    finally:
        stop(processes, interrupted)
        log("Stopped.")


if __name__ == "__main__":
    sys.exit(main())
