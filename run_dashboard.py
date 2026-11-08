"""Launch worker before the web server, independently of browser/page sessions."""
import os
import signal
import subprocess
import sys
import time


def main():
    worker = subprocess.Popen([sys.executable, "-m", "research_worker"])
    web = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app.py",
                            "--server.port", "5000", "--server.address", "0.0.0.0"])
    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True
        for child in (worker, web):
            if child.poll() is None:
                child.terminate()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        while not stopping and web.poll() is None:
            if worker.poll() is not None:
                worker = subprocess.Popen([sys.executable, "-m", "research_worker"])
            time.sleep(1)
    finally:
        stop()
        for child in (worker, web):
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
    return web.returncode or 0


if __name__ == "__main__":
    raise SystemExit(main())
