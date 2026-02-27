"""
serve_dashboard.py
------------------
Start Python's built-in HTTP server to serve the dashboard files.

Can run in foreground (default) or launch a detached background process
that survives shell session termination on Windows.

Usage:
    python serve_dashboard.py              # foreground (Ctrl+C to stop)
    python serve_dashboard.py --background # detached background process
    python serve_dashboard.py --stop       # kill background server
"""

import argparse
import http.server
import os
import signal
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
PID_FILE = PROJECT_ROOT / "logs" / "dashboard_server.pid"
PORT = 8080


def start_foreground():
    os.chdir(PROJECT_ROOT)
    print(f"Serving BlueprintLLM dashboard on http://localhost:{PORT}")
    print(f"  Live Monitor:  http://localhost:{PORT}/dashboard/live.html")
    print(f"  Training Hub:  http://localhost:{PORT}/dashboard/index.html")
    print(f"  Flowchart:     http://localhost:{PORT}/dashboard/flowchart.html")
    print(f"\nPress Ctrl+C to stop.\n")

    handler = http.server.SimpleHTTPRequestHandler
    with http.server.HTTPServer(("", PORT), handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")


def start_background():
    # Check if already running
    if PID_FILE.exists():
        pid = int(PID_FILE.read_text().strip())
        try:
            os.kill(pid, 0)
            print(f"Dashboard server already running (PID {pid})")
            return
        except OSError:
            PID_FILE.unlink(missing_ok=True)

    CREATE_NO_WINDOW = 0x08000000
    log_path = PROJECT_ROOT / "logs" / "dashboard_server.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = open(log_path, "w")

    proc = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve())],
        cwd=str(PROJECT_ROOT),
        stdout=log,
        stderr=subprocess.STDOUT,
        creationflags=CREATE_NO_WINDOW,
        close_fds=True,
    )

    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(proc.pid))
    print(f"Dashboard server started in background (PID {proc.pid})")
    print(f"  http://localhost:{PORT}/dashboard/live.html")
    print(f"  Stop with: python serve_dashboard.py --stop")


def stop_server():
    if not PID_FILE.exists():
        print("No dashboard server PID file found.")
        return

    pid = int(PID_FILE.read_text().strip())
    try:
        os.kill(pid, signal.SIGTERM)
        print(f"Dashboard server stopped (PID {pid})")
    except OSError:
        print(f"Process {pid} not running.")
    PID_FILE.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description="BlueprintLLM Dashboard Server")
    parser.add_argument("--background", action="store_true", help="Run as detached background process")
    parser.add_argument("--stop", action="store_true", help="Stop the background server")
    args = parser.parse_args()

    if args.stop:
        stop_server()
    elif args.background:
        start_background()
    else:
        start_foreground()


if __name__ == "__main__":
    main()
