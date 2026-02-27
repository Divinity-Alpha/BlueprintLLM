"""Launch the pipeline as a fully detached Windows process.

Also ensures the dashboard server (port 8080) is running so you can
monitor progress at http://localhost:8080/dashboard/live.html
"""
import subprocess
import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
PID_FILE = PROJECT_ROOT / "logs" / "dashboard_server.pid"
DASHBOARD_PORT = 8080


def ensure_dashboard():
    """Start the dashboard server if it isn't already running."""
    if PID_FILE.exists():
        pid = int(PID_FILE.read_text().strip())
        try:
            os.kill(pid, 0)
            print(f"Dashboard server already running (PID {pid})")
            return
        except OSError:
            PID_FILE.unlink(missing_ok=True)

    subprocess.Popen(
        [sys.executable, str(PROJECT_ROOT / "serve_dashboard.py"), "--background"],
        cwd=str(PROJECT_ROOT),
    ).wait()


def main():
    # --- Dashboard ---
    ensure_dashboard()

    # --- Pipeline ---
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = '0'

    # CUDA JIT cache — persist compiled kernels across runs
    cache_dir = str(PROJECT_ROOT / ".cuda_cache")
    os.makedirs(cache_dir, exist_ok=True)
    env.setdefault('CUDA_CACHE_PATH', cache_dir)
    env.setdefault('CUDA_CACHE_MAXSIZE', '4294967296')  # 4 GB

    # Use CREATE_NO_WINDOW to fully detach
    CREATE_NO_WINDOW = 0x08000000

    args = [
        sys.executable,
        'scripts/11_pipeline_orchestrator.py',
        '--full',
    ]

    log = open(PROJECT_ROOT / 'logs' / 'pipeline_detached.log', 'w')

    p = subprocess.Popen(
        args,
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        creationflags=CREATE_NO_WINDOW,
        close_fds=True,
    )

    print(f"Pipeline launched with PID {p.pid}")
    print(f"Log: logs/pipeline_detached.log")
    print(f"Dashboard: http://localhost:{DASHBOARD_PORT}/dashboard/live.html")

    # Don't wait - let it run independently
    # The log file handle stays open because the child process inherited it

if __name__ == '__main__':
    main()
