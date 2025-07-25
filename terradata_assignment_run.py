import subprocess
import sys
import os
import time
from pathlib import Path

# Ensure script always runs from project root
PROJECT_ROOT = Path(__file__).parent.resolve()
if Path.cwd() != PROJECT_ROOT:
    os.chdir(PROJECT_ROOT)

# Helper to run a command in the background

def run_background(cmd, cwd=None):
    if sys.platform == 'win32':
        # On Windows, use creationflags to hide the new window
        return subprocess.Popen(cmd, cwd=cwd, creationflags=subprocess.CREATE_NEW_CONSOLE)
    else:
        return subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

if __name__ == "__main__":
    print("Launching Terradata Assignment servers...\n")

    # Start FastAPI backend
    print("[1/2] Starting FastAPI backend at http://localhost:8000 ...")
    backend_proc = run_background([
        sys.executable, '-m', 'uvicorn', 'backend.api.main:app', '--reload', '--host', '0.0.0.0', '--port', '8000'
    ], cwd=str(PROJECT_ROOT))

    # Wait a bit to ensure backend starts first
    time.sleep(2)

    # Start Streamlit frontend
    print("[2/2] Starting Streamlit frontend at http://localhost:8501 ...")
    frontend_proc = run_background([
        sys.executable, '-m', 'streamlit', 'run', 'app.py'
    ], cwd=str(PROJECT_ROOT / 'frontend'))

    print("\nBoth servers are running!")
    print("- Backend:   http://localhost:8000")
    print("- Frontend:  http://localhost:8501")
    print("\nPress Ctrl+C in this terminal to stop both servers.")

    try:
        # Wait for both processes
        backend_proc.wait()
        frontend_proc.wait()
    except KeyboardInterrupt:
        print("\nShutting down servers...")
        backend_proc.terminate()
        frontend_proc.terminate()
        print("Done.") 