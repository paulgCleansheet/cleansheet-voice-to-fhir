"""
Combined Server Launcher for Voice-to-FHIR Pipeline

Launches both:
1. MedASR local server (port 3002) - Medical speech recognition
2. Voice-to-FHIR API server (port 8001) - Pipeline orchestration

Usage:
    python launch_servers.py

Prerequisites:
    pip install -r requirements.txt
    pip install flask flask-cors librosa  # For MedASR
    huggingface-cli login  # For MedASR model access
"""

import subprocess
import sys
import os
import time
import signal
import threading
from pathlib import Path

# Configuration
MEDASR_PORT = 3002
PIPELINE_PORT = 8001

# Track subprocesses for cleanup
processes = []


def find_medasr_server():
    """Find the MedASR server script."""
    # Check common locations
    locations = [
        Path(__file__).parent.parent / "CleansheetMedical" / "test" / "medasr-server.py",
        Path(__file__).parent / ".." / "CleansheetMedical" / "test" / "medasr-server.py",
        Path(os.environ.get("CLEANSHEET_MEDICAL", "")) / "test" / "medasr-server.py",
    ]

    for loc in locations:
        if loc.exists():
            return str(loc.resolve())

    # Try relative to current working directory
    cwd_path = Path.cwd().parent / "CleansheetMedical" / "test" / "medasr-server.py"
    if cwd_path.exists():
        return str(cwd_path.resolve())

    return None


def stream_output(process, name, color_code):
    """Stream subprocess output with prefix."""
    for line in iter(process.stdout.readline, ''):
        if line:
            print(f"\033[{color_code}m[{name}]\033[0m {line}", end='')
    for line in iter(process.stderr.readline, ''):
        if line:
            print(f"\033[{color_code}m[{name}]\033[0m \033[91m{line}\033[0m", end='')


def launch_medasr_server():
    """Launch the MedASR local server."""
    medasr_script = find_medasr_server()

    if not medasr_script:
        print("\033[93m[Launcher] MedASR server script not found. Skipping MedASR.\033[0m")
        print("\033[93m[Launcher] Expected at: ../CleansheetMedical/test/medasr-server.py\033[0m")
        print("\033[93m[Launcher] Pipeline will use Whisper fallback for transcription.\033[0m")
        return None

    print(f"\033[94m[Launcher] Starting MedASR server on port {MEDASR_PORT}...\033[0m")
    print(f"\033[94m[Launcher] Script: {medasr_script}\033[0m")

    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'

    process = subprocess.Popen(
        [sys.executable, "-B", medasr_script],  # -B disables bytecode caching
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=env
    )

    # Stream output in background thread
    threading.Thread(target=stream_output, args=(process, "MedASR", "92"), daemon=True).start()

    processes.append(process)
    return process


def launch_pipeline_server():
    """Launch the Voice-to-FHIR pipeline server."""
    server_script = Path(__file__).parent / "server.py"

    if not server_script.exists():
        print("\033[91m[Launcher] server.py not found!\033[0m")
        return None

    print(f"\033[94m[Launcher] Starting Voice-to-FHIR server on port {PIPELINE_PORT}...\033[0m")

    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    # Configure to use local MedASR by default
    env['MEDASR_BACKEND'] = 'local'
    env['MEDASR_LOCAL_URL'] = f'http://localhost:{MEDASR_PORT}'

    process = subprocess.Popen(
        [sys.executable, "-B", str(server_script)],  # -B disables bytecode caching
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=env,
        cwd=str(server_script.parent)
    )

    # Stream output in background thread
    threading.Thread(target=stream_output, args=(process, "Pipeline", "96"), daemon=True).start()

    processes.append(process)
    return process


def cleanup(signum=None, frame=None):
    """Clean up all subprocesses."""
    print("\n\033[94m[Launcher] Shutting down servers...\033[0m")

    for proc in processes:
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

    print("\033[94m[Launcher] All servers stopped.\033[0m")
    sys.exit(0)


def main():
    # Register signal handlers
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    print("""
+------------------------------------------------------------------------------+
|  Voice-to-FHIR Pipeline Launcher                                             |
+------------------------------------------------------------------------------+
|  Starting servers:                                                           |
|    * MedASR (medical ASR)     -> http://localhost:3002                       |
|    * Voice-to-FHIR Pipeline   -> http://localhost:8001                       |
|                                                                              |
|  Press Ctrl+C to stop all servers                                            |
+------------------------------------------------------------------------------+
""")

    # Launch MedASR first (it takes time to load the model)
    medasr_proc = launch_medasr_server()

    if medasr_proc:
        # Wait a bit for MedASR to start loading
        print("\033[94m[Launcher] Waiting for MedASR model to load...\033[0m")
        time.sleep(3)

    # Launch pipeline server
    pipeline_proc = launch_pipeline_server()

    if not pipeline_proc:
        print("\033[91m[Launcher] Failed to start pipeline server!\033[0m")
        cleanup()
        return

    print("""
+------------------------------------------------------------------------------+
|  Servers Starting...                                                         |
+------------------------------------------------------------------------------+
|  MedASR model loading may take 1-2 minutes on first run.                     |
|  Once ready, open the demo:                                                  |
|                                                                              |
|    cleansheet-voice-to-fhir/demo/index.html                                  |
|                                                                              |
|  API Documentation: http://localhost:8001/docs                               |
+------------------------------------------------------------------------------+
""")

    # Keep main thread alive and monitor processes
    try:
        while True:
            # Check if any process died
            for proc in processes:
                if proc and proc.poll() is not None:
                    print(f"\033[91m[Launcher] A server process exited with code {proc.returncode}\033[0m")
            time.sleep(1)
    except KeyboardInterrupt:
        cleanup()


if __name__ == "__main__":
    main()
