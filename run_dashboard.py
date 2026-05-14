"""
dashboard/run_dashboard.py — One-Click SOC Dashboard Launcher
=============================================================

Run from the LM_System project root:
    python dashboard/run_dashboard.py

This script:
  1. Checks if Mission4 has been run (outputs/mission4_hybrid_summary.json).
  2. If not, reminds you to run Mission4_run.py first.
  3. Starts the Flask API server and opens your browser automatically.
"""

import os
import sys
import subprocess
import webbrowser
import time

# Resolve paths relative to LM_System root (one level up from dashboard/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMMARY_PATH = os.path.join(PROJECT_ROOT, "outputs", "mission4_hybrid_summary.json")
APP_PATH     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")
DASHBOARD_URL = "http://localhost:5000"


def main():
    print("\n" + "=" * 60)
    print("  LMD-SOC SENTINEL  —  Dashboard Launcher")
    print("=" * 60)

    # Check for Mission 4 data
    if os.path.exists(SUMMARY_PATH):
        print("  [OK]  Mission 4 data found — displaying live results.")
    else:
        print("  [!!]  Mission 4 data NOT found.")
        print("        The dashboard will display pre-computed fallback results.")
        print()
        print("  To generate live results, run first:")
        print("    python Mission4_run.py")
        print()

    print(f"  Starting Flask server at {DASHBOARD_URL}")
    print("  Press Ctrl+C to stop.\n")
    print("=" * 60 + "\n")

    # Open browser after a short delay so Flask has time to start
    def open_browser():
        time.sleep(2)
        webbrowser.open(DASHBOARD_URL)

    import threading
    threading.Thread(target=open_browser, daemon=True).start()

    # Start Flask (replace current process so Ctrl+C works cleanly)
    os.execv(sys.executable, [sys.executable, APP_PATH])


if __name__ == "__main__":
    main()
