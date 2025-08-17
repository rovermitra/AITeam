#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RoverMitra data pipeline runner (one command = all datasets)

Order:
  1) users        -> users/data/users_core.json
  2) matchmaker   -> MatchMaker/data/matchmaker_profiles.json
  3) flights      -> Flight/data/travel_groups_integrated_v3.json
  4) hotels       -> Hotels/... (uses users/matchmaker/flights if your generator does)
  5) restaurants  -> Restaurants/data/*.json (reads users + flights)
  6) activities   -> Activities/data/*.json (reads users + flights)
  7) rentals      -> Rentals/data/*.json (reads users + flights)

Notes
- The script auto-locates generators inside ./Scripts.
- Missing optional generators (e.g., hotel) are skipped with a friendly note.
- Uses the same Python interpreter you launch it with.
"""

from __future__ import annotations
import subprocess
import sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "Scripts"

# Keep these names in the exact run order:
GEN_ORDER = [
    "user_data_generator.py",             # users
    "matchmaker_data_generator.py",       # matchmaker
    "Flight_data_generator.py",           # groups + flight/train + base inventories
    "hotel_data_generator.py",            # (optional) hotels (if you have it as a separate file)
    "restaurant_data_generator.py",       # restaurants (integrated, reads users + groups)
    "events_and_activities_data_generator.py",  # activities/events (reads users + groups)
    "rentals_data_generator.py",          # car rentals (reads users + groups)
]

def run_script(path: Path) -> tuple[bool, str]:
    """Run a generator and stream output. Return (ok, log_path_or_err)."""
    if not path.exists():
        return False, f"❗ Not found: {path.name} — skipping (optional or misnamed?)"
    try:
        print(f"\n──▶ Running {path.name}")
        print("   ", "-" * (10 + len(path.name)))
        # run with the same interpreter, inherit stdout/stderr
        result = subprocess.run(
            [sys.executable, str(path)],
            cwd=str(ROOT),
            check=False
        )
        if result.returncode == 0:
            return True, f"✅ {path.name} finished"
        return False, f"🛑 {path.name} exited with code {result.returncode}"
    except Exception as e:
        return False, f"🛑 {path.name} failed: {e}"

def ensure_std_dirs():
    # Create top-level data dirs if a generator expects them
    for d in [
        "users/data",
        "MatchMaker/data",
        "Flight/data",
        "Hotels/data",
        "Restaurants/data",
        "Activities/data",
        "Rentals/data",
    ]:
        (ROOT / d).mkdir(parents=True, exist_ok=True)

def main():
    print(f"RoverMitra Data Pipeline • {datetime.now().isoformat(timespec='seconds')}")
    print(f"Repo root: {ROOT}")
    ensure_std_dirs()

    results = []
    for fname in GEN_ORDER:
        ok, msg = run_script(SCRIPTS / fname)
        print("   ", msg)
        results.append((fname, ok, msg))

    # Pretty summary
    print("\n════════ SUMMARY ════════")
    success = [n for n, ok, _ in results if ok]
    failed  = [n for n, ok, _ in results if not ok]

    if success:
        print("✅ Completed:", ", ".join(success))
    if failed:
        print("🛠  Skipped/Failed:", ", ".join(failed))
        print("\nTips:")
        print(" • Make sure filenames in ./Scripts match the GEN_ORDER list above.")
        print(" • If a generator moved, update GEN_ORDER or the file name.")
        print(" • Run individual scripts to see detailed tracebacks.")

    # Minimal existence checks for the core three
    core_checks = {
        "users/data/users_core.json": "Users data",
        "MatchMaker/data/matchmaker_profiles.json": "Matchmaker data",
        "Flight/data/travel_groups_integrated_v3.json": "Groups & flights",
    }
    print("\nCore outputs:")
    for rel, label in core_checks.items():
        p = ROOT / rel
        print(f" • {label:<24} → {'✅' if p.exists() else '❌'} {rel}")

    print("\nDone.")

if __name__ == "__main__":
    main()
