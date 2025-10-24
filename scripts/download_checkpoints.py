#!/usr/bin/env python3
"""
Download checkpoints from Modal volume to local repo.

Usage:
    python scripts/download_checkpoints.py
    python scripts/download_checkpoints.py --pattern "*best*"
"""

import subprocess
import argparse
from pathlib import Path


def download_checkpoints(pattern: str = "*.pt", destination: str = "checkpoints"):
    """Download checkpoints from Modal volume."""
    dest_path = Path(destination)
    dest_path.mkdir(parents=True, exist_ok=True)

    print(f"📥 Downloading checkpoints matching '{pattern}' to {dest_path}/")

    # List available checkpoints
    print("\n🔍 Available checkpoints in Modal volume:")
    result = subprocess.run(
        ["modal", "volume", "ls", "nsm-checkpoints"],
        capture_output=True,
        text=True
    )
    print(result.stdout)

    # Download checkpoints
    cmd = [
        "modal", "volume", "get",
        "nsm-checkpoints",
        str(dest_path)
    ]

    print(f"\n⬇️  Downloading...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print("✅ Download complete!")

        # List what we downloaded
        checkpoints = list(dest_path.glob("*.pt"))
        if checkpoints:
            print(f"\n📦 Downloaded {len(checkpoints)} checkpoints:")
            for cp in sorted(checkpoints):
                size = cp.stat().st_size / (1024 * 1024)  # MB
                print(f"   {cp.name} ({size:.1f} MB)")
        else:
            print("⚠️  No .pt files found in volume")

        # Also check for JSON results
        json_files = list(dest_path.glob("*.json"))
        if json_files:
            print(f"\n📄 Also found {len(json_files)} result files:")
            for jf in sorted(json_files):
                print(f"   {jf.name}")

    else:
        print(f"❌ Error: {result.stderr}")

    return result.returncode == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download checkpoints from Modal")
    parser.add_argument(
        "--pattern",
        default="*.pt",
        help="Pattern to match checkpoint files"
    )
    parser.add_argument(
        "--destination",
        default="checkpoints",
        help="Local destination directory"
    )

    args = parser.parse_args()

    success = download_checkpoints(args.pattern, args.destination)
    exit(0 if success else 1)
