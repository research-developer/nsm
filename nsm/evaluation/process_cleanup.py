"""
Process cleanup utility for NSM training runs.

Helps identify and clean up orphaned training processes to prevent
resource conflicts and confusion about which runs are active.
"""

import subprocess
import sys
from typing import List, Dict, Optional


def find_training_processes() -> List[Dict[str, str]]:
    """
    Find all running NSM training processes.

    Returns:
        List of dicts with process info (pid, cpu, mem, time, cmd)
    """
    try:
        # Find all Python training processes
        result = subprocess.run(
            ["ps", "aux"],
            capture_output=True,
            text=True,
            check=True
        )

        processes = []
        for line in result.stdout.split('\n'):
            # Look for train_*.py processes
            if 'python' in line and any(x in line for x in ['train_planning.py', 'train_causal.py', 'train_kg.py']):
                parts = line.split()
                if len(parts) >= 11:
                    # Determine domain from command
                    domain = 'unknown'
                    if 'train_planning.py' in line:
                        domain = 'planning'
                    elif 'train_causal.py' in line:
                        domain = 'causal'
                    elif 'train_kg.py' in line:
                        domain = 'kg'

                    processes.append({
                        'pid': parts[1],
                        'cpu': parts[2],
                        'mem': parts[3],
                        'time': parts[9],
                        'cmd': ' '.join(parts[10:13]),
                        'domain': domain,
                        'full_cmd': ' '.join(parts[10:])
                    })

        return processes

    except subprocess.CalledProcessError as e:
        print(f"Error finding processes: {e}")
        return []


def kill_process(pid: str, force: bool = False) -> bool:
    """
    Kill a process by PID.

    Args:
        pid: Process ID to kill
        force: If True, use SIGKILL (-9) instead of SIGTERM (-15)

    Returns:
        True if successful, False otherwise
    """
    try:
        signal = '-9' if force else '-15'
        subprocess.run(['kill', signal, pid], check=True)
        return True
    except subprocess.CalledProcessError:
        return False


def check_and_cleanup(interactive: bool = True, auto_kill: bool = False) -> None:
    """
    Check for orphaned training processes and optionally clean them up.

    Args:
        interactive: If True, prompt user before killing processes
        auto_kill: If True, automatically kill all found processes (requires interactive=False)
    """
    print("\n" + "="*80)
    print("🔍 Checking for orphaned NSM training processes...")
    print("="*80 + "\n")

    processes = find_training_processes()

    if not processes:
        print("✅ No orphaned training processes found.\n")
        return

    print(f"⚠️  Found {len(processes)} training process(es):\n")

    for i, proc in enumerate(processes, 1):
        print(f"{i}. [{proc['domain'].upper()}] PID: {proc['pid']}")
        print(f"   CPU: {proc['cpu']}%, MEM: {proc['mem']}%, TIME: {proc['time']}")
        print(f"   CMD: {proc['cmd']}")
        print()

    if not interactive and not auto_kill:
        print("ℹ️  Run with interactive=True to clean up processes.\n")
        return

    if auto_kill and not interactive:
        print("🗑️  Auto-killing all processes...")
        for proc in processes:
            if kill_process(proc['pid']):
                print(f"   ✅ Killed PID {proc['pid']} ({proc['domain']})")
            else:
                print(f"   ❌ Failed to kill PID {proc['pid']} ({proc['domain']})")
        print()
        return

    # Interactive cleanup
    while True:
        response = input("\nKill these processes? [y/n/select] (y=all, n=none, select=choose): ").lower().strip()

        if response == 'n':
            print("Skipping cleanup.\n")
            break

        elif response == 'y':
            print("\n🗑️  Killing all processes...")
            for proc in processes:
                if kill_process(proc['pid']):
                    print(f"   ✅ Killed PID {proc['pid']} ({proc['domain']})")
                else:
                    print(f"   ❌ Failed to kill PID {proc['pid']} ({proc['domain']})")
            print()
            break

        elif response == 'select':
            pids_to_kill = input("\nEnter PIDs to kill (space-separated): ").strip().split()
            print()
            for pid in pids_to_kill:
                # Find the process
                proc = next((p for p in processes if p['pid'] == pid), None)
                if proc:
                    if kill_process(pid):
                        print(f"✅ Killed PID {pid} ({proc['domain']})")
                    else:
                        print(f"❌ Failed to kill PID {pid} ({proc['domain']})")
                else:
                    print(f"⚠️  PID {pid} not found in process list")
            print()
            break

        else:
            print("Invalid response. Please enter 'y', 'n', or 'select'.")


if __name__ == '__main__':
    # CLI mode
    import argparse

    parser = argparse.ArgumentParser(
        description="Find and clean up orphaned NSM training processes"
    )
    parser.add_argument(
        '--auto-kill',
        action='store_true',
        help="Automatically kill all found processes without prompting"
    )
    parser.add_argument(
        '--list-only',
        action='store_true',
        help="Only list processes, don't prompt for cleanup"
    )

    args = parser.parse_args()

    if args.auto_kill and args.list_only:
        print("Error: --auto-kill and --list-only are mutually exclusive")
        sys.exit(1)

    check_and_cleanup(
        interactive=not args.list_only and not args.auto_kill,
        auto_kill=args.auto_kill
    )
