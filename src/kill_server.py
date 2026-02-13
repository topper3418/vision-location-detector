#!/usr/bin/env python3
"""Utility script to kill the vision location detector server.

This script can be run directly or imported as a module.
"""

import sys
import signal
import psutil
import time


def find_server_processes():
    """Find all running server processes.
    
    Returns:
        List of psutil.Process objects for server processes
    """
    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info['cmdline']
            if cmdline and 'python' in cmdline[0].lower():
                # Check if this is our server process
                if any('src.main' in arg or 'src/main.py' in arg for arg in cmdline):
                    processes.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    
    return processes


def kill_server_processes(force: bool = False):
    """Kill all server processes.
    
    Args:
        force: If True, use SIGKILL instead of SIGTERM
    
    Returns:
        Number of processes killed
    """
    processes = find_server_processes()
    
    if not processes:
        print("No server processes found running.")
        return 0
    
    print(f"Found {len(processes)} server process(es)")
    
    killed = 0
    for proc in processes:
        try:
            print(f"Killing process {proc.pid} ({' '.join(proc.cmdline())})...")
            if force:
                proc.kill()  # SIGKILL
            else:
                proc.terminate()  # SIGTERM
            killed += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            print(f"Failed to kill process {proc.pid}: {e}")
    
    if not force:
        # Wait for processes to terminate gracefully
        print("Waiting for processes to terminate...")
        time.sleep(2)
        
        # Check if any are still running
        remaining = find_server_processes()
        if remaining:
            print(f"{len(remaining)} process(es) still running, force killing...")
            kill_server_processes(force=True)
    
    return killed


def main():
    """Main entry point."""
    force = '--force' in sys.argv or '-f' in sys.argv
    
    print("Stopping vision location detector server...")
    try:
        count = kill_server_processes(force=force)
        if count > 0:
            print(f"Successfully stopped {count} process(es).")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
