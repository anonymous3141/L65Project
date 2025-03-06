import subprocess
import os
import signal
import sys
"""
usage: 
0) type in all the launch commands you want to do in run_commands.txt 
    (as you'd usually run them in the terminal)
   feel free to add comments with # at the beginning of the line
   end the list with a line that says "END"
1) new terminal, run source clrs_env/bin/activate
2) run python run_command_list.py
"""

# List of commands to run
L = open("run_commands.txt","r").readlines()
L = ["export XLA_PYTHON_CLIENT_PREALLOCATE=false; bash -c 'source clrs_env/bin/activate;"+c.replace("\n","") + "' " for c in L]

# Store running processes
processes = []

def terminate_processes():
    """ Send SIGTERM to all running subprocesses """
    print("\nTerminating all subprocesses...")
    for process in processes:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)  # Graceful termination
        except ProcessLookupError:
            pass  # Process already terminated

# Handle Ctrl+C
def signal_handler(sig, frame):
    print("\nCtrl+C detected! Stopping all subprocesses...")
    terminate_processes()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

try:
    for cmd in L:
        if "#" in cmd:
            continue 
        if "END" in cmd:
            break

        # Launch each command in its own process group
        process = subprocess.Popen(cmd, shell=True)        
        processes.append(process)

    # Wait for all subprocesses
    for process in processes:
        process.communicate()

except KeyboardInterrupt:
    signal_handler(None, None)  # Handle Ctrl+C properly

finally:
    terminate_processes()  # Ensure cleanup