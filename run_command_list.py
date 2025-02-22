import subprocess
"""
usage: 
0) type in all the launch commands you want to do in run_commands.txt
   feel free to add comments with # at the beginning of the line
   end the list with a line that says "END"
1) new terminal, run source clrs_env/bin/activate
2) run python run_command_list.py
"""

# List of commands to run
L = open("run_commands.txt","r").readlines()
L = ["source clrs_env/bin/activate;"+c.replace("\n","") for c in L]

# Run each command in its own subprocess
processes = []
for cmd in L:
    if "#" in cmd:
        continue 
    if "END" in cmd:
        break
    process = subprocess.Popen(cmd, shell=True)
    processes.append(process)

# Wait for all subprocesses to complete
for process in processes:
    process.communicate()