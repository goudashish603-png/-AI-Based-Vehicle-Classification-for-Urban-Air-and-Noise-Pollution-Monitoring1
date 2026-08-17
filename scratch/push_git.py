import os
import subprocess
import sys

def find_git():
    possible_paths = [
        r"C:\Program Files\Git\cmd\git.exe",
        r"C:\Program Files\Git\bin\git.exe",
        os.path.expanduser(r"~\AppData\Local\Programs\Git\cmd\git.exe"),
        os.path.expanduser(r"~\AppData\Local\Programs\Git\bin\git.exe"),
        os.path.expanduser(r"~\AppData\Local\GitHubDesktop\app-*\resources\app\git\cmd\git.exe"),
    ]
    for p in possible_paths:
        if os.path.exists(p):
            return p
    return "git"

git_bin = find_git()
print(f"Using git executable: {git_bin}")

def run_git(args):
    cmd = [git_bin] + args
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(f"CMD: {' '.join(cmd)}")
    print(f"STDOUT: {res.stdout}")
    print(f"STDERR: {res.stderr}")
    return res.returncode

run_git(["status"])
run_git(["add", "."])
run_git(["commit", "-m", "Fix Render deployment config, opencv-python-headless, and FastAPI REST API"])
run_git(["push", "origin", "main"])
