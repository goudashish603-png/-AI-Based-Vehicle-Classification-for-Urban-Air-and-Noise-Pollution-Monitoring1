import os
import sys
import base64
import json
import requests
from pathlib import Path

# GitHub Credentials & Repository Target
TOKEN = os.environ.get("GITHUB_TOKEN", "").strip()
REPO = "AI-Based-Vehicle-Classification-for-Urban-Air-and-Noise-Pollution-Monitoring"
BRANCH = "main"

if not TOKEN:
    print("[ERROR] GITHUB_TOKEN environment variable is missing.")
    print(" -> Generate a token here: https://github.com/settings/tokens/new")
    print(" -> Then set it in PowerShell: $env:GITHUB_TOKEN='your_token_here'")
    sys.exit(1)

HEADERS = {
    "Authorization": f"token {TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

# Auto-detect account username from token
user_res = requests.get("https://api.github.com/user", headers=HEADERS)
if user_res.status_code == 200:
    OWNER = user_res.json()["login"]
    print(f"[OK] Authenticated as GitHub user: {OWNER}")
else:
    print(f"[ERROR] Token authentication failed (HTTP {user_res.status_code}): {user_res.text}")
    sys.exit(1)

BASE_URL = f"https://api.github.com/repos/{OWNER}/{REPO}"

# Ensure repository exists on GitHub (create if missing)
repo_check = requests.get(BASE_URL, headers=HEADERS)
if repo_check.status_code == 404:
    print(f"Repo '{REPO}' does not exist on GitHub under '{OWNER}'. Creating it now...")
    create_res = requests.post(
        "https://api.github.com/user/repos",
        headers=HEADERS,
        json={"name": REPO, "private": False, "auto_init": False}
    )
    if create_res.status_code in [200, 201]:
        print(f"[OK] Successfully created repository: https://github.com/{OWNER}/{REPO}")
    else:
        print(f"[ERROR] Failed to create repo: {create_res.status_code} {create_res.text}")

# Files/Folders to ignore during upload
IGNORE_PATTERNS = [
    ".git", ".venv", "__pycache__", ".pytest_cache", ".DS_Store", "Thumbs.db",
    "data/raw", "data/interim", "data/processed", "outputs/predictions", "yolov8n.pt"
]

def should_ignore(rel_path: str) -> bool:
    for pattern in IGNORE_PATTERNS:
        if rel_path.startswith(pattern) or f"/{pattern}/" in f"/{rel_path}/" or rel_path.endswith(pattern):
            return True
    return False

def ensure_repo_initialized():
    """Initializes empty GitHub repo by creating README.md via contents API if empty."""
    url = f"{BASE_URL}/contents/README.md"
    r = requests.get(url, headers=HEADERS)
    if r.status_code == 404:
        print("Initializing empty repository with README.md...")
        readme_path = Path("README.md")
        content_b64 = base64.b64encode(readme_path.read_bytes()).decode("utf-8")
        payload = {
            "message": "Initial repository setup",
            "content": content_b64,
            "branch": BRANCH
        }
        r_put = requests.put(url, headers=HEADERS, json=payload)
        print("Repo Init Result:", r_put.status_code)

def get_ref():
    url = f"{BASE_URL}/git/ref/heads/{BRANCH}"
    r = requests.get(url, headers=HEADERS)
    if r.status_code == 200:
        return r.json()["object"]["sha"]
    return None

def create_blob(file_path: Path):
    with open(file_path, "rb") as f:
        content = f.read()
    b64_content = base64.b64encode(content).decode("utf-8")
    
    url = f"{BASE_URL}/git/blobs"
    payload = {
        "content": b64_content,
        "encoding": "base64"
    }
    
    for attempt in range(4):
        try:
            r = requests.post(url, headers=HEADERS, json=payload, timeout=30)
            if r.status_code in [200, 201]:
                return r.json()["sha"]
            else:
                print(f"Failed to create blob for {file_path}: {r.status_code} {r.text}")
                return None
        except Exception as e:
            if attempt == 3:
                print(f"Network error creating blob for {file_path}: {e}")
                return None
            import time
            time.sleep(1)

def upload_repository():
    ensure_repo_initialized()
    
    root = Path(".").resolve()
    print(f"Scanning workspace files at: {root}")
    
    tree_items = []
    files_to_upload = []

    for path in root.rglob("*"):
        if path.is_file():
            rel_path = path.relative_to(root).as_posix()
            if not should_ignore(rel_path):
                files_to_upload.append((path, rel_path))

    print(f"Found {len(files_to_upload)} files to push to GitHub...")

    for path, rel_path in files_to_upload:
        print(f"Creating blob for: {rel_path}")
        blob_sha = create_blob(path)
        if blob_sha:
            tree_items.append({
                "path": rel_path,
                "mode": "100644",
                "type": "blob",
                "sha": blob_sha
            })

    if not tree_items:
        print("No files to commit.")
        return

    # Create Tree (full workspace snapshot without legacy base_tree)
    print("Creating Git Tree...")
    tree_url = f"{BASE_URL}/git/trees"
    tree_payload = {"tree": tree_items}
    
    parent_sha = get_ref()

    r_tree = requests.post(tree_url, headers=HEADERS, json=tree_payload)
    if r_tree.status_code not in [200, 201]:
        print(f"Failed to create tree: {r_tree.status_code} {r_tree.text}")
        return
    
    new_tree_sha = r_tree.json()["sha"]
    print(f"Created Git Tree: {new_tree_sha}")

    # Create Commit
    print("Creating Git Commit...")
    commit_url = f"{BASE_URL}/git/commits"
    commit_payload = {
        "message": "Initial Commit: AI-Based Vehicle Classification for Urban Air and Noise Pollution Monitoring System",
        "tree": new_tree_sha
    }
    if parent_sha:
        commit_payload["parents"] = [parent_sha]

    r_commit = requests.post(commit_url, headers=HEADERS, json=commit_payload)
    if r_commit.status_code not in [200, 201]:
        print(f"Failed to create commit: {r_commit.status_code} {r_commit.text}")
        return

    new_commit_sha = r_commit.json()["sha"]
    print(f"Created Commit SHA: {new_commit_sha}")

    # Update Ref
    ref_url = f"{BASE_URL}/git/refs/heads/{BRANCH}"
    ref_payload = {"sha": new_commit_sha, "force": True}
    r_ref = requests.patch(ref_url, headers=HEADERS, json=ref_payload)

    if r_ref.status_code in [200, 201]:
        print("Successfully updated branch 'main'!")
        print(f"[OK] Repository live at: https://github.com/{OWNER}/{REPO}")
    else:
        print(f"Failed to update ref: {r_ref.status_code} {r_ref.text}")

if __name__ == "__main__":
    upload_repository()
