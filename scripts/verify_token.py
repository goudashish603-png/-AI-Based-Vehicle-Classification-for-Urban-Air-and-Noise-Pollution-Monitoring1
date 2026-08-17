import os

TOKEN = os.environ.get("GITHUB_TOKEN", "")
HEADERS = {"Authorization": f"token {TOKEN}", "Accept": "application/vnd.github.v3+json"}

# 1. Check User
r_user = requests.get("https://api.github.com/user", headers=HEADERS)
print("USER INFO:", r_user.status_code, r_user.json().get("login"))

username = r_user.json().get("login", "narsagoudgantala741-ship-it")

# 2. Try creating repository if missing
repo_name = "AI-Based-Vehicle-Classification-for-Urban-Air-and-Noise-Pollution-Monitoring"
r_create = requests.post("https://api.github.com/user/repos", headers=HEADERS, json={
    "name": repo_name,
    "description": "AI-Based Vehicle Classification for Urban Air and Noise Pollution Monitoring System",
    "private": False,
    "auto_init": False
})
print("CREATE REPO:", r_create.status_code, r_create.json().get("message", "Created"))

