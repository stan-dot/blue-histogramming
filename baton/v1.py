import re
import signal
import subprocess
from pathlib import Path

import requests
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import BaseModel, HttpUrl


class GitRepoModel(BaseModel):
    package_name: str
    url: HttpUrl
    branch: str | None = "main"


my_repos = [
    GitRepoModel(
        url=HttpUrl("https://github.com/DiamondLightSource/dodal.git"),
        branch="main",
        package_name="dodal",
    ),
    GitRepoModel(
        url=HttpUrl(
            "https://github.com/DiamondLightSource/spectroscopy-bluesky.git",
        ),
        branch="i20_add_panda_and_motors",
        package_name="spectroscopy-bluesky",
    ),
]


def clone_or_pull_repo(base_path: Path, repo: GitRepoModel) -> None:
    try:
        path = base_path / repo.package_name
        # Initialize the repo if it doesn't exist
        if not (path / ".git").exists():
            print(f"Initializing new Git repo at {path}")
            # first make the dfir
            path.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "init"], cwd=path, check=True)
            subprocess.run(
                ["git", "remote", "add", "origin", str(repo.url)], cwd=path, check=True
            )

        # Pull the latest changes
        print(f"Pulling latest changes from {repo.url} ({repo.branch})")
        subprocess.run(
            ["git", "pull", "origin", repo.branch or "main"], cwd=path, check=True
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Failed to clone or pull repo: {repo.url} ({repo.branch})"
        ) from e


app = FastAPI()

SESSIONS: dict[str, subprocess.Popen] = {}
BATON_HOLDER = None
BASE_PATH = "/envs/instruments"
PORT_REGISTRY = set()
S2_URL = "http://example.com/api"


class NewSessionRequest(BaseModel):
    instrument: str
    userid: str


class RunRequest(BaseModel):
    instrument: str
    userid: str
    description: str


class ChangeBatonRequest(BaseModel):
    new_userid: str
    instrument: str


@app.get("/", response_class=HTMLResponse)
def root():
    session_list = (
        "<ul>" + "".join(f"<li>{user}</li>" for user in SESSIONS.keys()) + "</ul>"
    )
    baton_holder = (
        f"<p><strong>Current Baton Holder:</strong> {BATON_HOLDER or 'None'}</p>"
    )
    data = f"<html><body>{baton_holder}<h2>Active Sessions:</h2>{session_list}</body></html>"
    return PlainTextResponse(data)


@app.get("/show_current_user")
def show_current_user():
    if BATON_HOLDER is None:
        return {"message": "No user currently holds the baton."}
    return {"baton_holder": BATON_HOLDER}


@app.get("/show_users_queue")
def show_users_queue():
    return {"active_users": list(SESSIONS.keys())}


@app.post("/new_session")
def new_session(req: NewSessionRequest, background_tasks: BackgroundTasks):
    # check if marimo is installed
    if not subprocess.run(["which", "marimo"], stdout=subprocess.PIPE).stdout:
        raise HTTPException(status_code=500, detail="Marimo is not installed.")
    global BATON_HOLDER
    instrument_path = Path(f"{BASE_PATH}/{req.instrument}/users/{req.userid}/workspace")
    instrument_path.mkdir(parents=True, exist_ok=True)

    # Create venv if it doesn't exist
    venv_path = instrument_path / "venv"
    if not venv_path.exists():
        subprocess.run(["python3", "-m", "venv", str(venv_path)])

    # path with user
    user_path = instrument_path / "users" / req.userid

    for repo in my_repos:
        # Clone or pull the git repo
        clone_or_pull_repo(user_path, repo)
    # # Clone the git repo
    # subprocess.run(["git", "init"], cwd=instrument_path)
    # subprocess.run(
    #     ["git", "remote", "add", "origin", "https://example.com/repo.git"],
    #     cwd=instrument_path,
    # )
    # subprocess.run(["git", "pull", "origin", "main"], cwd=instrument_path)

    # Start the Marimo server in headless mode on a free port
    port = next(p for p in range(8000, 9000) if p not in PORT_REGISTRY)
    PORT_REGISTRY.add(port)
    proc = subprocess.Popen(
        ["marimo", "edit", "--headless", "--host", "0.0.0.0", "--port", str(port)],
        cwd=user_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    output = proc.stdout.readline()
    token_match = re.search(r"\?access_token=([a-zA-Z0-9-_]+)", output)
    if not token_match:
        proc.kill()
        PORT_REGISTRY.remove(port)
        raise HTTPException(status_code=500, detail="Failed to retrieve access token.")

    access_token = token_match.group(1)
    SESSIONS[req.userid] = proc
    if BATON_HOLDER is None:
        BATON_HOLDER = req.userid

    background_tasks.add_task(proc.wait)
    return {"message": "Session created", "access_token": access_token, "port": port}


@app.post("/terminate_session")
def terminate_session(req: NewSessionRequest):
    global BATON_HOLDER
    if req.userid not in SESSIONS:
        raise HTTPException(status_code=404, detail="User session not found.")
    proc = SESSIONS.pop(req.userid)
    proc.send_signal(signal.SIGTERM)
    PORT_REGISTRY.discard(proc.pid)
    if req.userid == BATON_HOLDER:
        BATON_HOLDER = None
    return {"message": "Session terminated"}


@app.post("/run")
def run(req: RunRequest):
    workspace_path = Path(f"{BASE_PATH}/{req.instrument}/users/{req.userid}/workspace")
    if not workspace_path.exists():
        raise HTTPException(status_code=404, detail="Workspace not found.")

    # Commit changes
    subprocess.run(["git", "add", "*"], cwd=workspace_path)
    subprocess.run(["git", "commit", "-m", req.description], cwd=workspace_path)
    return {"message": "Changes committed"}


@app.post("/change_baton")
def change_baton(req: ChangeBatonRequest):
    global BATON_HOLDER
    if req.new_userid not in SESSIONS:
        raise HTTPException(
            status_code=404, detail="User not found in active sessions."
        )
    BATON_HOLDER = req.new_userid
    user_path = f"{BASE_PATH}/{req.instrument}/users/{req.new_userid}/workspace"
    response = requests.post(
        f"{S2_URL}/environment/{req.new_userid}", json={"path": user_path}
    )
    response.raise_for_status()
    return {"message": "Baton changed", "new_baton_holder": BATON_HOLDER}


# here just run the server
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
