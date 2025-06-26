import asyncio
import os
import uuid
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Annotated

import h5py
import numpy as np
import redis
import uvicorn
from davidia.main import create_app
from fastapi import (
    Cookie,
    Depends,
    FastAPI,
    HTTPException,
    Response,
    WebSocket,
)
from fastapi.middleware.cors import CORSMiddleware

from blue_histogramming.models import Settings
from blue_histogramming.routers import file_router
from blue_histogramming.session_state_manager import (
    SessionStateManager,
)
from blue_histogramming.utils import (
    calculate_fractions,
    process_image_direct,
)


@lru_cache
def get_settings() -> Settings:
    return Settings()


# Global session manager registry (singleton pattern)
session_managers: dict[str, SessionStateManager] = {}


def get_redis() -> redis.Redis:
    settings = get_settings()
    # Use a single Redis connection for the app, cached with lru_cache
    return _get_redis(settings.redis.host, settings.redis.port)


@lru_cache
def _get_redis(host: str, port: int) -> redis.Redis:
    return redis.Redis(host=host, port=port)


app = FastAPI()

davidia_app = create_app()

# CORS setup for development
davidia_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

state: dict[str, SessionStateManager] = {}


def get_session_manager(
    settings: Annotated[Settings, Depends(get_settings)],
    session_id: str | None = Cookie(default=None),
) -> SessionStateManager:
    if not session_id:
        raise HTTPException(status_code=400, detail="Session ID is required")
    if session_id not in session_managers:
        session_managers[session_id] = SessionStateManager(
            session_id, get_redis(), settings
        )
    return session_managers[session_id]


app.include_router(file_router.router)
# https://fastapi.tiangolo.com/advanced/sub-applications/?h=mount#top-level-application
app.mount("/davidia", davidia_app)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    print(settings)
    yield
    print("finishing the application run ")


@app.post("/login")
async def login(
    response: Response,
    session_manager: Annotated[SessionStateManager, Depends(get_session_manager)],
    settings: Annotated[Settings, Depends(get_settings)],
):
    session_manager.set_session_data({"status": "active"})
    session_manager.redis.expire(
        f"session:{session_manager.session_id}", settings.session_timeout_seconds
    )
    response.set_cookie(
        key="session_id", value=session_manager.session_id, httponly=True
    )
    return {"message": "Session created", "session_id": session_manager.session_id}


@app.get("/files")
async def get_files(
    session_manager: Annotated[SessionStateManager, Depends(get_session_manager)],
):
    """
    List all the files in the session's allowed directory.
    """
    session_data = session_manager.get_session_data()
    session_path = session_data.get("filepath")
    print(f"Session path: {session_path}")
    if not session_path or not os.path.isdir(session_path):
        return {"files": []}
    try:
        files = [f for f in os.listdir(session_path) if f.endswith(".hdf")]
    except Exception:
        files = []
    return {"files": files}


@app.websocket("/ws/{client_id}/demo/{filename}")
async def stream_existing_dataset(
    session_manager: Annotated[SessionStateManager, Depends(get_session_manager)],
    client_id: str,
    websocket: WebSocket,
    filename: str,
):
    """
    endpoint with hardcoded read
    """
    folder = os.path.dirname(os.path.abspath(filename))

    # here the websocket is persisted
    session_manager.start_observer_for_session(
        folder=folder,
        websocket=websocket,
    )
    dataset_name: str = "entry/instrument/detector/data"
    filepath = "/workspaces/blue-histogramming/data"
    f: h5py.File = h5py.File(os.path.join(filepath, filename), "r")
    dset = file_router.guard_dataset_and_group(
        filename, "entry/instrument/detector", dataset_name, f
    )
    try:
        while True:
            if dset is None or len(dset) == 0:
                print("❌ No data available, sending reset message...")
                await websocket.send_json([])
                await asyncio.sleep(1)
                continue  # Restart loop to wait for new data

            # Loop through each batch of data points
            for i in range(1, len(dset) + 1):
                raw_data: np.ndarray = dset[-i:]
                stats_list = [(process_image_direct(img)) for img in raw_data]
                fractions = calculate_fractions(np.array(stats_list))
                try:
                    # Send the fractions to the frontend
                    await websocket.send_json(fractions)
                except Exception as e:
                    print(f"sending json error: {e}")

                # Wait for a small interval before sending the next batch
                await asyncio.sleep(1)  # You can adjust the sleep time as needed
            # Send an empty array to indicate end of batch
            print("✅ Sending reset message...")
            await websocket.send_json([])
            await asyncio.sleep(1)  # Wait before starting a new round of data
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        print("Closing WebSocket connection")
        await websocket.close()


@app.post("/set_dataset/")
def set_dataset(
    session_manager: Annotated[SessionStateManager, Depends(get_session_manager)],
    settings: Annotated[Settings, Depends(get_settings)],
    response: Response,
    filepath: str,
    filename: str,
    dataset_name: str,
):
    # check if path in allowed_hdf_path
    if not filepath.startswith(settings.allowed_hdf_path.as_posix()):
        raise HTTPException(
            status_code=403,
            detail=f"Path {filepath} is not allowed. Must start with {settings.allowed_hdf_path}",
        )
    # set the filepath, filename, and dataset_name for this session
    session_manager.set_session_data(
        {
            "filepath": filepath,
            "filename": filename,
            "dataset_name": dataset_name,
        }
    )

    try:
        full_path = f"{filepath}/{filename}"
        f = h5py.File(full_path, "r", libver="latest", swmr=True)
        if dataset_name in f:
            dset = file_router.guard_dataset_and_group(
                filename, "entry/instrument/detector", dataset_name, f
            )
        else:
            raise HTTPException(status_code=404, detail="Dataset not found")
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to set dataset: {str(e)}"
        ) from e

    return {
        "message": "Dataset set successfully",
        "shape": dset.shape if "dset" in locals() else None,
    }


@app.websocket("/ws/{client_id}/plan/demo")
async def demo_endpoint(
    session_manager: Annotated[SessionStateManager, Depends(get_session_manager)],
    client_id: str,
    websocket: WebSocket,
    plan_name: str,
    session_id: str | None = Cookie(default=None),
):
    """
    Orchestrates a demo process:
    1. Accepts the websocket connection.
    2. Starts listening to STOMP for events.
    3. Sends a POST request to BlueAPI to start a plan.
    4. Handles events, reads filesystem, processes data.
    5. Sends progress and final plot URL to the client.
    """
    await websocket.accept()
    plan_name = "demo_plan"  # Hardcoded for demo, replace with actual logic
    try:
        await websocket.send_json(
            {"status": "starting", "msg": f"Starting plan {plan_name}"}
        )

        # 1. Start STOMP listener (simulate or call your session_manager logic)
        await websocket.send_json(
            {"status": "info", "msg": "Listening for STOMP events..."}
        )

        # 2. Start the plan via BlueAPI (using your blueapi client)
        from blue_histogramming.proxy import get_blueapi_client

        blueapi_client = get_blueapi_client()
        # Example payload, adjust as needed
        plan_payload = {"name": plan_name, "params": {"detectors": ["manta"]}}
        try:
            response = await blueapi_client.post("/task", json=plan_payload)
            response.raise_for_status()
            plan_response = response.json()
            await websocket.send_json({"status": "plan_started", "msg": plan_response})
        except Exception as e:
            await websocket.send_json(
                {"status": "error", "msg": f"Failed to start plan: {e}"}
            )
            return

        # 3. Simulate or handle events, read/process data (replace with real logic)
        await websocket.send_json({"status": "processing", "msg": "Processing data..."})
        await asyncio.sleep(2)  # Simulate work

        # 4. When ready, send the plot URL (or plot_id) for the client to connect to
        plot_id = plan_response.get("plot_id", f"demo-{uuid.uuid4()}")
        plot_url = f"/davidia/plot/{{uuid}}/{plot_id}"
        await websocket.send_json(
            {
                "status": "ready",
                "plot_id": plot_id,
                "plot_url": plot_url,
                "msg": "Connect to this WebSocket for data streaming.",
            }
        )
    except Exception as e:
        await websocket.send_json({"status": "error", "msg": str(e)})
    finally:
        await websocket.close()


def run_server():
    """Start the FastAPI app and STOMP listener."""
    uvicorn.run(app, host="0.0.0.0", port=8002)
