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
from davidia.models.messages import ImageData, ImageDataMessage, MsgType, PlotMessage
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
from blue_histogramming.session_state_manager import (
    SessionStateManager,
)
from blue_histogramming.utils import (
    calculate_fractions,
    list_hdf5_tree_of_file,
    process_image_direct,
)


@lru_cache
def get_settings() -> Settings:
    return Settings()


def get_redis() -> redis.Redis:
    settings = get_settings()
    return redis.Redis(host=settings.redis_host, port=settings.redis_port)


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
    redis: Annotated[redis.Redis, Depends(get_redis)],
    settings: Annotated[Settings, Depends(get_settings)],
    session_id: str | None = Cookie(default=None),
) -> SessionStateManager:
    if not session_id:
        raise HTTPException(status_code=400, detail="Session ID is required")
    # Use global state dict to persist managers per session
    if session_id not in state:
        state[session_id] = SessionStateManager(session_id, redis, settings)
    return state[session_id]


# https://fastapi.tiangolo.com/advanced/sub-applications/?h=mount#top-level-application
app.mount("/davidia", davidia_app)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    # todo that is old from starlette, need to make into dependency
    app.state.redis = get_redis()

    print(settings)
    yield
    print("finishing the application run ")


@app.post("/login")
async def login(
    response: Response,
    redis: Annotated[redis.Redis, Depends(get_redis)],
    settings: Annotated[Settings, Depends(get_settings)],
):
    session_id = str(uuid.uuid4())

    redis.hset(f"session:{session_id}", mapping={"status": "active"})
    redis.expire(f"session:{session_id}", settings.session_timeout_seconds)
    response.set_cookie(key="session_id", value=session_id, httponly=True)
    return {"message": "Session created", "session_id": session_id}


@app.get("/files")
async def get_files(
    redis: Annotated[redis.Redis, Depends(get_redis)],
    session_id: str | None = Cookie(default=None),
):
    """
    List all the files in the session's allowed directory.
    """
    print(f"Session ID: {session_id}")
    if not session_id:
        return {"files": []}
    session_key = f"session:{session_id}"
    session_data = redis.hgetall(session_key)
    # Redis returns bytes, decode to str
    session_data = {k.decode(): v.decode() for k, v in session_data.items()}
    session_path = session_data.get("filepath")
    print(f"Session path: {session_path}")
    if not session_path or not os.path.isdir(session_path):
        return {"files": []}
    try:
        files = [f for f in os.listdir(session_path) if f.endswith(".hdf")]
    except Exception:
        files = []
    return {"files": files}


@app.get("/file/{id}/detail")
async def get_groups_in_file(
    redis: Annotated[redis.Redis, Depends(get_redis)],
    settings: Annotated[Settings, Depends(get_settings)],
    id: str,
):
    """
    show the groups in the file

    Parameters
    ----------
    redis : Annotated[redis.Redis, Depends
        _description_
    settings : Annotated[Settings, Depends
        _description_
    id : str
        _description_

    Returns
    -------
    _type_
        _description_
    """

    file_path = os.path.join(settings.allowed_hdf_path, id)
    if not os.path.exists(file_path):
        return {"groups": []}
    try:
        file: h5py.File = h5py.File(file_path, "r")
        groups = list_hdf5_tree_of_file(file)
        file.close()
    except Exception:
        groups = []
    print(groups)
    return {"groups": groups}


@app.get("/file/{id}/group/{group_name}")
async def get_dataset_in_group(
    redis: Annotated[redis.Redis, Depends(get_redis)],
    settings: Annotated[Settings, Depends(get_settings)],
    id: str,
    group_name: str,
):
    # list all the datasets in the group
    file_path = os.path.join(settings.allowed_hdf_path, id)
    with h5py.File(file_path, "r") as f:
        things = f[group_name]
        if not isinstance(things, h5py.Group):
            raise HTTPException(
                status_code=404, detail=f"Group {group_name} not found in file {id}"
            )
        datasets = list(things)
    return {"datasets": datasets}


@app.get("/file/{id}/group/{group_name}/dataset/{dataset_name}")
async def get_dataset(
    redis: Annotated[redis.Redis, Depends(get_redis)],
    settings: Annotated[Settings, Depends(get_settings)],
    id: str,
    group_name: str,
    dataset_name: str,
    latest_n_images: int = 10,
):
    # get the dataset
    file_path = os.path.join(settings.allowed_hdf_path, id)
    if file_path is None:
        raise HTTPException(status_code=404, detail="Dataset not present")


    with h5py.File(file_path, "r") as f:
        dset = guard_dataset_and_group(id, group_name, dataset_name, f)
        raw_data = dset[-latest_n_images:]
        stats_list = [(process_image_direct(img)) for img in raw_data]
        print(f"stats list: {stats_list}")
        fractions_list = calculate_fractions(np.array(stats_list))
        print(f"nice fractions: {fractions_list}")
        final_list: list[ImageDataMessage] = [
            ImageDataMessage(im_data=a) for a in fractions_list
        ]
        plot_message = PlotMessage(
            plot_id=dataset_name,
            type=MsgType.new_image_data,
            params=ImageData(values=fractions_list[0], aspect=1.0),
        )
        print(f"final list: {final_list}")

        print(f"stats: {stats_list}")
        return final_list  # ✅ FastAPI will return JSON array

    raise HTTPException(status_code=404, detail="Dataset not present")


def guard_dataset_and_group(
    file_id: str, group_name: str, dataset_name: str, f: h5py.File
):
    if group_name not in f:
        raise HTTPException(
            status_code=404, detail=f"Group {group_name} not found in file {file_id}"
        )
    group = f[group_name]
    if not isinstance(group, h5py.Group):
        raise HTTPException(
            status_code=404,
            detail=f"Group {group_name} is not a valid group in file {file_id}",
        )

    dset = group[dataset_name]
    if not isinstance(dset, h5py.Dataset):
        raise HTTPException(
            status_code=404,
            detail=f"Dataset {dataset_name} not found in group {group_name} of file {file_id}",  # noqa: E501
        )

    return dset


@app.get("/file/{id}/group/{group_name}/dataset/{dataset_name}/shape")
async def get_dataset_shape(
    redis: Annotated[redis.Redis, Depends(get_redis)],
    settings: Annotated[Settings, Depends(get_settings)],
    id: str,
    group_name: str,
    dataset_name: str,
):
    # get the dataset shape
    file_path = os.path.join(settings.allowed_hdf_path, id)
    with h5py.File(file_path, "r") as f:
        dset = guard_dataset_and_group(id, group_name, dataset_name, f)
        shape = dset.shape
    return {"shape": shape}


@app.post("/login/demo")
async def login_demo(
    response: Response,
    redis: Annotated[redis.Redis, Depends(get_redis)],
    settings: Annotated[Settings, Depends(get_settings)],
):
    session_id = str(uuid.uuid4())
    redis.hset(f"session:{session_id}", mapping={"status": "active"})
    redis.expire(f"session:{session_id}", settings.session_timeout_seconds)
    visr_dataset_name = "entry/instrument/detector/data"
    file_writing_path = "/workspaces/blue-histogramming/data"
    default_filename = "test2.hdf"
    redis.hset(
        f"session:{session_id}",
        mapping={
            "status": "active",
            "filepath": file_writing_path,
            "filename": default_filename,
            "dataset_name": visr_dataset_name,
        },
    )
    response.set_cookie(key="session_id", value=session_id, httponly=True)
    return {"message": "Session created", "session_id": session_id}


@app.websocket("/ws/{client_id}/dataset/{dataset_name}")
async def stream_dataset(
    session_manager: Annotated[SessionStateManager, Depends(get_session_manager)],
    client_id: str,
    websocket: WebSocket,
    dataset_name: str,
):
    # Example: accept the websocket and start observer
    await websocket.accept()
    # You can now use session_manager, e.g.:
    # session_manager.start_observer_for_session(session_id, folder, websocket)
    # ...rest of your logic...
    pass


@app.websocket("/ws/{client_id}/demo/{filename}")
async def demo_stream(
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
        session_id=client_id,
        folder=folder,
        websocket=websocket,
    )
    dataset_name: str = "entry/instrument/detector/data"
    filepath = "/workspaces/blue-histogramming/data"
    f: h5py.File = h5py.File(os.path.join(filepath, filename), "r")
    dset = guard_dataset_and_group(
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
    redis: Annotated[redis.Redis, Depends(get_redis)],
    settings: Annotated[Settings, Depends(get_settings)],
    response: Response,
    filepath: str,
    filename: str,
    dataset_name: str,
    session_id: str | None = Cookie(default=None),
):
    # check if path in allowed_hdf_path
    if not filepath.startswith(settings.allowed_hdf_path.as_posix()):
        raise HTTPException(
            status_code=403,
            detail=f"Path {filepath} is not allowed. Must start with {settings.allowed_hdf_path}",
        )
    # set the filepath for this session
    redis.hset(f"session:{session_id}", "filepath", filepath)
    # set the filename for this session
    redis.hset(f"session:{session_id}", "filename", filename)
    # set the dataset_name for this session
    redis.hset(f"session:{session_id}", "dataset_name", dataset_name)

    try:
        full_path = f"{state['filepath']}/{state['filename']}"
        f = h5py.File(full_path, "r", libver="latest", swmr=True)
        if dataset_name in f:
            pass
        else:
            raise HTTPException(status_code=404, detail="Dataset not found")
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to set dataset: {str(e)}"
        ) from e

    return {
        "message": "Dataset set successfully",
        "shape": state["dset"].shape if state["dset"] else None,  # type: ignore
    }


@app.post("/monitor_plan/")
def monitor_plan(
    redis: Annotated[redis.Redis, Depends(get_redis)],
    settings: Annotated[Settings, Depends(get_settings)],
    response: Response,
    plan_name: str,
    session_id: str | None = Cookie(default=None),
):
    """
    Monitor a plan by subscribing to a STOMP topic.
    This will make the stomp listener start listening for messages related to the plan for the start document
    """
    if not session_id:
        raise HTTPException(status_code=400, detail="Session ID is required")

    redis.hset(
        f"run_instance:{session_id}",
        mapping={
            "status": "active",
            "plan_name": plan_name,
            "session_id": session_id,
        },
    )
    return response


def run_server():
    """Start the FastAPI app and STOMP listener."""
    uvicorn.run(app, host="0.0.0.0", port=8002)
