import asyncio
import os
from functools import lru_cache
from threading import Thread
from typing import Annotated

import h5py
import numpy as np
import redis
from fastapi import Depends, HTTPException, WebSocket
from pydantic_settings import BaseSettings, SettingsConfigDict
from watchdog.observers import Observer

# hold in redis
visr_dataset_name = "entry/instrument/detector/data"
file_writing_path = "/dls/b01-1/data/2025/cm40661-1/bluesky"
default_filename = "-Stan-March-2025.hdf"
state = {
    "filepath": file_writing_path,
    "filename": default_filename,
    "dataset_name": visr_dataset_name,
    "dset": None,
    "file": None,
    "stats_array": [],
}


class Settings(BaseSettings):
    hdf_path: str
    redis_host: str = "localhost"
    redis_port: int = 6379
    model_config = SettingsConfigDict(env_file=".env")


@lru_cache
def get_settings() -> Settings:
    return Settings(hdf_path=file_writing_path)


def get_redis() -> redis.Redis:
    return redis.Redis(host="localhost", port=6379)


@app.on_event("startup")
async def startup():
    settings = get_settings()
    app.state.redis = get_redis()
    app.state.manager = manager

    print(settings)
    print(f"Starting notifier for {settings.hdf_path}")

    def start_notifier():
        observer = Observer()
        handler = EventHandler(app.state.redis, app.state.manager)
        observer.schedule(handler, settings.hdf_path, recursive=False)
        observer.start()
        observer.join()  # Keep it running

    Thread(target=start_notifier, daemon=True).start()


@app.get("/files")
async def get_files(
    redis: Annotated[dict, Depends(get_redis)],
    settings: Annotated[dict, Depends(get_settings)],
):
    # list all the files in the path
    files = [f for f in os.listdir(settings.hdf_path) if f.endswith(".hdf")]
    return {"files": files}


@app.get("/file/{id}/detail")
async def get_groups_in_file(
    redis: Annotated[dict, Depends(get_redis)],
    settings: Annotated[dict, Depends(get_settings)],
    id: str,
):
    # list all the groups in the file
    file_path = os.path.join(settings.hdf_path, id)
    groups = list_hdf5_tree(file_path)
    print(groups)
    return {"groups": groups}


@app.get("/file/{id}/group/{group_name}")
async def get_dataset_in_group(
    redis: Annotated[dict, Depends(get_redis)],
    settings: Annotated[dict, Depends(get_settings)],
    id: str,
    group_name: str,
):
    # list all the datasets in the group
    file_path = os.path.join(settings.hdf_path, id)
    with h5py.File(file_path, "r") as f:
        datasets = list(f[group_name])
    return {"datasets": datasets}


@app.get("/file/{id}/group/{group_name}/dataset/{dataset_name}")
async def get_dataset(
    redis: Annotated[dict, Depends(get_redis)],
    settings: Annotated[dict, Depends(get_settings)],
    id: str,
    group_name: str,
    dataset_name: str,
    latest_n_images: int = 10,
):
    # get the dataset
    file_path = os.path.join(settings.hdf_path, id)
    if file_path is None:
        raise HTTPException(status_code=404, detail="Dataset not present")

    with h5py.File(file_path, "r") as f:
        dset = f[group_name][dataset_name]
        raw_data = dset[-latest_n_images:]
        stats_list = [(process_image(img)) for img in raw_data]
        print(f"stats list: {stats_list}")
        fractions_list = calculate_fractions(stats_list)
        print(f"nice fractions: {fractions_list}")
        final_list = [ImageStatsDTO.from_array(list(a)) for a in fractions_list]
        print(f"final list: {final_list}")

        print(f"stats: {stats_list}")
        return final_list  # ✅ FastAPI will return JSON array

    raise HTTPException(status_code=404, detail="Dataset not present")


@app.post("/demo")
async def demo(
    redis: Annotated[dict, Depends(get_redis)],
    settings: Annotated[dict, Depends(get_settings)],
):
    # first blueapi call
    pass


@app.get("/file/{id}/group/{group_name}/dataset/{dataset_name}/shape")
async def get_dataset_shape(
    redis: Annotated[dict, Depends(get_redis)],
    settings: Annotated[dict, Depends(get_settings)],
    id: str,
    group_name: str,
    dataset_name: str,
):
    # get the dataset shape
    file_path = os.path.join(settings.hdf_path, id)
    with h5py.File(file_path, "r") as f:
        g = f[group_name]
        print(g)
        print(group_name)
        print(dataset_name)

        dset = g[dataset_name]
        print(dset)
        shape = dset.shape
    return {"shape": shape}


@app.websocket("/ws/{client_id}/dataset/{dataset_name}")
async def stream_dataset(
    settings: Annotated[dict, Depends(get_settings)],
    redis: Annotated[dict, Depends(get_redis)],
    client_id: str,
    websocket: WebSocket,
    dataset_name: str,
):
    pass


@app.websocket("/ws/{client_id}/demo/{filename}")
async def demo_stream(
    settings: Annotated[dict, Depends(get_settings)],
    redis: Annotated[dict, Depends(get_redis)],
    client_id: str,
    websocket: WebSocket,
    filename: str,
):
    """
    endpoint with hardcoded read
    """

    dataset_name: str = "entry/instrument/detector/data"
    filepath = "/dls/b01-1/data/2025/cm40661-1/bluesky"
    f: h5py.File = h5py.File(os.path.join(filepath, filename), "r")
    await manager.connect(websocket, client_id)
    data_points = f[dataset_name]
    try:
        while True:
            if data_points is None or len(data_points) == 0:
                print("❌ No data available, sending reset message...")
                await websocket.send_json([])
                await asyncio.sleep(1)
                continue  # Restart loop to wait for new data

            # Loop through each batch of data points
            for i in range(1, len(data_points) + 1):
                raw_data: np.ndarray = data_points[-i:]
                # print(f"raw data shape: {raw_data.shape}")
                stats_list = [(process_image(img)) for img in raw_data]
                # print(f"stats list: {stats_list}")
                fractions = calculate_fractions(stats_list)
                # print(f"fractions: {fractions}")
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
        await manager.disconnect(client_id)


@app.post("/set_dataset/")
def set_dataset(
    filepath: str | None = None,
    filename: str | None = None,
    dataset_name: str | None = None,
):
    state["filepath"] = filepath or state["filepath"]
    state["filename"] = filename or state["filename"]
    state["dataset_name"] = dataset_name or state["dataset_name"]
    print(state)
    dataset_name = dataset_name or state["dataset_name"]

    try:
        full_path = f"{state['filepath']}/{state['filename']}"
        print(f"full path: {full_path}")
        state["file"] = h5py.File(full_path, "r", libver="latest", swmr=True)
        if dataset_name in state["file"]:
            print(dataset_name)
            print(state["file"]["entry"])
            state["dset"] = state["file"][dataset_name]
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8002)
