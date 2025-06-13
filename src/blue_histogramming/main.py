import asyncio
import json
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import lru_cache
from logging import debug
from pathlib import Path
from threading import Thread
from typing import Annotated, Any, TypedDict
from urllib.parse import urlparse

import h5py
import numpy as np
import redis
import stomp
import uvicorn
from davidia.main import create_app
from davidia.models.messages import (
    Aspect,
    ImageData,
    ImageDataMessage,
    MsgType,
    PlotConfig,
    PlotMessage,
)
from event_model import Event, EventDescriptor, RunStart, StreamDatum, StreamResource
from fastapi import Depends, FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
from watchdog.observers import Observer


class Range(BaseModel):
    min_value: float
    max_value: float


class ColorSpectra(BaseModel):
    red: Range
    green: Range
    blue: Range


class RunMetadata(BaseModel):
    path: Path
    start: RunStart
    descriptor: EventDescriptor
    resource: StreamResource | None
    current_max_index: int = 0
    group_structure: dict[str, Any] | None = None
    rois: ColorSpectra | None = None
    shape: tuple[float, float]


# todo move to redis
@dataclass
class RunState:
    dataset: h5py.Dataset | None
    big_matrix: np.ndarray


# todo move to redis
class RunInstance:
    def __init__(self, meta: RunMetadata, state: RunState):
        self.meta = meta
        self.state = state


from pydantic import BaseModel
from datetime import datetime


class SessionData(BaseModel):
    session_id: str
    created_at: datetime
    user_agent: str | None = None
    metadata: dict = {}


class Settings(BaseSettings):
    hdf_path: str
    redis_host: str = "localhost"
    redis_port: int = 6379
    model_config = SettingsConfigDict(env_file=".env")
    channel = "topic/public.worker.event"


@lru_cache
def get_settings() -> Settings:
    return Settings(hdf_path=file_writing_path)


def get_redis() -> redis.Redis:
    settings = get_settings()

    return redis.Redis(host=settings.redis_host, port=settings.redis_port)


app = FastAPI()


@asynccontextmanager
async def lifespan(app: FastAPI):
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
    yield
    print("finishing the application run ")


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


events: list[Event] = []


class CurveFittingState(BaseModel):
    motor_names: list[str]
    main_detector_name: str
    shape: tuple[int, int]


state = CurveFittingState(motor_names=[], main_detector_name="", shape=(2, 100))


def on_event(event: Event):
    # process the event
    print(f"Processing event: {event}")
    events.append(event)
    if event.get("name") == "start":
        # start the curve fitting process
        # doc plan name has the plan_name variable, crucial to listen to
        print("Starting curve fitting process...")
        state["shape"] = event.get("shape") or (event.get("num_points", 0)) or (0, 0)
        # shape = event.get("shape") or (event.get("num_points", 0))
        print(f"Shape: {state.shape}")
        state["motor_names"] = event.get("motors", [])
        state["main_detector_name"] = event.get("detectors", [])[0]


def start_stomp_connection():
    # todo change this to use bluesky-stomp, like in blue-histogramming streaming context, using auth really - and this should be read from the params
    conn = stomp.Connection([("rmq", 61613)], auto_content_length=False)
    conn.set_listener("", STOMPListener())
    try:
        conn.connect("user", "password", wait=True)
    except stomp.exception.ConnectFailedException as e:  # type: ignore
        print(
            f"Connection failed. Please check your credentials and server address., error: {e}"  # noqa: E501
        )
        return None
    return conn


# todo read the topic from env vars to suit many deployments
STOP_TOPIC = "/queue/test"
# NOTE this defines a Davidia streaming app
davidia_app = create_app()

# CORS setup for development
davidia_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
state = {
    "filepath": "",
}

# https://fastapi.tiangolo.com/advanced/sub-applications/?h=mount#top-level-application
app.mount("/davidia", davidia_app)

# todo add to the per-session state
descriptors: set[str] = set()

# https://stackoverflow.com/questions/55311399/fastest-way-to-store-a-numpy-array-in-redis
# todo use redis for this
this_instance = RunInstance(
    meta=RunMetadata(
        path=Path(""),
        start=RunStart(uid="example_uid", time=0.0, scan_id=0, owner="example_owner"),
        descriptor=EventDescriptor(
            uid="example_descriptor_uid",
            time=0.0,
            data_keys={},
            run_start="example_run_start_uid",
        ),
        resource=None,
        rois=None,
        shape=(1, 1),
    ),
    state=RunState(
        big_matrix=np.array([]),
        dataset=None,
    ),
)


def start_stomp_listener():
    conn = stomp.Connection([("rmq", 61613)])
    print("Connecting to STOMP broker...")
    conn.set_listener("", STOMPListener(state_manager=this_instance))
    try:
        # todo read from venv
        conn.connect("user", "password", wait=True)
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
    print(f"trying to subscribe to topic, {STOP_TOPIC}")
    conn.subscribe(destination=STOP_TOPIC, id=1, ack="auto")


def process_image_direct(
    image: np.ndarray, use_roi: bool = False, spectra: ColorSpectra | None = None
) -> np.ndarray:
    """
    Efficiently processes the image to compute the sum for each color channel.
    Returns a 3-layer ndarray where each layer corresponds to r, g, b channels.
    """
    # Get dimensions
    h, _ = image.shape[0], image.shape[1]
    # todo divide by ROIs
    segment_height = h // 3

    # Vectorized approach to sum each segment
    r_sum = np.sum(image[:segment_height, :, :], axis=(0, 1))
    g_sum = np.sum(image[segment_height : 2 * segment_height, :, :], axis=(0, 1))
    b_sum = np.sum(image[2 * segment_height :, :, :], axis=(0, 1))

    if use_roi and spectra is not None:
        r_sum = np.sum(
            image[spectra.red.min_value : spectra.red.max_value, :, :], axis=(0, 1)
        )
        g_sum = np.sum(
            image[spectra.green.min_value : spectra.green.max_value, :, :], axis=(0, 1)
        )
        b_sum = np.sum(
            image[spectra.blue.min_value : spectra.blue.max_value :, :, :], axis=(0, 1)
        )

    # Stack results into a single 3-layer array
    return np.array([r_sum, g_sum, b_sum])


def calculate_fractions(stats_array: np.ndarray) -> np.ndarray:
    """
    Calculate the fractional values of the r, g, b sums for normalization.
    Returns a normalized array with the same shape as the input.
    """
    # Calculate min and max for each column (r, g, b, total)
    mins = np.min(stats_array, axis=0)
    maxs = np.max(stats_array, axis=0)

    # Avoid divide-by-zero by ensuring non-zero ranges
    ranges = np.where(maxs - mins == 0, 1, maxs - mins)

    # Vectorized normalization
    return (stats_array - mins) / ranges


def uri_to_path(uri: str) -> Path:
    print(f"URI: {uri}")
    parsed = urlparse(uri)
    print(f"Parsed URI: {parsed}")
    if parsed.scheme != "file":
        raise ValueError(f"Unsupported URI scheme: {parsed.scheme}")
    # Remove leading slash if running on Windows (drive letters)
    return Path(parsed.path)


def to_serializable(value: Any) -> str | int | float | list | None:
    """Convert NumPy types to standard Python types for JSON serialization."""
    if isinstance(value, np.bytes_):  # Convert bytes to string
        return value.decode()
    elif isinstance(value, np.integer):  # Convert np.int32, np.int64, etc. to int
        return int(value)
    elif isinstance(value, np.floating):  # Convert np.float32, np.float64 to float
        return float(value)
    elif isinstance(value, np.ndarray):  # Convert np.array to list
        return value.tolist()
    return value  # Assume already serializable


def list_hdf5_tree_of_file(file: h5py.File) -> dict[str, Any]:
    """
    Recursively lists all groups and datasets in an HDF5 file,
    returning a structured format.
    """
    structure = {"groups": []}

    def extract_from_entry(name: str, obj):
        obj_type = "Group" if isinstance(obj, h5py.Group) else "Dataset"
        entry = {
            "name": name,
            "type": obj_type,
            "items": [
                {"name": k, "value": to_serializable(v)} for k, v in obj.attrs.items()
            ],
        }
        structure["groups"].append(entry)

    file.visititems(extract_from_entry)

    return structure


class STOMPListener(stomp.ConnectionListener):
    def __init__(self, state_manager: RunInstance):
        self.state = state_manager
        self.handlers = {
            "start": self.handle_start,
            "descriptor": self.handle_descriptor,
            "stream_resource": self.handle_stream_resource,
            "stream_datum": self.handle_stream_datum,
        }
        self.descriptors = []

    def on_error(self, frame):
        print(f"Error: {frame.body}")

    def on_message(self, frame):
        print(f"Received message: {frame.body}")
        message = json.loads(frame.body)
        name = message.get("name")

        handler = self.handlers.get(name)
        if not handler:
            print(f"Unhandled message type: {name}")
            return

        handler(message["doc"])

    def handle_start(self, doc: dict):
        start_doc = RunStart(doc)  # type: ignore
        self.state.meta.shape = start_doc["shape"]  # type: ignore
        self.state.meta.rois = start_doc["color_rois"]  # type: ignore

    def handle_descriptor(self, doc: dict):
        descriptor = EventDescriptor(doc)
        print(f"Descriptor: {descriptor}")
        self.state.meta.descriptor = descriptor
        self.descriptors.append(descriptor["uid"])
        # todo add descriptors inject, maybe only here?

    def handle_stream_resource(self, doc: dict):
        resource = StreamResource(doc)
        filepath = uri_to_path(resource["uri"])
        self.state.meta.resource = resource
        self.state.meta.path = filepath

        with h5py.File(filepath, "r") as file:
            structures = list_hdf5_tree_of_file(file)
            debug(f"Structures: {structures}")
            self.state.meta.group_structure = structures

            dataset_path = resource["parameters"]["dataset"]
            dataset = file[dataset_path]

            if not isinstance(dataset, h5py.Dataset):
                print("Error: 'data' is not a dataset.")
                return

            self.state.state.dataset = dataset

    def handle_stream_datum(self, doc: dict):
        uid = doc["uid"]
        if uid not in self.descriptors:
            print("No descriptor associated with this datum")
            return

        datum = StreamDatum({"doc": doc})
        start_point = datum["indices"]["start"]
        stop = datum["indices"]["stop"]

        dataset = self.state.state.dataset
        if dataset is None:
            print("Error: No dataset found.")
            return

        x_bound, _ = self.state.meta.shape
        xcoor, ycoor = divmod(start_point, x_bound)

        raw_data = dataset[start_point:stop]
        if self.state.meta.rois is None:
            print("No region of interest specified")
            return

        raw_rgb = process_image_direct(raw_data, True, self.state.meta.rois)
        self.state.state.big_matrix[int(xcoor)][int(ycoor)][0] = raw_rgb
        self.state.state.big_matrix[:, :, 1] = calculate_fractions(
            self.state.state.big_matrix[:, :, 0]
        )

        relevant_axes = self.state.state.big_matrix[:, :, 1]  # future use?
        print(f"relevant axes: {relevant_axes}")
        msg = PlotMessage(
            plot_id=0,
            type=MsgType.new_image_data,
            plot_config=PlotConfig(),
            params={},
        )
        asyncio.run(app._plot_server.prepare_data(msg))  # type: ignore # noqa: SLF001
        asyncio.run(app._plot_server.send_next_message())  # type: ignore # noqa: SLF001
        # todo fix this


def tranform_dataset_into_dto(dataset: h5py.Dataset) -> ImageDataMessage:
    x_values = np.arange(dataset.shape[1])
    y_values = np.arange(dataset.shape[0])
    data = ImageData(values=dataset[...], aspect=Aspect.equal)
    plot_config = PlotConfig(
        x_label="x-axis",
        y_label="y-axis",
        x_values=x_values,
        y_values=y_values,
        title="image benchmarking plot",
    )
    return ImageDataMessage(im_data=data, plot_config=plot_config)


async def start_default_session():
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


if __name__ == "__main__":
    import uvicorn

    """Start the FastAPI app and STOMP listener."""
    thread_for_stomp = Thread(target=start_stomp_listener)
    thread_for_stomp.start()
    uvicorn.run(app, host="0.0.0.0", port=8002)
