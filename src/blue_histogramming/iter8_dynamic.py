"""
overall plan.
1. listen to stomp messages
2. parse the messages to get file path
3. from filepath parse data into davidia classes (ImageStats?)
4. let Davidia send that data over websocket to the client
"""

import asyncio
import json
from logging import debug
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import h5py
import numpy as np
import stomp
from davidia.main import create_app
from davidia.models.messages import Aspect, ImageData, ImageDataMessage, PlotConfig
from davidia.server.fastapi_utils import ws_pack
from event_model import EventDescriptor, RunStart, StreamDatum, StreamResource
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.websockets import WebSocket
from pydantic import BaseModel

# todo read the topic from env vars to suit many deployments
STOP_TOPIC = "/queue/test"


# ✅ Pydantic model for API response (converts np.uint64 to int)
class ImageStats(BaseModel):
    r: float
    g: float
    b: float
    total: float


def process_image_direct(image: np.ndarray) -> ImageStats:
    """
    Divide the image into 3 parts, compute sums for each part, and store in a dataclass.
    """
    # print(f"processing image: {image}")
    # print(f"shape: {image.shape}")
    h, _ = image.shape[0], image.shape[1]  # Get height and width of each 2D slice

    segment_height = h // 3
    # print(f"h and segment h: {h}, {segment_height}")

    # Divide image into three parts along the height dimension
    r_sum = np.sum(image[:, :segment_height])
    g_sum = np.sum(image[:, segment_height : 2 * segment_height])
    b_sum = np.sum(image[:, 2 * segment_height :])

    # Return results as a dataclass
    return ImageStats(r=r_sum, g=g_sum, b=b_sum, total=r_sum + g_sum + b_sum)


def calculate_fractions(
    stats_list: list[ImageStats],
) -> list[ImageStats]:
    # Extract all r, g, b, t values from the stats_list
    r_values = [stat.r for stat in stats_list]
    g_values = [stat.g for stat in stats_list]
    b_values = [stat.b for stat in stats_list]
    t_values = [stat.total for stat in stats_list]

    print(r_values, g_values, b_values, t_values)
    # Find min and max for each of the properties (r, g, b, t)
    r_min, r_max = min(map(int, r_values)), max(map(int, r_values))
    g_min, g_max = min(map(int, g_values)), max(map(int, g_values))
    b_min, b_max = min(map(int, b_values)), max(map(int, b_values))
    t_min, t_max = min(map(int, t_values)), max(map(int, t_values))

    # If any property min == max, fractions for that property will be 0
    if r_min == r_max:
        r_fractions = [0] * len(stats_list)
    else:
        r_fractions = [(stat.r - r_min) / (r_max - r_min) for stat in stats_list]

    if g_min == g_max:
        g_fractions = [0] * len(stats_list)
    else:
        g_fractions = [(stat.g - g_min) / (g_max - g_min) for stat in stats_list]

    if b_min == b_max:
        b_fractions = [0] * len(stats_list)
    else:
        b_fractions = [(stat.b - b_min) / (b_max - b_min) for stat in stats_list]

    if t_min == t_max:
        t_fractions = [0] * len(stats_list)
    else:
        t_fractions = [(stat.total - t_min) / (t_max - t_min) for stat in stats_list]

    fractions = [
        ImageStats(
            r=r_fractions[i], g=g_fractions[i], b=b_fractions[i], total=t_fractions[i]
        )
        for i in range(len(stats_list))
    ]

    print(fractions)
    return fractions


def get_dtos_from_dataset(
    dataset: h5py.Dataset, start: int, stop: int
) -> list[ImageStats]:
    raw_data: np.ndarray = dataset[start:stop]
    stats_list = [(process_image_direct(image)) for image in raw_data]
    fractions = calculate_fractions(stats_list)
    return fractions


def uri_to_path(uri: str) -> Path:
    print(f"URI: {uri}")
    parsed = urlparse(uri)
    print(f"Parsed URI: {parsed}")
    if parsed.scheme != "file":
        raise ValueError(f"Unsupported URI scheme: {parsed.scheme}")
    # Remove leading slash if running on Windows (drive letters)
    return Path(parsed.path)


# NOTE this defines a Davidia streaming app
app = create_app()


# CORS setup for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
state = {
    "filepath": "",
}

active_websockets: set[WebSocket] = set()


descriptors: set[str] = set()


async def send_to_clients(image_data: ImageDataMessage):
    """Send the transformed image data to all connected WebSocket clients."""
    message_json = json.dumps(
        image_data.model_dump()
    )  # FastAPI's Pydantic models support `model_dump`
    for websocket in active_websockets:
        await websocket.send_text(message_json)


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
    else:
        return value  # Assume already serializable


def list_hdf5_tree_of_file(file: h5py.File) -> dict[str, Any]:
    """Recursively lists all groups and datasets in an HDF5 file, returning a structured format."""
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


class RunInstance(BaseModel):
    path: Path
    start: RunStart
    descriptor: EventDescriptor
    resource: StreamResource | None
    current_max_index: int = 0
    group_structure: dict[str, Any] | None = None
    dataset: h5py.Dataset | None = None
    events_record: dict[int, list[ImageStats]] = {}
    # todo make something smarter than keeping 140MB in memory


this_instance = RunInstance(
    path=Path(""),
    start=RunStart(uid="example_uid", time=0.0, scan_id=0, owner="example_owner"),
    descriptor=EventDescriptor(
        uid="example_descriptor_uid",
        time=0.0,
        data_keys={},
        run_start="example_run_start_uid",
    ),
    resource=None,
)


class STOMPListener(stomp.ConnectionListener):
    def on_error(self, frame):
        print(f"Error: {frame.body}")

    def on_message(self, frame):
        print(f"Received message: {frame.body}")
        message = frame.body
        print(f"Message: {message}")
        message = json.loads(message)
        if message["name"] == "descriptor":
            descriptor = EventDescriptor(message["doc"])
            print(f"Descriptor: {descriptor}")
            this_instance.descriptor = descriptor
            descriptors.add(descriptor["uid"])

        if message["name"] == "stream_resource":
            resource = StreamResource(message["doc"])
            filepath = uri_to_path(resource["uri"])
            this_instance.resource = resource
            this_instance.path = filepath
            file = h5py.File(filepath, "r")

            structures = list_hdf5_tree_of_file(file)
            debug(f"Structures: {structures}")
            this_instance.group_structure = structures

            dataset_path = resource["parameters"]["dataset"]
            dataset = file[dataset_path]

            if not isinstance(dataset, h5py.Dataset):
                print("Error: 'data' is not a dataset.")
                return
            this_instance.dataset = dataset

        if message["name"] == "stream_datum":
            if message["doc"]["uid"] not in descriptors:
                print("no descriptor associated with this datum")
                return
            specific_slice = StreamDatum(message)
            print(f"Specific Slice: {specific_slice}")
            start = specific_slice["indices"][
                "start"
            ]  # here we know that it is only one step
            stop = specific_slice["indices"]["stop"]
            # todo for now always start at 0
            if this_instance.dataset is None:
                print("Error: No dataset found.")
                return
            # NOTE: relying on the notifications not SWMR mode
            list_of_image_stats = get_dtos_from_dataset(this_instance.dataset, 0, stop)
            this_instance.events_record[stop] = list_of_image_stats
            packed_image_stats = ws_pack(list_of_image_stats)
            packed_image_objects = ws_pack(
                tranform_dataset_into_dto(this_instance.dataset[start:stop])
            )

            # todo should use davidia format to send this? or just the custom websocket solution?
            self._forward_slice_to_websockets(start, stop)

    def _forward_slice_to_websockets(self, start, stop):
        if this_instance.dataset is None:
            print("Error: No dataset found.")
            return
        image_data = tranform_dataset_into_dto(this_instance.dataset[start:stop])
        # todo are images that indexable?
        print(f"Image Data: {image_data}")

        # Send to WebSocket clients
        packed = ws_pack(image_data)
        print(f"Packed: {packed}")

        if packed is None:
            print("Packed is None")
            return
            # todo maybe change from sending here to davidia sending this
        for websocket in active_websockets:
            asyncio.run(websocket.send_bytes(packed))


def start_stomp_listener():
    conn = stomp.Connection([("rmq", 61613)])
    print("Connecting to STOMP broker...")
    conn.set_listener("", STOMPListener())
    try:
        conn.connect("user", "password", wait=True)
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
    print(f"trying to subscribe to topic, {STOP_TOPIC}")
    conn.subscribe(destination=STOP_TOPIC, id=1, ack="auto")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for clients to receive live image data."""
    await websocket.accept()
    active_websockets.add(websocket)
    try:
        while True:
            await websocket.receive_text()  # Keep connection alive
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        active_websockets.remove(websocket)


@app.get("/get_dataset_shape/")
def get_dataset_shape():
    if this_instance.dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not initialized")
    try:
        return {"shape": this_instance.dataset.shape}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get dataset shape: {str(e)}"
        ) from e


if __name__ == "__main__":
    from threading import Thread

    import uvicorn

    thread_for_stomp = Thread(target=start_stomp_listener)

    thread_for_stomp.start()
    uvicorn.run(app, port=8001)
