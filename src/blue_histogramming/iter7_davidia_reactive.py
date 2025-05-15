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
from fastapi.middleware.cors import CORSMiddleware
from fastapi.websockets import WebSocket
from pydantic import BaseModel


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

"""
overall plan.
1. listen to stomp messages
2. parse the messages to get file path
3. from filepath parse data into davidia classes (ImageStats?)
4. let Davidia send that data over websocket to the client
"""

state = {
    "filepath": "",
}

active_websockets = set()


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
    resource: StreamResource
    current_max_index: int = 0


this_instance = RunInstance(
    path=Path(""),
    start=RunStart(uid="example_uid", time=0.0, scan_id=0, owner="example_owner"),
    descriptor=EventDescriptor(
        uid="example_descriptor_uid",
        time=0.0,
        data_keys={},
        run_start="example_run_start_uid",
    ),
    resource=null,
    # todo consider better the global state management
)


class STOMPListener(stomp.ConnectionListener):
    def on_error(self, frame):
        print(f"Error: {frame.body}")

    def on_message(self, frame):
        print(f"Received message: {frame.body}")
        message = frame.body
        print(f"Message: {message}")
        message = json.loads(message)
        if message["name"] == "descritptor":
            descriptor = EventDescriptor(message["doc"])
            print(f"Descriptor: {descriptor}")
            this_instance.descriptor = descriptor
            # todo check if the descriptor is already in the list
            descriptors.add(descriptor["uid"])

        if message["name"] == "stream_resource":
            resource = StreamResource(message["doc"])
            filepath = uri_to_path(resource["uri"])
            this_instance.resource = resource
            this_instance.path = filepath
            file = h5py.File(filepath, "r")

            structures = list_hdf5_tree_of_file(file)
            debug(f"Structures: {structures}")

            dataset_path = resource["parameters"]["dataset"]
            dataset = file[dataset_path]
            # todo what exactly to do with the dataset? when to send? should cache the whole thing in memory?

            if not isinstance(dataset, h5py.Dataset):
                print("Error: 'data' is not a dataset.")
                return

            image_data = tranform_dataset_into_dto(dataset)
            print(f"Image Data: {image_data}")

            # Send to WebSocket clients
            packed = ws_pack(image_data)
            print(f"Packed: {packed}")

            # todo change from this manual setup so that the davidia app will send the item
            for websocket in active_websockets:
                websocket.send_text(packed)

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
            # todo here we expand how much from the dataset we can read


def start_stomp_listener():
    conn = stomp.Connection([("rmq", 61613)])
    print("Connecting to STOMP broker...")
    conn.set_listener("", STOMPListener())
    try:
        conn.connect("user", "password", wait=True)
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
    # todo read the topic from env vars to suit many deployments
    print("trying to subscribe to topic")
    conn.subscribe(destination="/queue/test", id=1, ack="auto")


if __name__ == "__main__":
    from threading import Thread

    import uvicorn

    thread_for_stomp = Thread(target=start_stomp_listener)

    thread_for_stomp.start()
    uvicorn.run(app, port=8001)


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
