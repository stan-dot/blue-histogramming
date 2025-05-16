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
from typing import Any, TypedDict
from urllib.parse import urlparse

import h5py
import numpy as np
import stomp
from davidia.main import create_app
from davidia.models import MsgType, PlotMessage
from event_model import EventDescriptor, RunStart, StreamDatum, StreamResource
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# todo read the topic from env vars to suit many deployments
STOP_TOPIC = "/queue/test"


# NOTE this is copied from the VISR repo
class ColorSpectra(TypedDict):
    red: tuple[float, float]
    green: tuple[float, float]
    blue: tuple[float, float]


def process_image_direct(image: np.ndarray, spectra: ColorSpectra) -> np.ndarray:
    """
    Efficiently processes the image to compute the sum for each color channel.
    Returns a 3-layer ndarray where each layer corresponds to r, g, b channels.
    """
    # Get dimensions
    h, _ = image.shape[0], image.shape[1]
    segment_height = h // 3

    # Vectorized approach to sum each segment
    r_sum = np.sum(image[:segment_height, :, :], axis=(0, 1))
    g_sum = np.sum(image[segment_height : 2 * segment_height, :, :], axis=(0, 1))
    b_sum = np.sum(image[2 * segment_height :, :, :], axis=(0, 1))

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


descriptors: set[str] = set()


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


class RunInstance(BaseModel):
    path: Path
    start: RunStart
    descriptor: EventDescriptor
    resource: StreamResource | None
    current_max_index: int = 0
    group_structure: dict[str, Any] | None = None
    dataset: h5py.Dataset | None = None
    big_matrix: np.ndarray
    # todo make something smarter than keeping 140MB in memory - yeah, ndarray is that
    rois: ColorSpectra | None = None
    shape: tuple[float, float]


# https://stackoverflow.com/questions/55311399/fastest-way-to-store-a-numpy-array-in-redis
# alternatively use redis for this
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
    rois=None,
    shape=(1, 1),
    big_matrix=np.array([]),
)


class STOMPListener(stomp.ConnectionListener):
    def on_error(self, frame):
        print(f"Error: {frame.body}")

    def on_message(self, frame):
        print(f"Received message: {frame.body}")
        message = frame.body
        print(f"Message: {message}")
        message = json.loads(message)

        if message["name"] == "start":
            start_doc = RunStart(message["doc"])
            # todo this will be ok but it's custom binding
            shape = start_doc["shape"]  # type: ignore
            rois = start_doc["color_rois"]  # type: ignore
            this_instance.rois = rois
            this_instance.shape = shape

        elif message["name"] == "descriptor":
            descriptor = EventDescriptor(message["doc"])
            print(f"Descriptor: {descriptor}")
            this_instance.descriptor = descriptor
            descriptors.add(descriptor["uid"])

        elif message["name"] == "stream_resource":
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

        elif message["name"] == "stream_datum":
            if message["doc"]["uid"] not in descriptors:
                print("no descriptor associated with this datum")
                return
            specific_slice = StreamDatum(message)
            print(f"Specific Slice: {specific_slice}")
            start_point = specific_slice["indices"][
                "start"
            ]  # here we know that it is only one step
            stop = specific_slice["indices"]["stop"]
            if this_instance.dataset is None:
                print("Error: No dataset found.")
                return
            # NOTE: relying on the notifications not SWMR mode
            x_bound, _ = this_instance.shape
            # todo this must be an int
            xcoor, ycoor = divmod(start_point, x_bound)

            raw_data: np.ndarray = this_instance.dataset[start_point:stop]
            if this_instance.rois is None:
                print("no region of interest specified in the start doc")

            raw_rgb = process_image_direct(raw_data, this_instance.rois)  # type: ignore
            # todo count this not assuming square shapes, but with real proportions
            this_instance.big_matrix[int(xcoor)][int(ycoor)][0] = (
                raw_rgb  # here the 3 raw values
            )
            this_instance.big_matrix[:][:][1] = calculate_fractions(
                this_instance.big_matrix[:][:][0]
            )
            # here all the normalized values, along the correct axes

            relevant_axes = this_instance.big_matrix[:][:][1]
            message = PlotMessage(
                0, MsgType.new_image_data, relevant_axes, plot_config={}
            )
            # forward to davidia
            # todo how to access that internal plotserver?
            asyncio.run(app._plot_server.prepare_data(message))  # noqa: SLF001
            asyncio.run(app._plot_server.send_next_message())  # noqa: SLF001


def start_stomp_listener():
    conn = stomp.Connection([("rmq", 61613)])
    print("Connecting to STOMP broker...")
    conn.set_listener("", STOMPListener())
    try:
        # todo read from venv
        conn.connect("user", "password", wait=True)
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
    print(f"trying to subscribe to topic, {STOP_TOPIC}")
    conn.subscribe(destination=STOP_TOPIC, id=1, ack="auto")


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
