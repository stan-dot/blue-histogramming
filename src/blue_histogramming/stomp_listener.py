import asyncio
import json
from logging import debug
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import h5py
import numpy as np
import stomp
from davidia.models.messages import (
    Aspect,
    ImageData,
    ImageDataMessage,
    MsgType,
    PlotConfig,
    PlotMessage,
)
from event_model import EventDescriptor, RunStart, StreamDatum, StreamResource

from blue_histogramming.types import ColorSpectra, RunInstance

# todo read from settgins
CHANNEL = "/topic/public.worker.event"


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
