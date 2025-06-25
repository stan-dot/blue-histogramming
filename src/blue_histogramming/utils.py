from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import h5py
import numpy as np
from davidia.models.messages import (
    Aspect,
    ImageData,
    ImageDataMessage,
    PlotConfig,
)

from blue_histogramming.models import ColorSpectra


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


def process_image_direct(
    image: np.ndarray, use_roi: bool = False, spectra: ColorSpectra | None = None
) -> np.ndarray:
    """
    Efficiently processes the image to compute the sum for each color channel.
    Returns a 3-layer ndarray where each layer corresponds to r, g, b channels.
    """
    # Get dimensions
    h, _ = image.shape[0], image.shape[1]
    DEFAULT_SEGMENTS_NUMBER = 3
    segment_height = h // DEFAULT_SEGMENTS_NUMBER

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
