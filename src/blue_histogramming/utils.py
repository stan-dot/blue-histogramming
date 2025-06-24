import asyncio
import json
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from functools import lru_cache
from logging import debug
from pathlib import Path
from typing import Annotated, Any, cast
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
from event_model import EventDescriptor, RunStart, StreamDatum, StreamResource
from fastapi import (
    Cookie,
    Depends,
    FastAPI,
    HTTPException,
    Response,
    WebSocket,
)
from fastapi.middleware.cors import CORSMiddleware
from redis import Redis

from blue_histogramming.models import ColorSpectra, RunMetadata, RunState, Settings
from blue_histogramming.session_state_manager import (
    SessionStateManager,
)


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
