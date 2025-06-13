
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

import h5py
import numpy as np
from event_model import EventDescriptor, RunStart, StreamResource
from pydantic import BaseModel


# NOTE this is copied from the VISR repo
class ColorSpectra(TypedDict):
    red: tuple[float, float]
    green: tuple[float, float]
    blue: tuple[float, float]


class RunMetadata(BaseModel):
    path: Path
    start: RunStart
    descriptor: EventDescriptor
    resource: StreamResource | None
    current_max_index: int = 0
    group_structure: dict[str, Any] | None = None
    rois: ColorSpectra | None = None
    shape: tuple[float, float]


@dataclass
class RunState:
    dataset: h5py.Dataset | None
    big_matrix: np.ndarray


class RunInstance:
    def __init__(self, meta: RunMetadata, state: RunState):
        self.meta = meta
        self.state = state

