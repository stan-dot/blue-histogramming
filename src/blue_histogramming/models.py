from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from event_model import EventDescriptor, RunStart, StreamResource
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


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


class RunState(BaseModel):
    dataset: h5py.Dataset | None
    big_matrix: np.ndarray
    descriptors: list[EventDescriptor] = []

    class Config:
        arbitrary_types_allowed = True


class SessionData(BaseModel):
    session_id: str
    created_at: datetime
    user_agent: str | None = None
    metadata: dict = {}


class Settings(BaseSettings):
    allowed_hdf_path: Path = Path("/dls/b01-1/data/2025/cm40661-1/bluesky")
    redis_host: str = "dragonfly"
    redis_port: int = 6379
    rmq_host: str = "rmq"
    rmq_port: int = 61613
    model_config = SettingsConfigDict(env_file=".env")
    channel: str = "topic/public.worker.event"
    session_timeout_seconds: int = 900  # 15 minutes
