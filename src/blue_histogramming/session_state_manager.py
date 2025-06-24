
import asyncio
import json
import os
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from logging import debug
from pathlib import Path
from threading import Thread
from typing import Annotated, Any, TypedDict, cast
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
from fastapi import (
    Cookie,
    Depends,
    FastAPI,
    HTTPException,
    Request,
    Response,
    WebSocket,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
from redis import Redis
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer




class SessionStateManager:
    redis: Redis
    path_watched: Path | None = None
    metadata: RunMetadata | None = None
    state: RunState
    connection: stomp.Connection | None = None

    def __init__(self, session_id: str, redis_client: Redis, settings: Settings):
        self.redis = redis_client
        self.memory_state = {}
        self.connection = self._start_stomp_listener(session_id, settings, redis_client)
        self.state = RunState(
            dataset=None,
            big_matrix=np.zeros((100, 100, 2), dtype=np.float32),
            descriptors=[],
        )

    def _start_stomp_listener(
        self, session_id: str, settings: Settings, redis_client: Redis
    ) -> stomp.Connection:
        conn = stomp.Connection([(settings.rmq_host, settings.rmq_port)])
        print("Connecting to STOMP broker...")
        conn.set_listener(
            session_id, STOMPListener(redis=redis_client, state=self.state)
        )
        try:
            # Read STOMP credentials from environment variables or .env
            stomp_user = os.environ.get("STOMP_USER", "user")
            stomp_password = os.environ.get("STOMP_PASSWORD", "password")
            conn.connect(stomp_user, stomp_password, wait=True)
        except Exception as e:
            print(f"Error: {e}")
            exit(1)
        print(f"trying to subscribe to topic, {STOP_TOPIC}")
        conn.subscribe(destination=STOP_TOPIC, id=1, ack="auto")
        return conn

    def set_numpy_state(self, session_id, numpy_data):
        self.memory_state[session_id] = numpy_data

    def get_numpy_state(self, session_id):
        return self.memory_state.get(session_id)

    def set_metadata(self, session_id, metadata: dict):
        self.redis.set(f"session:{session_id}:metadata", json.dumps(metadata))

    def get_metadata(self, session_id):
        data = self.redis.get(f"session:{session_id}:metadata")
        return json.loads(data) if data else {}

    def start_observer_for_session(self, session_id: str, folder: str):
        """Start a watchdog observer for this session and folder."""
        handler = FileSystemEventHandler()
        observer = Observer()
        observer.schedule(handler, folder, recursive=False)
        observer.start()
        # Store observer and handler in memory state
        self.memory_state[session_id] = self.memory_state.get(session_id, {})
        self.memory_state[session_id]["observer"] = observer
        self.memory_state[session_id]["handler"] = handler

    def stop_observer_for_session(self, session_id: str):
        """Stop and remove the observer for this session."""
        session = self.memory_state.get(session_id)
        if session and "observer" in session:
            observer = session["observer"]
            observer.stop()
            observer.join()
            del session["observer"]
            del session["handler"]

    def clear_session(self, session_id):
        self.stop_observer_for_session(session_id)
        self.memory_state.pop(session_id, None)
        self.redis.delete(f"session:{session_id}:metadata")

