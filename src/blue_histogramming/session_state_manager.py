import asyncio
import json
import os
from pathlib import Path

import numpy as np
import stomp
from redis import Redis
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from blue_histogramming.models import RunMetadata, RunState, Settings
from blue_histogramming.stomp_listener import STOMPListener


class WebSocketFileSystemEventHandler(FileSystemEventHandler):
    def __init__(self, websocket, loop):
        super().__init__()
        self.websocket = websocket
        self.loop = loop

    def on_any_event(self, event):
        # Send event info to websocket in the main event loop
        msg = {"event": event.event_type, "src_path": event.src_path}
        asyncio.run_coroutine_threadsafe(self.websocket.send_json(msg), self.loop)


class SessionStateManager:
    redis: Redis
    path_watched: Path | None = None
    metadata: RunMetadata | None = None
    state: RunState
    connection: stomp.Connection | None = None
    settings: Settings
    session_id: str

    def __init__(self, session_id: str, redis_client: Redis, settings: Settings):
        self.session_id = session_id
        self.redis = redis_client
        self.settings = settings
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
        topic = settings.channel
        print(f"trying to subscribe to topic, {topic}")
        conn.subscribe(destination=topic, id=1, ack="auto")
        return conn

    def set_numpy_state(self, numpy_data):
        self.memory_state[self.session_id] = numpy_data

    def get_numpy_state(self):
        return self.memory_state.get(self.session_id)

    def set_metadata(self, metadata: dict):
        self.redis.set(f"session:{self.session_id}:metadata", json.dumps(metadata))

    def get_metadata(self):
        data = self.redis.get(f"session:{self.session_id}:metadata")
        return json.loads(data) if data else {}

    def start_observer_for_session(self, folder: str, websocket=None):
        """Start a watchdog observer for this session and folder, optionally notifying a websocket."""
        loop = asyncio.get_event_loop()
        if websocket:
            handler = WebSocketFileSystemEventHandler(websocket, loop)
        else:
            handler = FileSystemEventHandler()
        observer = Observer()
        observer.schedule(handler, folder, recursive=False)
        observer.start()
        # Store observer, handler, and websocket in memory state
        self.memory_state[self.session_id] = self.memory_state.get(self.session_id, {})
        self.memory_state[self.session_id]["observer"] = observer
        self.memory_state[self.session_id]["handler"] = handler
        if websocket:
            self.memory_state[self.session_id]["websocket"] = websocket

    def stop_observer_for_session(self):
        """Stop and remove the observer for this session."""
        session = self.memory_state.get(self.session_id)
        if session and "observer" in session:
            observer = session["observer"]
            observer.stop()
            observer.join()
            del session["observer"]
            del session["handler"]

    def clear_session(self):
        self.stop_observer_for_session()
        self.memory_state.pop(self.session_id, None)
        self.redis.delete(f"session:{self.session_id}:metadata")

    def set_dataset(self, dataset_name: str):
        # set this session dataset and path
        self.redis.hset(f"session:{self.session_id}", mapping={"status": "active"})
        self.redis.set(f"session:{self.session_id}:dataset", dataset_name)

    def get_session_data(self) -> dict:
        """Get all session data as a dict (decoding bytes to str)."""
        data = self.redis.hgetall(f"session:{self.session_id}")
        if not data:
            return {}
        return {k.decode(): v.decode() for k, v in data.items()}

    def set_session_data(self, mapping: dict):
        """Set multiple session fields at once."""
        self.redis.hset(f"session:{self.session_id}", mapping=mapping)

    def new_session(self, metadata: RunMetadata):
        """Initialize a new session with metadata."""
        self.metadata = metadata
        self.set_metadata(metadata.dict())
        self.set_session_data(
            {"status": "active", "created_at": str(metadata.start.time)}
        )
        self.redis.expire(
            f"session:{self.session_id}", self.settings.session_timeout_seconds
        )
        self.state = RunState(
            dataset=None,
            big_matrix=np.zeros((100, 100, 2), dtype=np.float32),
            descriptors=[],
        )
        self.memory_state[self.session_id] = {"state": self.state}
