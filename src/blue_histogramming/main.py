from pathlib import Path
from typing import TypedDict

import numpy as np
import stomp
from davidia.main import create_app
from event_model import Event, EventDescriptor, RunStart
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware

from blue_histogramming.stomp_listener import STOMPListener
from blue_histogramming.types import RunInstance, RunMetadata, RunState

events: list[Event] = []


class AppState(TypedDict):
    motor_names: list[str]
    main_detector_name: str
    shape: tuple[int, int]


state: AppState = {"motor_names": [], "main_detector_name": "", "shape": (2, 100)}


def on_event(event: Event):
    # process the event
    print(f"Processing event: {event}")
    events.append(event)
    if event.get("name") == "start":
        # start the curve fitting process
        print("Starting curve fitting process...")
        state["shape"] = event.get("shape") or (event.get("num_points", 0)) or (0, 0)
        # shape = event.get("shape") or (event.get("num_points", 0))
        print(f"Shape: {state.shape}")
        state["motor_names"] = event.get("motors", [])
        state["main_detector_name"] = event.get("detectors", [])[0]


def start_stomp_connection():
    # todo change this to use bluesky-stomp, like in blue-histogramming streaming context, using auth really - and this should be read from the params
    conn = stomp.Connection([("rmq", 61613)], auto_content_length=False)
    conn.set_listener("", STOMPListener())
    try:
        conn.connect("user", "password", wait=True)
    except stomp.exception.ConnectFailedException as e:  # type: ignore
        print(
            f"Connection failed. Please check your credentials and server address., error: {e}"  # noqa: E501
        )
        return None
    return conn


# conn = start_stomp_connection()
# if conn is None:
#     print("Failed to connect to STOMP server.")
# else:
#     conn.subscribe(CHANNEL, id=1, ack="auto")
#     conn.disconnect()
# return

# todo read the topic from env vars to suit many deployments
# todo read the topic from env vars to suit many deployments
STOP_TOPIC = "/queue/test"
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

# https://stackoverflow.com/questions/55311399/fastest-way-to-store-a-numpy-array-in-redis
# alternatively use redis for this
this_instance = RunInstance(
    meta=RunMetadata(
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
    ),
    state=RunState(
        big_matrix=np.array([]),
        dataset=None,
    ),
)


def start_stomp_listener():
    conn = stomp.Connection([("rmq", 61613)])
    print("Connecting to STOMP broker...")
    conn.set_listener("", STOMPListener(state_manager=this_instance))
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
    if this_instance.state.dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not initialized")
    try:
        return {"shape": this_instance.state.dataset.shape}
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
