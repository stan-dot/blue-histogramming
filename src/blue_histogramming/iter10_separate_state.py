"""
overall plan.
1. listen to stomp messages
2. parse the messages to get file path
3. from filepath parse data into davidia classes (ImageStats?)
4. let Davidia send that data over websocket to the client
"""

from pathlib import Path

import numpy as np
import stomp
from davidia.main import create_app
from event_model import EventDescriptor, RunStart
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware

from blue_histogramming.i10_types import RunInstance, RunMetadata, RunState

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
