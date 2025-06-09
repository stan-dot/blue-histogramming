from typing import TypedDict

from event_model import Event

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


# normalize based on events
# xvals = [e["data"][motor_names[-1]] for e in events]
# yvals = [e["data"][main_detector_name] for e in events]
# # normalize
# xvals = [x - xvals[0] for x in xvals]

# # todo consider a dataframe instead? maybe with polars https://pola.rs/
results = []
# # https://docs.pydantic.dev/latest/examples/files/#csv-files

# https://docs.python.org/3/library/errno.html

"""
Callback listener that processes collected documents and
fits detector data with curve :
<li>Single curve for 1-dimensional line scan,
<li> N curves for grid scans with shape NxM (M points per curve).

Uses scipy curve_fit function for curve fitting
fit_function -> function to be used during fitting
fit_bounds -> range for each parameter to be used when fitting.
    A tuple of (min, max) value for each parameter.
    e.g. for parameters a, b,c : ( (min a, max a), (min b, max b), (min c, max c))
"""
import stomp

CHANNEL = "/topic/public.worker.event"


class STOMPListener(stomp.PrintingListener):
    _conn: stomp.Connection | None

    def on_error(self, frame):
        print(f"Error: {frame.body}")

    def send_callback(self, data):
        """
        todo add the type
        """
        if self._conn is not None:
            self._conn.send(body=data, destination=CHANNEL)

    # todo need to parse the message
    # todo start streaming an hdf5 file if needed https://docs.h5py.org/en/latest/quick.html
    def on_message(self, frame):
        message = frame.body
        print(f"Received message: {message}")
        # todo parse the message into event model represeantations - use the stuff from above
        on_event(message, self.send_callback)


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


conn = start_stomp_connection()
if conn is None:
    print("Failed to connect to STOMP server.")
else:
    conn.subscribe(CHANNEL, id=1, ack="auto")
    conn.disconnect()
return
