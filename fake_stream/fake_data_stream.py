import asyncio
import json
from pathlib import Path

import stomp

CHANNEL = "/topic/public.worker.event"


class STOMPListener(stomp.PrintingListener):
    def on_error(self, frame):
        print(f"Error: {frame.body}")

    # todo need to parse the message and start streaming a file if needed https://docs.h5py.org/en/latest/quick.html
    def on_message(self, frame):
        message = frame.body
        print(f"Received message: {message}")


def start_stomp_connection():
    conn = stomp.Connection([("rmq", 61613)], auto_content_length=False)
    conn.set_listener("", STOMPListener())
    conn.connect("user", "password", wait=True)
    return conn


async def replay_events_from_json(
    json_path: Path, conn: stomp.Connection, delay: float = 0.05
):
    """
    Replay events from a JSON file to a STOMP connection.
    Args:
        json_path (Path): Path to the JSON file containing events.
        conn (stomp.Connection): STOMP connection object.
        delay (float): Delay between sending events in seconds.
    """
    if not json_path.exists():
        print(f"File {json_path} does not exist.")
        return
    if not json_path.is_file():
        print(f"{json_path} is not a file.")
        return
    if not json_path.suffix == ".json":
        print(f"{json_path} is not a JSON file.")
        return
    print(f"Reading events from {json_path}")
    with open(json_path) as f:
        data = json.load(f)
    length = len(data)
    print(f"Loaded {length} events from {json_path}")  
    if length == 0:
        print(f"No events found in {json_path}")
        return

    events = data.get("events", [])
    for event in events:
        print(f"🔁 Emitting: {event['name']} - {event.get('doc', {}).get('uid', '')}")
        message = json.dumps(event)
        print(f"🔁 tring to send message: {message}")
        conn.send(destination=CHANNEL, body=message)
        print(f"🔁 Sent event to STOMP: {message}")
        await asyncio.sleep(delay)


async def main():
    conn = start_stomp_connection()
    print("Connected to STOMP broker")
    await replay_events_from_json(Path("events-to-emit-6.json"), conn, 2)
    conn.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
