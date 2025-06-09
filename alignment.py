### main.py
import asyncio
from message_listener import listen_for_events

async def main():
    await listen_for_events()

if __name__ == "__main__":
    asyncio.run(main())


### config.py
import os
from pydantic import BaseSettings

class Settings(BaseSettings):
    MESSAGE_BUS_URL: str = os.getenv("MESSAGE_BUS_URL", "nats://localhost:4222")
    GRAPHQL_ENDPOINT: str = os.getenv("GRAPHQL_ENDPOINT", "http://localhost:8000/graphql")
    CONFIG_SERVER_URL: str = os.getenv("CONFIG_SERVER_URL", "http://localhost:9000/update")
    STORAGE_PATH: str = os.getenv("STORAGE_PATH", "/data")

settings = Settings()


### message_listener.py
import asyncio
from event_aggregator import handle_event_stream

async def listen_for_events():
    print("Listening to message bus...")
    await handle_event_stream()


### event_aggregator.py
import asyncio
from data_encoder import dataframe_to_base64_csv
from graphql_client import submit_to_graphql, poll_job
from config_updater import update_config
import pandas as pd

RUN_BUFFERS = {}

async def handle_event_stream():
    # Simulated event source
    for i in range(3):
        run_id = "run123"
        event = {"timestamp": f"2024-01-01T00:0{i}:00Z", "temperature": 20+i, "humidity": 50-i}
        await handle_event(run_id, event)
        await asyncio.sleep(1)

    # Simulate end event
    await handle_event("run123", {"type": "END"})

async def handle_event(run_id: str, event: dict):
    if event.get("type") == "END":
        df = pd.DataFrame(RUN_BUFFERS.pop(run_id, []))
        b64 = dataframe_to_base64_csv(df)
        job_id = await submit_to_graphql(b64)
        artifact = await poll_job(job_id)
        await update_config(artifact)
    else:
        RUN_BUFFERS.setdefault(run_id, []).append(event)


### data_encoder.py
import pandas as pd
import base64
from io import BytesIO

def dataframe_to_base64_csv(df: pd.DataFrame) -> str:
    buffer = BytesIO()
    df.to_csv(buffer, index=False)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def base64_csv_to_dataframe(b64_string: str) -> pd.DataFrame:
    decoded = base64.b64decode(b64_string)
    return pd.read_csv(BytesIO(decoded))


### graphql_client.py
import asyncio
import random

async def submit_to_graphql(b64_data: str) -> str:
    print("Submitting to GraphQL...")
    await asyncio.sleep(1)  # Simulate latency
    return f"job_{random.randint(1000, 9999)}"

async def poll_job(job_id: str) -> str:
    print(f"Polling job {job_id}...")
    for _ in range(5):
        await asyncio.sleep(1)
        if random.random() > 0.5:
            return f"artifact_for_{job_id}.json"
    return f"timeout_{job_id}"


### config_updater.py
import asyncio

async def update_config(artifact_path: str):
    print(f"Updating config with {artifact_path}...")
    await asyncio.sleep(0.5)
    print("Config updated.")


### storage.py
# Placeholder for future file download/upload if needed


### utils.py
# General utilities or logging setup could go here
