import asyncio
import json
from datetime import datetime
from logging import debug
from typing import cast

import h5py
import stomp
from davidia.models.messages import (
    MsgType,
    PlotConfig,
    PlotMessage,
)
from event_model import EventDescriptor, RunStart, StreamDatum, StreamResource
from redis import Redis

from blue_histogramming.models import RunState
from blue_histogramming.utils import (
    calculate_fractions,
    list_hdf5_tree_of_file,
    process_image_direct,
    uri_to_path,
)


class STOMPListener(stomp.ConnectionListener):
    state: RunState
    redis: Redis

    def __init__(self, redis: Redis, state: RunState):
        self.state = state
        self.redis = redis
        self.handlers = {
            "start": self.handle_start,
            "descriptor": self.handle_descriptor,
            "stream_resource": self.handle_stream_resource,
            "stream_datum": self.handle_stream_datum,
        }

    def on_error(self, frame):
        print(f"Error: {frame.body}")

    def on_message(self, frame):
        print(f"Received message: {frame.body}")
        message = json.loads(frame.body)
        name = message.get("name")

        handler = self.handlers.get(name)
        if not handler:
            print(f"Unhandled message type: {name}")
            return

        handler(message["doc"])

    def handle_start(self, doc: dict):
        start_doc = cast(RunStart, doc)
        print(f"Start document: {start_doc}")
        self.redis.hset(
            f"run_instance:{start_doc['uid']}",
            mapping={
                "status": "active",
                "start_time": datetime.fromtimestamp(start_doc["time"]).isoformat(),
                "plan_name": start_doc["plan_name"],  # type: ignore
                "shape": start_doc["shape"],  # type: ignore
                "rois": json.dumps(start_doc["color_rois"], default=to_serializable),  # type: ignore  # noqa: F821
                "uid": start_doc["uid"],
            },
        )

    def handle_descriptor(self, doc: dict):
        descriptor = cast(EventDescriptor, doc)
        print(f"Descriptor: {descriptor}")
        self.state.descriptors.append(descriptor)

    def handle_stream_resource(self, doc: dict):
        resource = cast(StreamResource, doc)
        filepath = uri_to_path(resource["uri"])
        self.metadata.resource = resource
        self.state.metadata.path = filepath

        with h5py.File(filepath, "r") as file:
            structures = list_hdf5_tree_of_file(file)
            debug(f"Structures: {structures}")
            self.state.meta.group_structure = structures

            dataset_path = resource["parameters"]["dataset"]
            dataset = file[dataset_path]

            if not isinstance(dataset, h5py.Dataset):
                print("Error: 'data' is not a dataset.")
                return

            self.state.state.dataset = dataset

    def handle_stream_datum(self, doc: dict):
        uid = doc["uid"]
        if uid not in self.state.descriptors:
            print("No descriptor associated with this datum")
            return

        datum = cast(StreamDatum, doc)
        start_point = datum["indices"]["start"]
        stop = datum["indices"]["stop"]

        dataset = self.state.dataset
        if dataset is None:
            print("Error: No dataset found.")
            return

        x_bound, _ = self.state.meta.shape
        xcoor, ycoor = divmod(start_point, x_bound)

        raw_data = dataset[start_point:stop]
        if self.state.meta.rois is None:
            print("No region of interest specified")
            return

        raw_rgb = process_image_direct(raw_data, True, self.state.meta.rois)
        self.state.state.big_matrix[int(xcoor)][int(ycoor)][0] = raw_rgb
        self.state.state.big_matrix[:, :, 1] = calculate_fractions(
            self.state.state.big_matrix[:, :, 0]
        )

        relevant_axes = self.state.big_matrix[:, :, 1]  # future use?
        print(f"relevant axes: {relevant_axes}")
        msg = PlotMessage(
            plot_id="0",
            type=MsgType.new_image_data,
            plot_config=PlotConfig(),
            params="",
        )
        asyncio.run(app._plot_server.prepare_data(msg))  # type: ignore # noqa: SLF001
        asyncio.run(app._plot_server.send_next_message())  # type: ignore # noqa: SLF001
