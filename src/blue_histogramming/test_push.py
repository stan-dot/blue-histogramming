from __future__ import annotations

import datetime

import pytest
from davidia.main import _create_bare_app
from davidia.models.messages import (
    LineData,
    LineParams,
    MsgType,
    PlotMessage,
)
from davidia.server.fastapi_utils import (
    ws_pack,
    ws_unpack,
)
from fastapi.testclient import TestClient
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_push_points():
    x = list(range(10))
    y = [j % 10 for j in x]
    time_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    line = LineData(
        key=time_id,
        line_params=LineParams(colour="purple"),
        x=x,
        y=y,
    )
    new_line = PlotMessage(
        plot_id="plot_0", type=MsgType.new_multiline_data, params=[line]
    )
    msg = ws_pack(new_line)
    headers = {
        "Content-Type": "application/x-msgpack",
        "Accept": "application/x-msgpack",
    }
    app = _create_bare_app()

    with TestClient(app) as client:
        with client.websocket_connect("/plot/99a81b01/plot_0"):
            async with AsyncClient(app=app, base_url="http://test") as ac:
                response = await ac.post("/push_data", content=msg, headers=headers)
            assert response.status_code == 200
            assert ws_unpack(response._content) == "data sent"
