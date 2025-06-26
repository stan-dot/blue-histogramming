import os
from typing import Annotated, Any

import h5py
import httpx
import numpy as np
from davidia.models.messages import (
    ImageData,
    ImageDataMessage,
    MsgType,
    PlotMessage,
)
from fastapi import APIRouter, Cookie, Depends, HTTPException, Request, WebSocket

from blue_histogramming.main import get_session_manager
from blue_histogramming.models import Settings
from blue_histogramming.proxy import get_davidia_client
from blue_histogramming.session_state_manager import SessionStateManager
from blue_histogramming.utils import (
    calculate_fractions,
    list_hdf5_tree_of_file,
    process_image_direct,
)


def encode_ndarray(obj) -> dict[str, Any]:
    if isinstance(obj, np.ndarray):
        kind = obj.dtype.kind
        if kind == "i":  # reduce integer array byte size if possible
            vmin = obj.min() if obj.size > 0 else 0
            if vmin >= 0:
                kind = "u"
            else:
                vmax = obj.max() if obj.size > 0 else 0
                minmax_type = [np.min_scalar_type(vmin), np.min_scalar_type(vmax)]
                if minmax_type[1].kind == "u":
                    isize = minmax_type[1].itemsize
                    stype = np.dtype(f"i{isize}")
                    if isize == 8 and vmax > np.iinfo(stype).max:
                        minmax_type[1] = np.dtype(np.float64)
                    else:
                        minmax_type[1] = stype
                obj = obj.astype(np.promote_types(*minmax_type))
        if kind == "u":
            obj = obj.astype(np.min_scalar_type(obj.max() if obj.size > 0 else 0))
        obj = {
            "nd": True,
            "dtype": obj.dtype.str,
            "shape": obj.shape,
            "data": obj.data.tobytes(),
        }
    return obj


def ws_pack(obj) -> bytes | None:
    """Pack object for a websocket message

    Packs object by converting Pydantic models and ndarrays to dicts before
    using MessagePack
    """
    if isinstance(obj, BaseModel):
        obj = obj.model_dump(by_alias=True)
    return _mp_packb(obj, use_bin_type=True, default=encode_ndarray)


router = APIRouter()


def guard_dataset_and_group(
    file_id: str, group_name: str, dataset_name: str, f: h5py.File
):
    if group_name not in f:
        raise HTTPException(
            status_code=404, detail=f"Group {group_name} not found in file {file_id}"
        )
    group = f[group_name]
    if not isinstance(group, h5py.Group):
        raise HTTPException(
            status_code=404,
            detail=f"Group {group_name} is not a valid group in file {file_id}",
        )

    dset = group[dataset_name]
    if not isinstance(dset, h5py.Dataset):
        raise HTTPException(
            status_code=404,
            detail=f"Dataset {dataset_name} not found in group {group_name} of file {file_id}",
        )

    return dset


@router.get("/file/{id}/detail")
async def get_groups_in_file(
    session_manager: Annotated[SessionStateManager, Depends()],
    settings: Annotated[Settings, Depends()],
    id: str,
    session_id: str | None = Cookie(default=None),
):
    file_path = os.path.join(settings.allowed_hdf_path, id)
    if not os.path.exists(file_path):
        return {"groups": []}
    try:
        file: h5py.File = h5py.File(file_path, "r")
        groups = list_hdf5_tree_of_file(file)
        file.close()
    except Exception:
        groups = []
    return {"groups": groups}


@router.get("/file/{id}/group/{group_name}")
async def get_dataset_in_group(
    session_manager: Annotated[SessionStateManager, Depends()],
    settings: Annotated[Settings, Depends()],
    id: str,
    group_name: str,
    session_id: str | None = Cookie(default=None),
):
    file_path = os.path.join(settings.allowed_hdf_path, id)
    with h5py.File(file_path, "r") as f:
        things = f[group_name]
        if not isinstance(things, h5py.Group):
            raise HTTPException(
                status_code=404, detail=f"Group {group_name} not found in file {id}"
            )
        datasets = list(things)
    return {"datasets": datasets}


@router.get("/file/{id}/group/{group_name}/dataset/{dataset_name}")
async def get_dataset(
    session_manager: Annotated[SessionStateManager, Depends()],
    settings: Annotated[Settings, Depends()],
    id: str,
    group_name: str,
    dataset_name: str,
    latest_n_images: int = 10,
    client: httpx.AsyncClient = Depends(get_davidia_client),  # noqa: B008
    session_id: str | None = Cookie(default=None),
):
    file_path = os.path.join(settings.allowed_hdf_path, id)
    if file_path is None:
        raise HTTPException(status_code=404, detail="Dataset not present")

    with h5py.File(file_path, "r") as f:
        dset = guard_dataset_and_group(id, group_name, dataset_name, f)
        raw_data = dset[-latest_n_images:]
        stats_list = process_image_direct(raw_data)
        fractions_list = calculate_fractions(stats_list)
        final_list: list[ImageDataMessage] = [
            ImageDataMessage(im_data=a) for a in fractions_list
        ]

        # Msgpack encode the messages
        encoded = ws_pack([msg.model_dump() for msg in final_list])

        # Send to the davidia endpoint (example: /push_data)
        response = await client.post("/push_data", content=encoded)
        response.raise_for_status()
        return response.json()

    raise HTTPException(status_code=404, detail="Dataset not present")


@router.get("/file/{id}/group/{group_name}/dataset/{dataset_name}/shape")
async def get_dataset_shape(
    session_manager: Annotated[SessionStateManager, Depends()],
    settings: Annotated[Settings, Depends()],
    id: str,
    group_name: str,
    dataset_name: str,
    session_id: str | None = Cookie(default=None),
):
    file_path = os.path.join(settings.allowed_hdf_path, id)
    with h5py.File(file_path, "r") as f:
        dset = guard_dataset_and_group(id, group_name, dataset_name, f)
        shape = dset.shape
    return {"shape": shape}


@router.websocket("/ws/{client_id}/dataset/{dataset_name}")
async def stream_dataset(
    session_manager: Annotated[SessionStateManager, Depends(get_session_manager)],
    client_id: str,
    websocket: WebSocket,
    dataset_name: str,
):
    # Example: accept the websocket and start observer
    await websocket.accept()
    # You can now use session_manager, e.g.:
    # session_manager.start_observer_for_session(session_id, folder, websocket)
    # ...rest of your logic...
    pass
