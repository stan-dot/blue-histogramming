import os
from typing import Annotated

import h5py
import numpy as np
from davidia.models.messages import ImageData, ImageDataMessage, MsgType, PlotMessage
from fastapi import APIRouter, Cookie, Depends, HTTPException

from blue_histogramming.models import Settings
from blue_histogramming.session_state_manager import SessionStateManager
from blue_histogramming.utils import (
    calculate_fractions,
    list_hdf5_tree_of_file,
    process_image_direct,
)

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
    session_id: str | None = Cookie(default=None),
):
    file_path = os.path.join(settings.allowed_hdf_path, id)
    if file_path is None:
        raise HTTPException(status_code=404, detail="Dataset not present")

    with h5py.File(file_path, "r") as f:
        dset = guard_dataset_and_group(id, group_name, dataset_name, f)
        raw_data = dset[-latest_n_images:]
        # Vectorized processing
        stats_list = process_image_direct(raw_data)
        fractions_list = calculate_fractions(stats_list)
        final_list: list[ImageDataMessage] = [
            ImageDataMessage(im_data=a) for a in fractions_list
        ]
        return final_list

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
